"""
PlantDoc Class-conditional Latent Diffusion with CFG
Based on latent_diffusion_cond.py but using PlantDoc dataset
"""

import os
import logging
import time
import math
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as tvu
from PIL import Image

from models.diffusion import Model
from models.ema import EMAHelper
from datasets.plantdoc import PlantDoc

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from diffusers import AutoencoderKL
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False

try:
    from cleanfid import fid
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class PlantDocDiffusion:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Setup betas
        if config.diffusion.beta_schedule == "cosine":
            betas = cosine_beta_schedule(config.diffusion.num_diffusion_timesteps)
        else:
            betas = linear_beta_schedule(
                config.diffusion.num_diffusion_timesteps,
                config.diffusion.beta_start,
                config.diffusion.beta_end
            )

        self.betas = betas.float().to(self.device)
        self.num_timesteps = len(betas)

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Load VAE
        if VAE_AVAILABLE:
            vae_id = getattr(config.vae, 'model_id', 'stabilityai/sd-vae-ft-mse')
            logging.info(f"Loading VAE: {vae_id}")
            self.vae = AutoencoderKL.from_pretrained(vae_id).to(self.device)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae_scale = 0.18215
        else:
            raise RuntimeError("diffusers required for VAE")

        # Class dropout prob for CFG
        self.class_dropout = getattr(config.model, 'class_dropout', 0.1)
        self.num_classes = getattr(config.model, 'num_classes', 28)

    @torch.no_grad()
    def encode(self, x):
        """Encode images [0,1] to latent"""
        x = 2 * x - 1
        return self.vae.encode(x).latent_dist.sample() * self.vae_scale

    @torch.no_grad()
    def decode(self, z):
        """Decode latent to images [0,1]"""
        z = z / self.vae_scale
        x = self.vae.decode(z).sample
        return ((x + 1) / 2).clamp(0, 1)

    def train(self):
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        # Wandb
        use_wandb = WANDB_AVAILABLE and getattr(config.wandb, 'enabled', False)
        if use_wandb:
            try:
                wandb.init(
                    project=getattr(config.wandb, 'project', 'ddim-plantdoc'),
                    name=getattr(config.wandb, 'name', None),
                    tags=getattr(config.wandb, 'tags', []),
                    config={
                        'batch_size': config.training.batch_size,
                        'lr': config.optim.lr,
                        'n_iters': config.training.n_iters,
                        'num_classes': self.num_classes,
                        'num_timesteps': self.num_timesteps,
                        'augment_prob': config.data.augment_prob,
                    }
                )
                logging.info("Wandb initialized")
            except Exception as e:
                logging.warning(f"Wandb init failed: {e}")
                use_wandb = False

        # Dataset
        train_dataset = PlantDoc(
            root=config.data.data_root,
            split='train',
            config=config,
            augment_prob=getattr(config.data, 'augment_prob', 0.25),
        )
        test_dataset = PlantDoc(
            root=config.data.data_root,
            split='test',
            config=config,
            augment_prob=0.0,  # No augmentation for test
        )

        # Weighted sampler for class imbalance
        if getattr(config.data, 'use_weighted_sampler', False):
            sampler = train_dataset.get_sampler()
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Save real images for FID
        self._prepare_real_images(test_dataset, 'test')

        # Model
        orig_img_size = config.data.image_size
        config.data.image_size = config.data.latent_size
        model = Model(config)
        config.data.image_size = orig_img_size

        # Load pretrained weights if specified
        if getattr(config.model, 'use_pretrained', False):
            pretrained_path = getattr(config.model, 'pretrained_path', None)
            if pretrained_path and os.path.exists(pretrained_path):
                logging.info(f"Loading pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                logging.info("✓ Pretrained weights loaded successfully")
            else:
                logging.warning(f"Pretrained path not found: {pretrained_path}")
                logging.warning("Training from scratch instead")

        model = model.to(self.device)
        model = nn.DataParallel(model)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, getattr(config.optim, 'beta2', 0.999)),
            eps=config.optim.eps,
        )

        # LR scheduler with warmup
        warmup_steps = getattr(config.training, 'warmup_steps', 1000)
        total_steps = config.training.n_iters

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # EMA
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # Mixed precision
        use_amp = getattr(config, 'training_precision', None) in ['fp16', 'bf16']
        amp_dtype = torch.bfloat16 if getattr(config, 'training_precision', '') == 'bf16' else torch.float16
        scaler = torch.amp.GradScaler('cuda') if use_amp and amp_dtype == torch.float16 else None

        # Resume
        start_step = 0
        if args.resume_training:
            ckpt_path = os.path.join(args.log_path, "ckpt.pth")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt[0])
                optimizer.load_state_dict(ckpt[1])
                start_step = ckpt[2]
                if config.model.ema and len(ckpt) > 3:
                    ema_helper.load_state_dict(ckpt[3])
                logging.info(f"Resumed from step {start_step}")
            else:
                logging.warning(f"Checkpoint not found at {ckpt_path}, starting from scratch")

        # Training loop
        logging.info("=" * 80)
        logging.info(f"Training PlantDoc: {len(train_dataset)} samples, {self.num_classes} classes")
        logging.info(f"Batch: {config.training.batch_size}, Steps: {config.training.n_iters}")
        logging.info(f"Diffusion timesteps: {self.num_timesteps}")
        logging.info(f"Augmentation prob: {config.data.augment_prob * 100}%")
        logging.info(f"Class dropout (CFG): {self.class_dropout * 100}%")
        logging.info(f"AMP: {use_amp} ({amp_dtype if use_amp else 'N/A'})")
        logging.info(f"Weighted sampler: {sampler is not None}")
        logging.info("=" * 80)

        step = start_step
        best_fid = float('inf')
        data_iter = iter(train_loader)

        while step < config.training.n_iters:
            # Get batch
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            step += 1
            x, y = x.to(self.device), y.to(self.device)

            # Encode to latent
            with torch.no_grad():
                z = self.encode(x)

            # Class dropout for CFG training
            if self.class_dropout > 0:
                mask = torch.rand(y.shape[0], device=self.device) < self.class_dropout
                # Use num_classes as "null" class for unconditional
                y = torch.where(mask, torch.full_like(y, self.num_classes), y)

            # Sample noise and timesteps
            noise = torch.randn_like(z)
            t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device)

            # Add noise
            sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            z_noisy = sqrt_alpha * z + sqrt_one_minus * noise

            # Forward
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    noise_pred = model(z_noisy, t, y)
                    loss = F.mse_loss(noise_pred, noise)
            else:
                noise_pred = model(z_noisy, t, y)
                loss = F.mse_loss(noise_pred, noise)

            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                optimizer.step()

            scheduler.step()

            if config.model.ema:
                ema_helper.update(model)

            # Logging
            if step % getattr(config.training, 'log_freq', 100) == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Step {step}/{config.training.n_iters}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
                tb_logger.add_scalar("loss", loss.item(), step)
                if use_wandb:
                    wandb.log({"loss": loss.item(), "lr": lr, "step": step}, step=step)

            # Sample images
            if step % getattr(config.training, 'sample_freq', 5000) == 0:
                self._log_samples(model, ema_helper, step, use_wandb, train_dataset)

            # Save checkpoint and eval FID
            if step % config.training.snapshot_freq == 0:
                logging.info(f"Step {step}: Saving checkpoint...")

                states = [model.state_dict(), optimizer.state_dict(), step]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                # Save latest checkpoint
                torch.save(states, os.path.join(args.log_path, "ckpt.pth"))

                # Eval FID
                if config.model.ema:
                    backup = {k: v.clone() for k, v in model.state_dict().items()}
                    ema_helper.ema(model)

                model.eval()
                test_fid, imgs = self._eval_fid(model, 'test', step)
                model.train()

                if config.model.ema:
                    model.load_state_dict(backup)

                logging.info(f"Step {step}: Test FID = {test_fid:.2f}")
                tb_logger.add_scalar("fid/test", test_fid, step)

                if use_wandb:
                    wandb.log({"fid/test": test_fid}, step=step)
                    if imgs is not None:
                        grid = tvu.make_grid(imgs[:16], nrow=4, padding=2)
                        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        wandb.log({"fid_samples": wandb.Image(grid_np, caption=f"Step {step}")}, step=step)

                if test_fid < best_fid:
                    best_fid = test_fid
                    torch.save(states, os.path.join(args.log_path, "ckpt_best.pth"))
                    logging.info(f"New best FID: {best_fid:.2f}")

        logging.info("=" * 80)
        logging.info(f"Training complete! Best FID: {best_fid:.2f}")
        logging.info("=" * 80)
        if use_wandb:
            wandb.finish()

    def _prepare_real_images(self, dataset, split):
        save_dir = os.path.join(self.args.log_path, f"real_{split}")
        if os.path.exists(save_dir):
            logging.info(f"Real images already saved for {split}")
            return

        os.makedirs(save_dir, exist_ok=True)
        num_to_save = min(len(dataset), 500)
        for i in range(num_to_save):
            img, _ = dataset[i]
            transforms.ToPILImage()(img).save(os.path.join(save_dir, f"{i:04d}.png"))
        logging.info(f"Saved {num_to_save} real images for {split}")

    @torch.no_grad()
    def _log_samples(self, model, ema_helper, step, use_wandb, dataset):
        """Generate and log sample images per class"""
        config = self.config
        model.eval()

        if config.model.ema and ema_helper:
            backup = {k: v.clone() for k, v in model.state_dict().items()}
            ema_helper.ema(model)

        # Generate samples for first 16 classes or all if < 16
        n = min(self.num_classes, 16)
        z = torch.randn(n, config.data.channels, config.data.latent_size,
                        config.data.latent_size, device=self.device)
        y = torch.arange(n, device=self.device)

        steps = getattr(config.sampling, 'num_inference_steps', 100)
        cfg = getattr(config.sampling, 'cfg_scale', 3.0)
        z = self._sample_ddim(model, z, y, steps=steps, cfg_scale=cfg)
        imgs = self.decode(z)

        # Save
        save_dir = os.path.join(self.args.log_path, "samples")
        os.makedirs(save_dir, exist_ok=True)
        grid = tvu.make_grid(imgs, nrow=4, padding=2)
        tvu.save_image(grid, os.path.join(save_dir, f"step_{step}.png"))

        # Log class names
        class_names = [dataset.get_class_name(i) for i in range(n)]
        logging.info(f"Generated samples for classes: {', '.join(class_names[:4])}...")

        if use_wandb:
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            wandb.log({"samples_per_class": wandb.Image(grid_np, caption=f"Step {step}")}, step=step)

        if config.model.ema and ema_helper:
            model.load_state_dict(backup)

        model.train()

    @torch.no_grad()
    def _eval_fid(self, model, split, step):
        config = self.config
        num_samples = getattr(config.fid, 'num_samples', 100)
        batch_size = config.sampling.batch_size
        steps = getattr(config.sampling, 'num_inference_steps', 100)
        cfg = getattr(config.sampling, 'cfg_scale', 3.0)

        fake_dir = os.path.join(self.args.log_path, f"fake_{split}_{step}")
        os.makedirs(fake_dir, exist_ok=True)

        all_imgs = []
        generated = 0

        logging.info(f"Generating {num_samples} samples for FID evaluation...")
        while generated < num_samples:
            n = min(batch_size, num_samples - generated)
            z = torch.randn(n, config.data.channels, config.data.latent_size,
                            config.data.latent_size, device=self.device)
            y = torch.randint(0, self.num_classes, (n,), device=self.device)

            z = self._sample_ddim(model, z, y, steps=steps, cfg_scale=cfg)
            imgs = self.decode(z)
            all_imgs.append(imgs.cpu())

            for j in range(imgs.shape[0]):
                transforms.ToPILImage()(imgs[j].cpu()).save(
                    os.path.join(fake_dir, f"{generated:04d}.png"))
                generated += 1

            if generated % 50 == 0:
                logging.info(f"  Generated {generated}/{num_samples} samples")

        all_imgs = torch.cat(all_imgs, dim=0)

        if FID_AVAILABLE:
            real_dir = os.path.join(self.args.log_path, f"real_{split}")
            logging.info(f"Computing FID between {real_dir} and {fake_dir}...")
            fid_score = fid.compute_fid(real_dir, fake_dir, device=self.device, mode="clean")
        else:
            logging.warning("cleanfid not available, FID score will be 0")
            fid_score = 0.0

        shutil.rmtree(fake_dir)
        return fid_score, all_imgs

    @torch.no_grad()
    def _sample_ddim(self, model, z, y, steps=100, cfg_scale=3.0):
        """DDIM sampling with classifier-free guidance"""
        step_size = self.num_timesteps // steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:steps]
        timesteps = list(reversed(timesteps))

        # Null class for CFG
        y_null = torch.full_like(y, self.num_classes)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)

            # CFG: conditional + unconditional
            if cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                y_in = torch.cat([y, y_null], dim=0)
                noise_pred = model(z_in, t_in, y_in)
                noise_cond, noise_uncond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(z, t_tensor, y)

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)

            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            x0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred = x0_pred.clamp(-4, 4)

            direction = (z - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t
            z = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * direction

        return z

    def sample(self):
        """Generate samples from trained model"""
        model = Model(self.config).to(self.device)
        model = nn.DataParallel(model)

        ckpt_path = os.path.join(self.args.log_path, "ckpt_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")

        logging.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt[0])
        model.eval()

        # Generate samples
        num_samples = 100
        steps = getattr(self.config.sampling, 'num_inference_steps', 100)
        cfg = getattr(self.config.sampling, 'cfg_scale', 3.0)

        os.makedirs(self.args.image_folder, exist_ok=True)
        logging.info(f"Generating {num_samples} samples with {steps} steps, CFG={cfg}")

        generated = 0
        batch_size = self.config.sampling.batch_size

        while generated < num_samples:
            n = min(batch_size, num_samples - generated)
            z = torch.randn(n, self.config.data.channels, self.config.data.latent_size,
                            self.config.data.latent_size, device=self.device)
            y = torch.randint(0, self.num_classes, (n,), device=self.device)

            z = self._sample_ddim(model, z, y, steps=steps, cfg_scale=cfg)
            imgs = self.decode(z)

            for j in range(imgs.shape[0]):
                transforms.ToPILImage()(imgs[j].cpu()).save(
                    os.path.join(self.args.image_folder, f"{generated:04d}.png"))
                generated += 1

            logging.info(f"Generated {generated}/{num_samples}")

        logging.info(f"Samples saved to {self.args.image_folder}")

    def test(self):
        pass
