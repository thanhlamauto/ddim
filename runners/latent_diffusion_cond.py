"""
Class-conditional Latent Diffusion with CFG support.
Optimized for PlantVillage on RTX 4090.
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
from datasets.plantvillage_latent import PlantVillageLatent

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

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.inception_score import calculate_inception_score
    IS_AVAILABLE = True
except ImportError:
    IS_AVAILABLE = False


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


class LatentDiffusionCond:
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
        self.num_classes = getattr(config.model, 'num_classes', 15)

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
                    project=getattr(config.wandb, 'project', 'latent-diffusion'),
                    name=getattr(config.wandb, 'name', None),
                    tags=getattr(config.wandb, 'tags', []),
                    config={
                        'batch_size': config.training.batch_size,
                        'lr': config.optim.lr,
                        'n_iters': config.training.n_iters,
                        'num_classes': self.num_classes,
                    }
                )
                logging.info("Wandb initialized")
            except Exception as e:
                logging.warning(f"Wandb init failed: {e}")
                use_wandb = False

        # Dataset
        train_dataset = PlantVillageLatent(
            root=config.data.data_root,
            split='train',
            config=config,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.data.split_seed,
        )
        val_dataset = PlantVillageLatent(
            root=config.data.data_root,
            split='val',
            config=config,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.data.split_seed,
        )

        # Weighted sampler
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
        self._prepare_real_images(val_dataset, 'val')

        # Model
        orig_img_size = config.data.image_size
        config.data.image_size = config.data.latent_size
        model = Model(config)
        config.data.image_size = orig_img_size

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
            ckpt = torch.load(os.path.join(args.log_path, "ckpt.pth"))
            model.load_state_dict(ckpt[0])
            optimizer.load_state_dict(ckpt[1])
            start_step = ckpt[2]
            if config.model.ema and len(ckpt) > 3:
                ema_helper.load_state_dict(ckpt[3])
            logging.info(f"Resumed from step {start_step}")

        # Training loop
        logging.info(f"Training: {len(train_dataset)} samples, {self.num_classes} classes")
        logging.info(f"Batch: {config.training.batch_size}, Steps: {config.training.n_iters}")
        logging.info(f"AMP: {use_amp} ({amp_dtype if use_amp else 'N/A'})")

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
                # Use num_classes as "null" class
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
                logging.info(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
                tb_logger.add_scalar("loss", loss.item(), step)
                if use_wandb:
                    wandb.log({"loss": loss.item(), "lr": lr, "step": step}, step=step)

            # Sample images
            if step % getattr(config.training, 'sample_freq', 5000) == 0:
                self._log_samples(model, ema_helper, step, use_wandb)

            # Save checkpoint and eval FID
            if step % config.training.snapshot_freq == 0:
                logging.info(f"Step {step}: Saving checkpoint...")

                states = [model.state_dict(), optimizer.state_dict(), step]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                # Only keep latest checkpoint to save disk space
                torch.save(states, os.path.join(args.log_path, "ckpt.pth"))

                # Delete old numbered checkpoints to save space
                import glob
                old_ckpts = glob.glob(os.path.join(args.log_path, "ckpt_*.pth"))
                for old_ckpt in old_ckpts:
                    if "best" not in old_ckpt:
                        os.remove(old_ckpt)
                        logging.info(f"Removed old checkpoint: {old_ckpt}")

                # Eval FID
                if config.model.ema:
                    backup = {k: v.clone() for k, v in model.state_dict().items()}
                    ema_helper.ema(model)

                model.eval()
                val_fid, is_mean, is_std, imgs = self._eval_fid(model, 'val', step)
                model.train()

                if config.model.ema:
                    model.load_state_dict(backup)

                logging.info(f"Step {step}: Val FID = {val_fid:.2f}, IS = {is_mean:.2f} ± {is_std:.2f}")
                tb_logger.add_scalar("fid/val", val_fid, step)
                tb_logger.add_scalar("is/val", is_mean, step)
                tb_logger.add_scalar("is/val_std", is_std, step)

                if use_wandb:
                    wandb.log({"fid/val": val_fid, "is/val": is_mean, "is/val_std": is_std}, step=step)
                    if imgs is not None:
                        grid = tvu.make_grid(imgs[:16], nrow=4, padding=2)
                        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        wandb.log({"samples": wandb.Image(grid_np, caption=f"Step {step}")}, step=step)

                if val_fid < best_fid:
                    best_fid = val_fid
                    torch.save(states, os.path.join(args.log_path, "ckpt_best.pth"))
                    logging.info(f"New best FID: {best_fid:.2f}")

        logging.info(f"Training complete. Best FID: {best_fid:.2f}")
        if use_wandb:
            wandb.finish()

    def _prepare_real_images(self, dataset, split):
        save_dir = os.path.join(self.args.log_path, f"real_{split}")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(len(dataset), 500)):
            img, _ = dataset[i]
            transforms.ToPILImage()(img).save(os.path.join(save_dir, f"{i:04d}.png"))
        logging.info(f"Saved {min(len(dataset), 500)} real images for {split}")

    @torch.no_grad()
    def _log_samples(self, model, ema_helper, step, use_wandb):
        """Generate and log sample images per class"""
        config = self.config
        model.eval()

        if config.model.ema and ema_helper:
            backup = {k: v.clone() for k, v in model.state_dict().items()}
            ema_helper.ema(model)

        # Generate 1 sample per class
        n = min(self.num_classes, 16)
        z = torch.randn(n, config.data.channels, config.data.latent_size,
                        config.data.latent_size, device=self.device)
        y = torch.arange(n, device=self.device)

        z = self._sample_ddim(model, z, y, steps=50, cfg_scale=3.0)
        imgs = self.decode(z)

        # Save
        save_dir = os.path.join(self.args.log_path, "samples")
        os.makedirs(save_dir, exist_ok=True)
        grid = tvu.make_grid(imgs, nrow=4, padding=2)
        tvu.save_image(grid, os.path.join(save_dir, f"step_{step}.png"))

        if use_wandb:
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            wandb.log({"samples_per_class": wandb.Image(grid_np)}, step=step)

        if config.model.ema and ema_helper:
            model.load_state_dict(backup)

        model.train()

    @torch.no_grad()
    def _eval_fid(self, model, split, step):
        config = self.config
        num_samples = getattr(config.fid, 'num_samples', 500)
        batch_size = config.sampling.batch_size
        steps = getattr(config.sampling, 'num_inference_steps', 50)
        cfg = getattr(config.sampling, 'cfg_scale', 3.0)

        fake_dir = os.path.join(self.args.log_path, f"fake_{split}_{step}")
        os.makedirs(fake_dir, exist_ok=True)

        all_imgs = []
        generated = 0

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

        all_imgs = torch.cat(all_imgs, dim=0)

        # Calculate FID
        if FID_AVAILABLE:
            real_dir = os.path.join(self.args.log_path, f"real_{split}")
            fid_score = fid.compute_fid(real_dir, fake_dir, device=self.device, mode="clean")
        else:
            fid_score = 0.0

        # Calculate Inception Score
        if IS_AVAILABLE:
            is_mean, is_std = calculate_inception_score(fake_dir, batch_size=50, device=self.device)
            logging.info(f"IS: {is_mean:.2f} ± {is_std:.2f}")
        else:
            is_mean, is_std = 0.0, 0.0

        shutil.rmtree(fake_dir)
        return fid_score, is_mean, is_std, all_imgs

    @torch.no_grad()
    def _sample_ddim(self, model, z, y, steps=50, cfg_scale=3.0):
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
        pass

    def test(self):
        pass
