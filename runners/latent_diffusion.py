import os
import logging
import time
import glob
import shutil

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as tvu
from PIL import Image

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets.potato_single import PotatoLateBlight

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed")

try:
    from diffusers import AutoencoderKL
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    logging.warning("diffusers not installed, VAE not available")

try:
    from cleanfid import fid
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logging.warning("cleanfid not installed")


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "const":
        return beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)


class LatentDiffusion:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Setup betas
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - self.betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Load VAE
        if VAE_AVAILABLE:
            vae_model_id = getattr(config.vae, 'model_id', 'stabilityai/sd-vae-ft-mse')
            logging.info(f"Loading VAE: {vae_model_id}")
            self.vae = AutoencoderKL.from_pretrained(vae_model_id).to(self.device)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae_scale_factor = 0.18215  # SD VAE scaling factor
        else:
            raise RuntimeError("diffusers package required for latent diffusion")

    @torch.no_grad()
    def encode_to_latent(self, x):
        """Encode images to latent space. x: [B, 3, H, W] in [0, 1]"""
        x = 2 * x - 1  # Scale to [-1, 1]
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.vae_scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z):
        """Decode latents to images. Returns [B, 3, H, W] in [0, 1]"""
        z = z / self.vae_scale_factor
        images = self.vae.decode(z).sample
        images = (images + 1) / 2  # Scale to [0, 1]
        return images.clamp(0, 1)

    def train(self):
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        # Initialize wandb
        use_wandb = WANDB_AVAILABLE and getattr(config, 'wandb', None) and getattr(config.wandb, 'enabled', False)
        if use_wandb:
            try:
                wandb.init(
                    project=getattr(config.wandb, 'project', 'latent-diffusion'),
                    entity=getattr(config.wandb, 'entity', None),
                    name=getattr(config.wandb, 'name', None),
                    tags=getattr(config.wandb, 'tags', []),
                    notes=getattr(config.wandb, 'notes', ''),
                    config={
                        'image_size': config.data.image_size,
                        'latent_size': config.data.latent_size,
                        'batch_size': config.training.batch_size,
                        'lr': config.optim.lr,
                        'epochs': config.training.n_epochs,
                        'num_timesteps': config.diffusion.num_diffusion_timesteps,
                        'inference_steps': getattr(config.sampling, 'num_inference_steps', 100),
                    }
                )
                logging.info("Wandb initialized")
            except Exception as e:
                logging.warning(f"Failed to init wandb: {e}. Continuing without wandb.")
                use_wandb = False

        # Create datasets
        train_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.RandomHorizontalFlip() if config.data.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
        ])

        train_dataset = PotatoLateBlight(
            root=config.data.data_root,
            split='train',
            transform=train_transform,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.data.split_seed,
        )
        val_dataset = PotatoLateBlight(
            root=config.data.data_root,
            split='val',
            transform=val_transform,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.data.split_seed,
        )
        test_dataset = PotatoLateBlight(
            root=config.data.data_root,
            split='test',
            transform=val_transform,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.data.split_seed,
        )

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Prepare real images for FID (save to disk)
        self._prepare_real_images_for_fid(val_dataset, 'val')
        self._prepare_real_images_for_fid(test_dataset, 'test')

        # Create model (operates on latent space)
        # Override image_size to latent_size for model
        original_image_size = config.data.image_size
        config.data.image_size = config.data.latent_size
        model = Model(config)
        config.data.image_size = original_image_size  # Restore

        model = model.to(self.device)
        model = nn.DataParallel(model)

        optimizer = get_optimizer(config, model.parameters())

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if args.resume_training:
            states = torch.load(os.path.join(args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if config.model.ema:
                ema_helper.load_state_dict(states[4])

        logging.info(f"Starting training from epoch {start_epoch}, step {step}")
        logging.info(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logging.info(f"Snapshot freq: {config.training.snapshot_freq} steps")

        best_fid = float('inf')

        for epoch in range(start_epoch, config.training.n_epochs):
            model.train()

            for i, (x, _) in enumerate(train_loader):
                step += 1
                x = x.to(self.device)

                # Encode to latent space
                with torch.no_grad():
                    z = self.encode_to_latent(x)

                # Sample noise
                noise = torch.randn_like(z)

                # Sample timesteps
                t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()

                # Add noise
                sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                z_noisy = sqrt_alpha * z + sqrt_one_minus_alpha * noise

                # Predict noise
                noise_pred = model(z_noisy, t)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                optimizer.step()

                if config.model.ema:
                    ema_helper.update(model)

                # Logging
                if step % 10 == 0:
                    logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                    tb_logger.add_scalar("loss", loss.item(), step)
                    if use_wandb:
                        wandb.log({"loss": loss.item(), "epoch": epoch, "step": step}, step=step)

                # Save checkpoint and evaluate
                if step % config.training.snapshot_freq == 0:
                    logging.info(f"Step {step}: Saving checkpoint and evaluating...")

                    # Save checkpoint
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if config.model.ema:
                        states.append(ema_helper.state_dict())
                    # Only keep latest checkpoint to save disk space
                    torch.save(states, os.path.join(args.log_path, "ckpt.pth"))

                    # Delete old numbered checkpoints
                    import glob as glob_module
                    for old in glob_module.glob(os.path.join(args.log_path, "ckpt_*.pth")):
                        if "best" not in old:
                            os.remove(old)

                    # Generate samples and compute FID
                    # Backup current weights before applying EMA
                    if config.model.ema:
                        # Save current model weights
                        model_state_backup = {k: v.clone() for k, v in model.state_dict().items()}
                        ema_helper.ema(model)

                    model.eval()
                    val_fid, generated_images = self._evaluate_fid(model, 'val', step)
                    model.train()

                    if config.model.ema:
                        # Restore original weights
                        model.load_state_dict(model_state_backup)

                    logging.info(f"Step {step}: Val FID = {val_fid:.2f}")
                    tb_logger.add_scalar("fid/val", val_fid, step)

                    if use_wandb:
                        # Log FID
                        wandb.log({"fid/val": val_fid}, step=step)

                        # Log generated images
                        if generated_images is not None and len(generated_images) > 0:
                            # Create grid of 16 images (4x4)
                            num_images = min(16, len(generated_images))
                            grid = tvu.make_grid(generated_images[:num_images], nrow=4, padding=2, normalize=False)
                            # Convert to numpy for wandb (C, H, W) -> (H, W, C)
                            grid_np = grid.permute(1, 2, 0).cpu().numpy()
                            grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
                            wandb.log({
                                "generated_images": wandb.Image(grid_np, caption=f"Step {step}, FID={val_fid:.2f}")
                            }, step=step)

                    # Save best model
                    if val_fid < best_fid:
                        best_fid = val_fid
                        torch.save(states, os.path.join(args.log_path, "ckpt_best.pth"))
                        logging.info(f"Step {step}: New best FID = {best_fid:.2f}")

        # Final evaluation on test set
        logging.info("Training complete. Evaluating on test set...")

        # Load best model
        states = torch.load(os.path.join(args.log_path, "ckpt_best.pth"))
        model.load_state_dict(states[0])
        if config.model.ema and ema_helper:
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)

        model.eval()
        test_fid, test_images = self._evaluate_fid(model, 'test', step)

        logging.info(f"=" * 50)
        logging.info(f"FINAL RESULTS")
        logging.info(f"Best Val FID: {best_fid:.2f}")
        logging.info(f"Test FID: {test_fid:.2f}")
        logging.info(f"=" * 50)

        if use_wandb:
            wandb.log({"fid/test": test_fid, "fid/best_val": best_fid})
            if test_images is not None:
                grid = tvu.make_grid(test_images[:16], nrow=4, normalize=True)
                wandb.log({"test_generated_images": wandb.Image(grid, caption="Final Test")})
            wandb.finish()

        # Save final results
        with open(os.path.join(args.log_path, "results.txt"), "w") as f:
            f.write(f"Best Val FID: {best_fid:.2f}\n")
            f.write(f"Test FID: {test_fid:.2f}\n")

    def _prepare_real_images_for_fid(self, dataset, split):
        """Save real images to disk for FID calculation"""
        save_dir = os.path.join(self.args.log_path, f"real_images_{split}")
        os.makedirs(save_dir, exist_ok=True)

        for i, (img, _) in enumerate(dataset):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img.save(os.path.join(save_dir, f"{i:04d}.png"))

        logging.info(f"Saved {len(dataset)} real images for {split} FID")

    @torch.no_grad()
    def _evaluate_fid(self, model, split, step):
        """Generate samples and compute FID"""
        config = self.config
        num_samples = getattr(config.fid, 'num_samples', 100)
        batch_size = config.sampling.batch_size
        num_inference_steps = getattr(config.sampling, 'num_inference_steps', 100)

        # Generate samples
        fake_dir = os.path.join(self.args.log_path, f"fake_images_{split}_{step}")
        os.makedirs(fake_dir, exist_ok=True)

        all_images = []
        num_generated = 0

        while num_generated < num_samples:
            current_batch = min(batch_size, num_samples - num_generated)

            # Start from random noise
            z = torch.randn(
                current_batch,
                config.data.channels,
                config.data.latent_size,
                config.data.latent_size,
                device=self.device
            )

            # DDIM sampling
            z = self._ddim_sample(model, z, num_inference_steps)

            # Decode to images
            images = self.decode_from_latent(z)
            all_images.append(images.cpu())

            # Save images
            for j in range(images.shape[0]):
                img = transforms.ToPILImage()(images[j].cpu())
                img.save(os.path.join(fake_dir, f"{num_generated:04d}.png"))
                num_generated += 1

        all_images = torch.cat(all_images, dim=0)

        # Compute FID
        if FID_AVAILABLE:
            real_dir = os.path.join(self.args.log_path, f"real_images_{split}")
            fid_score = fid.compute_fid(real_dir, fake_dir, device=self.device, mode="clean")
        else:
            fid_score = 0.0
            logging.warning("cleanfid not available, FID set to 0")

        # Clean up fake images
        shutil.rmtree(fake_dir)

        return fid_score, all_images

    @torch.no_grad()
    def _ddim_sample(self, model, z, num_steps):
        """DDIM sampling with specified number of steps"""
        # Create timestep sequence (from T-1 down to 0)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))  # [T-1, ..., 0]

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = model(z, t_tensor)

            # Get alpha values
            alpha_t = self.alphas_cumprod[t]

            # For the last step, alpha_prev should be 1.0 (fully denoised)
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)

            # Predict x0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            # Clip x0 prediction (latent space typically in [-4, 4] range after VAE scaling)
            x0_pred = x0_pred.clamp(-4, 4)

            # DDIM step (eta=0 for deterministic sampling)
            # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1}) * noise_direction
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)

            # Direction pointing to x_t
            direction = (z - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t

            z = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * direction

        return z

    def sample(self):
        """Generate samples from trained model"""
        config = self.config
        args = self.args

        # Load model
        original_image_size = config.data.image_size
        config.data.image_size = config.data.latent_size
        model = Model(config)
        config.data.image_size = original_image_size

        model = model.to(self.device)
        model = nn.DataParallel(model)

        states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=self.device)
        model.load_state_dict(states[0])

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)

        model.eval()

        # Generate samples
        num_samples = getattr(config.sampling, 'batch_size', 16)
        num_inference_steps = getattr(config.sampling, 'num_inference_steps', 100)

        z = torch.randn(
            num_samples,
            config.data.channels,
            config.data.latent_size,
            config.data.latent_size,
            device=self.device
        )

        z = self._ddim_sample(model, z, num_inference_steps)
        images = self.decode_from_latent(z)

        # Save images
        os.makedirs(args.image_folder, exist_ok=True)
        for i in range(images.shape[0]):
            tvu.save_image(images[i], os.path.join(args.image_folder, f"{i:04d}.png"))

        logging.info(f"Saved {num_samples} samples to {args.image_folder}")

    def test(self):
        pass
