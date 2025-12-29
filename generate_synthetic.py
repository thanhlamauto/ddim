"""
Generate synthetic PlantVillage dataset using trained DDIM model.
Generates the same number of images per class as the original dataset.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.diffusion import Model


# Class names in sorted order (matching dataset)
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Number of images per class (same as original PlantVillage)
CLASS_COUNTS = {
    "Pepper__bell___Bacterial_spot": 997,
    "Pepper__bell___healthy": 1478,
    "Potato___Early_blight": 1000,
    "Potato___Late_blight": 1000,
    "Potato___healthy": 152,
    "Tomato_Bacterial_spot": 2127,
    "Tomato_Early_blight": 1000,
    "Tomato_Late_blight": 1909,
    "Tomato_Leaf_Mold": 952,
    "Tomato_Septoria_leaf_spot": 1771,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 1676,
    "Tomato__Target_Spot": 1404,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 3209,
    "Tomato__Tomato_mosaic_virus": 373,
    "Tomato_healthy": 1591,
}


class AttrDict:
    """Simple attribute dictionary."""
    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)


def create_config():
    """Create config matching the trained checkpoint."""
    config = AttrDict()

    # Data config
    config.data = AttrDict()
    config.data.channels = 4
    config.data.image_size = 32  # Model operates on latent space 32x32
    config.data.latent_size = 32

    # Model config
    config.model = AttrDict()
    config.model.ch = 128
    config.model.out_ch = 4
    config.model.ch_mult = [1, 2, 2, 4]
    config.model.num_res_blocks = 2
    config.model.attn_resolutions = [32, 16]  # Attention at resolutions 32 and 16 in latent space
    config.model.dropout = 0.0
    config.model.in_channels = 4
    config.model.resamp_with_conv = True
    config.model.type = 'simple'
    config.model.var_type = 'fixedlarge'
    config.model.conditional = True
    config.model.num_classes = 15
    config.model.ema = True
    config.model.ema_rate = 0.9999

    # Diffusion config
    config.diffusion = AttrDict()
    config.diffusion.beta_schedule = 'cosine'
    config.diffusion.beta_start = 0.0001
    config.diffusion.beta_end = 0.02
    config.diffusion.num_diffusion_timesteps = 1000

    # VAE config
    config.vae = AttrDict()
    config.vae.model_id = 'stabilityai/sd-vae-ft-mse'

    # Sampling config
    config.sampling = AttrDict()
    config.sampling.batch_size = 16
    config.sampling.cfg_scale = 3.0
    config.sampling.num_inference_steps = 50

    return config


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps):
    """Get beta schedule for diffusion."""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        # Cosine schedule from "Improved DDPM"
        steps = num_timesteps + 1
        s = 0.008
        x = np.linspace(0, num_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0.0001, 0.9999)
    elif beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
    return betas.astype(np.float32)


class DDIMSampler:
    """DDIM sampler with classifier-free guidance support."""

    def __init__(self, model, betas, num_timesteps, device):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps

        # Precompute alphas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod = torch.from_numpy(self.alphas_cumprod).float().to(device)

    @torch.no_grad()
    def sample(self, shape, class_labels, num_steps=50, cfg_scale=3.0, eta=0.0):
        """
        DDIM sampling with classifier-free guidance.

        Args:
            shape: (B, C, H, W) output shape
            class_labels: (B,) class indices
            num_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            eta: DDIM eta (0 = deterministic)
        """
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Create timestep sequence
        timesteps = np.linspace(0, self.num_timesteps - 1, num_steps + 1, dtype=int)
        timesteps = list(reversed(timesteps))

        # Null class for CFG (num_classes is the null token)
        null_labels = torch.full((batch_size,), 15, device=self.device, dtype=torch.long)

        for i, (t_curr, t_next) in enumerate(tqdm(
            zip(timesteps[:-1], timesteps[1:]),
            total=len(timesteps) - 1,
            desc="Sampling",
            leave=False
        )):
            t = torch.full((batch_size,), t_curr, device=self.device, dtype=torch.long)

            # Get alpha values
            alpha_curr = self.alphas_cumprod[t_curr]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)

            # Classifier-free guidance: combine conditional and unconditional predictions
            if cfg_scale > 1.0:
                # Conditional prediction
                noise_cond = self.model(x, t.float(), class_labels)
                # Unconditional prediction
                noise_uncond = self.model(x, t.float(), null_labels)
                # CFG combination
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.model(x, t.float(), class_labels)

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_next - eta ** 2 * (1 - alpha_next) / (1 - alpha_curr) * (1 - alpha_curr / alpha_next)) * noise_pred

            # Random noise (if eta > 0)
            if eta > 0 and t_next > 0:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_curr) * (1 - alpha_curr / alpha_next))
            else:
                noise = 0
                sigma = 0

            x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise

        return x


def decode_latents(vae, latents):
    """Decode latents to images using VAE."""
    # Scale latents (SD VAE uses 0.18215 scaling factor)
    latents = latents / 0.18215

    # Decode
    with torch.no_grad():
        images = vae.decode(latents).sample

    # Convert to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    return images


def save_images(images, output_dir, class_name, start_idx):
    """Save batch of images to disk."""
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        # Convert to PIL
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Save
        filename = f"synthetic_{start_idx + i:05d}.jpg"
        img_pil.save(os.path.join(output_dir, filename), quality=95)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PlantVillage dataset")
    parser.add_argument("--ckpt", type=str,
                        default="/workspace/ddim/exp/logs/plantvillage_cond/ckpt_best.pth",
                        help="Path to checkpoint")
    parser.add_argument("--config", type=str,
                        default="/workspace/ddim/exp/logs/plantvillage_cond/config.yml",
                        help="Path to config")
    parser.add_argument("--output", type=str,
                        default="/workspace/PlantVillage_Synthetic",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of DDIM steps")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create config (hardcoded to match checkpoint)
    print("Creating config...")
    config = create_config()

    # Load model
    print("Loading diffusion model...")
    model = Model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    states = torch.load(args.ckpt, map_location=device)

    # Handle DataParallel state dict
    state_dict = states[0]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    # Load EMA weights if available (usually better quality)
    if config.model.ema and len(states) > 4:
        print("Loading EMA weights...")
        from models.ema import EMAHelper
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[4])
        ema_helper.ema(model)
        print("EMA weights loaded")

    model.eval()
    print("Model loaded successfully")

    # Load VAE
    print("Loading VAE decoder...")
    vae = AutoencoderKL.from_pretrained(config.vae.model_id)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully")

    # Setup beta schedule
    betas = get_beta_schedule(
        config.diffusion.beta_schedule,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.num_diffusion_timesteps
    )

    # Create sampler
    sampler = DDIMSampler(model, betas, config.diffusion.num_diffusion_timesteps, device)

    # Image shape (latent space)
    latent_size = config.data.latent_size
    latent_channels = config.data.channels

    # Generate images for each class
    print(f"\nGenerating synthetic dataset to {args.output}")
    print(f"Settings: {args.num_steps} steps, CFG scale {args.cfg_scale}")
    print("=" * 60)

    total_images = sum(CLASS_COUNTS.values())
    generated_total = 0

    for class_idx, class_name in enumerate(CLASS_NAMES):
        num_images = CLASS_COUNTS[class_name]
        class_dir = os.path.join(args.output, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n[{class_idx + 1}/15] {class_name}: {num_images} images")

        # Generate in batches
        num_generated = 0
        pbar = tqdm(total=num_images, desc=f"  Generating")

        while num_generated < num_images:
            batch_size = min(args.batch_size, num_images - num_generated)

            # Create class labels
            labels = torch.full((batch_size,), class_idx, device=device, dtype=torch.long)

            # Sample latents
            shape = (batch_size, latent_channels, latent_size, latent_size)
            latents = sampler.sample(
                shape,
                labels,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale
            )

            # Decode to images
            images = decode_latents(vae, latents)

            # Save images
            save_images(images, class_dir, class_name, num_generated)

            num_generated += batch_size
            generated_total += batch_size
            pbar.update(batch_size)

        pbar.close()

    print("\n" + "=" * 60)
    print(f"Generation complete!")
    print(f"Total images generated: {generated_total}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
