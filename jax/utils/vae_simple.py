"""Simplified VAE using PyTorch with JAX conversion."""

import jax
import jax.numpy as jnp
import numpy as np

# Patch huggingface_hub for compatibility with older diffusers
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        from huggingface_hub import hf_hub_download
        # Add the deprecated function back
        huggingface_hub.cached_download = hf_hub_download
except Exception:
    pass


class SimpleVAE:
    """Simple VAE wrapper that uses PyTorch for encode/decode and converts to JAX."""

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        """Initialize VAE.

        Args:
            model_id: HuggingFace model ID for VAE
        """
        try:
            from diffusers import AutoencoderKL
            import torch
        except ImportError as e:
            raise ImportError(f"Need diffusers and torch: {e}")

        # Load PyTorch VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )

        # Move to CPU (TPU will handle JAX arrays)
        self.vae = self.vae.to("cpu")
        self.vae.eval()

        self.scale_factor = 0.18215
        print(f"VAE loaded: {model_id}")

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latent space.

        Args:
            images: Images in [0, 1] range, shape [B, H, W, 3]

        Returns:
            latents: Latent vectors, shape [B, H//8, W//8, 4]
        """
        import torch

        # Convert JAX to numpy to torch
        images_np = np.array(images)

        # Scale to [-1, 1]
        images_np = 2 * images_np - 1

        # Convert to [B, C, H, W] for PyTorch
        images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()

        # Encode
        with torch.no_grad():
            latent_dist = self.vae.encode(images_torch).latent_dist
            latents = latent_dist.mean  # Use mean for deterministic
            latents = latents * self.scale_factor

        # Convert back to [B, H, W, C] for JAX
        latents_np = latents.permute(0, 2, 3, 1).cpu().numpy()

        return jnp.array(latents_np)

    def decode(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latents to images.

        Args:
            latents: Latent vectors, shape [B, H, W, 4]

        Returns:
            images: Images in [0, 1] range, shape [B, H*8, W*8, 3]
        """
        import torch

        # Convert to numpy
        latents_np = np.array(latents)

        # Unscale
        latents_np = latents_np / self.scale_factor

        # Convert to [B, C, H, W]
        latents_torch = torch.from_numpy(latents_np).permute(0, 3, 1, 2).float()

        # Decode
        with torch.no_grad():
            images = self.vae.decode(latents_torch).sample

        # Convert to [B, H, W, C]
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()

        # Scale to [0, 1]
        images_np = (images_np + 1) / 2
        images_np = np.clip(images_np, 0, 1)

        return jnp.array(images_np)


def create_vae(model_id: str = "stabilityai/sd-vae-ft-mse") -> SimpleVAE:
    """Create simple VAE wrapper."""
    return SimpleVAE(model_id)
