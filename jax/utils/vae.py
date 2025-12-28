"""VAE wrapper using diffusers FlaxAutoencoderKL."""

import jax
import jax.numpy as jnp

# Monkey patch for huggingface_hub compatibility
try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = lambda *args, **kwargs: hf_hub_download(*args, **kwargs)
except ImportError:
    pass

from diffusers import FlaxAutoencoderKL


class VAEWrapper:
    """Wrapper for Stable Diffusion VAE."""

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        """Initialize VAE.

        Args:
            model_id: HuggingFace model ID for VAE
        """
        self.vae, self.params = FlaxAutoencoderKL.from_pretrained(
            model_id,
            dtype=jnp.bfloat16
        )
        self.scale_factor = 0.18215

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latent space.

        Args:
            images: Images in [0, 1] range, shape [B, H, W, 3]

        Returns:
            latents: Latent vectors, shape [B, H//8, W//8, 4]
        """
        # Scale to [-1, 1]
        images = 2 * images - 1

        # VAE expects [B, C, H, W] format
        images = jnp.transpose(images, (0, 3, 1, 2))

        # Encode
        latent_dist = self.vae.apply(
            {'params': self.params},
            images,
            method=self.vae.encode
        )

        # Sample from distribution
        latents = latent_dist.latent_dist.mean  # Use mean for deterministic encoding
        latents = latents * self.scale_factor

        # Convert back to [B, H, W, C] for Flax conv
        latents = jnp.transpose(latents, (0, 2, 3, 1))

        return latents

    def decode(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latents to images.

        Args:
            latents: Latent vectors, shape [B, H, W, 4]

        Returns:
            images: Images in [0, 1] range, shape [B, H*8, W*8, 3]
        """
        # Unscale
        latents = latents / self.scale_factor

        # VAE expects [B, C, H, W]
        latents = jnp.transpose(latents, (0, 3, 1, 2))

        # Decode
        images = self.vae.apply(
            {'params': self.params},
            latents,
            method=self.vae.decode
        ).sample

        # Convert to [B, H, W, C]
        images = jnp.transpose(images, (0, 2, 3, 1))

        # Scale to [0, 1]
        images = (images + 1) / 2
        images = jnp.clip(images, 0, 1)

        return images


def create_vae(model_id: str = "stabilityai/sd-vae-ft-mse") -> VAEWrapper:
    """Create VAE wrapper."""
    return VAEWrapper(model_id)
