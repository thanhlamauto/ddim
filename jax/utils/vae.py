"""VAE wrapper using diffusers FlaxAutoencoderKL.

Provides encode/decode functions to convert between pixel space and latent space.
Compatible with Stable Diffusion VAE.
"""

import jax
import jax.numpy as jnp
from functools import partial

# Fix huggingface_hub compatibility with older diffusers
try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download

    # Add back the deprecated cached_download for compatibility
    if not hasattr(huggingface_hub, 'cached_download'):
        def cached_download(url, *args, **kwargs):
            # Extract repo_id and filename from the old API
            if 'legacy_cache_layout' in kwargs:
                del kwargs['legacy_cache_layout']
            return hf_hub_download(*args, **kwargs)

        huggingface_hub.cached_download = cached_download
except Exception:
    pass

from diffusers import FlaxAutoencoderKL

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class VAEWrapper:
    """Wrapper for Stable Diffusion VAE.

    This wrapper provides encode/decode methods that work with images in [0, 1] range
    for backward compatibility with existing code.
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        """Initialize VAE.

        Args:
            model_id: HuggingFace model ID for VAE
        """
        print(f"Loading VAE model: {model_id}")

        # Try different loading strategies
        try:
            # Try Flax-native model first (faster)
            if "flax" in model_id.lower() or "pcuenq" in model_id:
                self.vae, self.params = FlaxAutoencoderKL.from_pretrained(model_id)
                print(f"Loaded Flax-native VAE")
            else:
                # Try with subfolder for models like stabilityai/sd-vae-ft-mse
                self.vae, self.params = FlaxAutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae" if "/" not in model_id.split("/")[-1] else None,
                    from_pt=True,  # Convert from PyTorch if needed
                    dtype=jnp.bfloat16
                )
        except Exception as e:
            print(f"Failed with subfolder, trying direct load: {e}")
            self.vae, self.params = FlaxAutoencoderKL.from_pretrained(
                model_id,
                from_pt=True,
                dtype=jnp.bfloat16
            )

        # Get scaling factor from model config (typically 0.18215)
        self.scale_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)

        # Get downscale factor (typically 8 for SD VAE)
        self.downscale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        print(f"VAE loaded successfully")
        print(f"  Scaling factor: {self.scale_factor}")
        print(f"  Downscale factor: {self.downscale_factor}x")
        print(f"  Using einops: {EINOPS_AVAILABLE}")

    @partial(jax.jit, static_argnums=(0,))
    def encode(self, images: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Encode images to latent space.

        Args:
            images: Images in [0, 1] range, shape [B, H, W, 3]
            deterministic: If True, use mean of latent distribution (no sampling)

        Returns:
            latents: Latent vectors, shape [B, H//8, W//8, 4]
        """
        # Scale to [-1, 1] (VAE expects this range)
        images = 2 * images - 1

        # VAE expects [B, C, H, W] format (channels-first)
        if EINOPS_AVAILABLE:
            images = rearrange(images, "b h w c -> b c h w")
        else:
            images = jnp.transpose(images, (0, 3, 1, 2))

        # Encode
        latent_dist = self.vae.apply(
            {'params': self.params},
            images,
            method=self.vae.encode
        )

        # Get latents (mean for deterministic, or could sample)
        if deterministic:
            latents = latent_dist.latent_dist.mean
        else:
            # For stochastic encoding, you'd need a random key here
            latents = latent_dist.latent_dist.mean  # Fallback to mean

        # Scale by VAE scaling factor
        latents = latents * self.scale_factor

        # Convert back to [B, H, W, C] (channels-last) for consistency
        if EINOPS_AVAILABLE:
            latents = rearrange(latents, "b c h w -> b h w c")
        else:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        return latents

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latents to images.

        Args:
            latents: Latent vectors, shape [B, H, W, 4]

        Returns:
            images: Images in [0, 1] range, shape [B, H*8, W*8, 3]
        """
        # Unscale latents
        latents = latents / self.scale_factor

        # VAE expects [B, C, H, W] (channels-first)
        if EINOPS_AVAILABLE:
            latents = rearrange(latents, "b h w c -> b c h w")
        else:
            latents = jnp.transpose(latents, (0, 3, 1, 2))

        # Decode
        images = self.vae.apply(
            {'params': self.params},
            latents,
            method=self.vae.decode
        ).sample

        # Convert to [B, H, W, C] (channels-last)
        if EINOPS_AVAILABLE:
            images = rearrange(images, "b c h w -> b h w c")
        else:
            images = jnp.transpose(images, (0, 2, 3, 1))

        # Scale from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = jnp.clip(images, 0, 1)

        return images

    def encode_internal(self, images: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Encode with internal format (for advanced usage).

        Args:
            images: Images in [-1, 1] range, shape [B, H, W, 3]
            key: JAX random key for sampling

        Returns:
            latents: Scaled latents in [B, H, W, 4] format
        """
        # Images already in [-1, 1]
        if EINOPS_AVAILABLE:
            images = rearrange(images, "b h w c -> b c h w")
        else:
            images = jnp.transpose(images, (0, 3, 1, 2))

        latent_dist = self.vae.apply(
            {'params': self.params},
            images,
            method=self.vae.encode
        )

        latents = latent_dist.latent_dist.sample(key)
        latents = latents * self.scale_factor

        if EINOPS_AVAILABLE:
            latents = rearrange(latents, "b c h w -> b h w c")
        else:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        return latents

    def decode_internal(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode with internal format (for advanced usage).

        Args:
            latents: Scaled latents, shape [B, H, W, 4]

        Returns:
            images: Images in [-1, 1] range, shape [B, H, W, 3]
        """
        latents = latents / self.scale_factor

        if EINOPS_AVAILABLE:
            latents = rearrange(latents, "b h w c -> b c h w")
        else:
            latents = jnp.transpose(latents, (0, 3, 1, 2))

        images = self.vae.apply(
            {'params': self.params},
            latents,
            method=self.vae.decode
        ).sample

        if EINOPS_AVAILABLE:
            images = rearrange(images, "b c h w -> b h w c")
        else:
            images = jnp.transpose(images, (0, 2, 3, 1))

        # Keep in [-1, 1] range
        return images


def create_vae(model_id: str = "stabilityai/sd-vae-ft-mse") -> VAEWrapper:
    """Create VAE wrapper.

    Recommended models:
    - "stabilityai/sd-vae-ft-mse" (default, good quality)
    - "pcuenq/sd-vae-ft-mse-flax" (Flax-native, faster loading)

    Args:
        model_id: HuggingFace model ID

    Returns:
        VAE wrapper instance
    """
    return VAEWrapper(model_id)
