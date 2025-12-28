"""DDIM sampling with classifier-free guidance."""

import jax
import jax.numpy as jnp
from typing import Callable


def ddim_sample(
    model_fn: Callable,
    params: dict,
    z: jnp.ndarray,
    y: jnp.ndarray,
    diffusion_params: dict,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    num_classes: int = 15,
) -> jnp.ndarray:
    """DDIM sampling with classifier-free guidance.

    Args:
        model_fn: Model apply function
        params: Model parameters
        z: Initial noise, shape [B, H, W, C]
        y: Class labels, shape [B]
        diffusion_params: Dict with alphas_cumprod etc.
        num_steps: Number of DDIM steps
        cfg_scale: Classifier-free guidance scale
        num_classes: Number of classes (for null class)

    Returns:
        Denoised samples, shape [B, H, W, C]
    """
    alphas_cumprod = diffusion_params['alphas_cumprod']
    num_timesteps = len(alphas_cumprod)

    # Create timestep sequence
    step_size = num_timesteps // num_steps
    timesteps = list(range(0, num_timesteps, step_size))[:num_steps]
    timesteps = list(reversed(timesteps))

    # Null class for CFG
    y_null = jnp.full_like(y, num_classes)

    for i, t in enumerate(timesteps):
        t_batch = jnp.full((z.shape[0],), t, dtype=jnp.int32)

        # CFG: run both conditional and unconditional
        if cfg_scale > 1.0:
            z_in = jnp.concatenate([z, z], axis=0)
            t_in = jnp.concatenate([t_batch, t_batch], axis=0)
            y_in = jnp.concatenate([y, y_null], axis=0)

            noise_pred = model_fn(params, z_in, t_in, y_in, train=False)
            noise_cond, noise_uncond = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model_fn(params, z, t_batch, y, train=False)

        # DDIM step
        alpha_t = alphas_cumprod[t]
        if i + 1 < len(timesteps):
            alpha_prev = alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prev = 1.0

        sqrt_alpha_t = jnp.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = jnp.sqrt(1 - alpha_t)

        # Predict x0
        x0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        x0_pred = jnp.clip(x0_pred, -4, 4)

        # Direction
        direction = (z - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t

        # Update z
        z = jnp.sqrt(alpha_prev) * x0_pred + jnp.sqrt(1 - alpha_prev) * direction

    return z


def ddim_sample_pmap(
    model_fn: Callable,
    params: dict,
    z: jnp.ndarray,
    y: jnp.ndarray,
    diffusion_params: dict,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    num_classes: int = 15,
) -> jnp.ndarray:
    """DDIM sampling for pmap (per-device).

    Same as ddim_sample but designed to work within pmap.
    """
    return ddim_sample(
        model_fn, params, z, y, diffusion_params,
        num_steps, cfg_scale, num_classes
    )
