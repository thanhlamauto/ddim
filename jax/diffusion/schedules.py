"""Beta schedules for diffusion models."""

import jax.numpy as jnp
import math


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> jnp.ndarray:
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> jnp.ndarray:
    """Linear schedule."""
    return jnp.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(betas: jnp.ndarray) -> dict:
    """Precompute diffusion parameters from betas."""
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': jnp.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': jnp.sqrt(1.0 - alphas_cumprod),
    }
