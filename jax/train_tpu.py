"""
Class-conditional Latent Diffusion Training on TPU v5e-8.

Usage:
    python train_tpu.py
"""

import os
import functools
from typing import Any

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from PIL import Image

# Local imports
from config import get_config, Config
from models.unet_flax import UNet
from data.dataset import get_plantvillage_dataset, prepare_batch_for_pmap
from diffusion.schedules import cosine_beta_schedule, linear_beta_schedule, get_diffusion_params
from diffusion.sampling import ddim_sample
from utils.checkpoint import CheckpointManager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available")

try:
    from utils.vae import create_vae
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("VAE not available - will train in pixel space")


class TrainState(train_state.TrainState):
    """Extended train state with EMA."""
    ema_params: Any = None


def create_train_state(rng, config: Config, model: nn.Module) -> TrainState:
    """Initialize training state."""
    # Initialize model
    dummy_x = jnp.ones((1, config.data.latent_size, config.data.latent_size, config.data.channels))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_y = jnp.zeros((1,), dtype=jnp.int32)

    params = model.init(rng, dummy_x, dummy_t, dummy_y, train=False)['params']

    # Optimizer with warmup
    schedule = optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip),
        optax.adamw(schedule, weight_decay=config.training.weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        ema_params=params,  # Initialize EMA with same params
    )


def update_ema(state: TrainState, ema_decay: float) -> TrainState:
    """Update EMA parameters."""
    new_ema = jax.tree_util.tree_map(
        lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
        state.ema_params,
        state.params,
    )
    return state.replace(ema_params=new_ema)


@functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(state, images, labels, noise, timesteps, dropout_rng, diffusion_params, config_dict):
    """Single training step (pmap'd across devices)."""

    def loss_fn(params):
        # Add noise
        sqrt_alpha = diffusion_params['sqrt_alphas_cumprod'][timesteps]
        sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod'][timesteps]

        sqrt_alpha = sqrt_alpha[:, None, None, None]
        sqrt_one_minus = sqrt_one_minus[:, None, None, None]

        noisy = sqrt_alpha * images + sqrt_one_minus * noise

        # Predict noise
        noise_pred = state.apply_fn(
            {'params': params},
            noisy, timesteps, labels,
            train=True,
            rngs={'dropout': dropout_rng},
        )

        # MSE loss
        loss = jnp.mean((noise_pred - noise) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')

    # Update state
    state = state.apply_gradients(grads=grads)

    return state, loss


def encode_batch(vae, images):
    """Encode images to latent space."""
    if vae is not None:
        return vae.encode(images)
    else:
        # Fallback: just resize (for testing without VAE)
        return jax.image.resize(images, images.shape[:-1] + (4,), method='bilinear')


def generate_samples(state, vae, diffusion_params, config, rng, num_samples=16):
    """Generate sample images."""
    # Random latents
    z = jax.random.normal(rng, (num_samples, config.data.latent_size,
                                 config.data.latent_size, config.data.channels))

    # Random classes (one per class for visualization)
    y = jnp.arange(min(num_samples, config.data.num_classes)) % config.data.num_classes

    # Pad if needed
    if len(y) < num_samples:
        y = jnp.concatenate([y, jnp.zeros(num_samples - len(y), dtype=jnp.int32)])

    # Model function for sampling
    def model_fn(params, z, t, y, train=False):
        return state.apply_fn({'params': params}, z, t, y, train=train)

    # Use EMA params for sampling
    z = ddim_sample(
        model_fn=model_fn,
        params=state.ema_params,
        z=z,
        y=y,
        diffusion_params=diffusion_params,
        num_steps=config.sampling.num_steps,
        cfg_scale=config.sampling.cfg_scale,
        num_classes=config.data.num_classes,
    )

    # Decode
    if vae is not None:
        images = vae.decode(z)
    else:
        images = z[..., :3]  # Fallback

    return images


def save_image_grid(images, path, nrow=4):
    """Save images as a grid."""
    images = np.array(images)
    images = (images * 255).clip(0, 255).astype(np.uint8)

    n, h, w, c = images.shape
    ncol = (n + nrow - 1) // nrow

    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c_idx = i // ncol, i % ncol
        grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = img

    Image.fromarray(grid).save(path)


def main():
    # Setup
    config = get_config()
    print(f"JAX devices: {jax.devices()}")
    print(f"Num devices: {jax.device_count()}")

    num_devices = jax.device_count()
    config.training.num_devices = num_devices

    # Wandb
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config={
                    'batch_size': config.training.batch_size,
                    'num_steps': config.training.num_steps,
                    'learning_rate': config.training.learning_rate,
                    'num_classes': config.data.num_classes,
                    'num_devices': num_devices,
                }
            )
        except Exception as e:
            print(f"Wandb init failed: {e}")
            use_wandb = False

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    # Dataset
    train_ds, num_samples, num_classes = get_plantvillage_dataset(
        data_root=config.data.data_root,
        split='train',
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        num_devices=num_devices,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
        random_crop=config.data.random_crop,
        random_flip=config.data.random_flip,
        color_jitter=config.data.color_jitter,
    )

    config.data.num_classes = num_classes
    print(f"Dataset: {num_samples} samples, {num_classes} classes")

    # VAE
    vae = None
    if VAE_AVAILABLE:
        try:
            vae = create_vae(config.vae_model_id)
            print("VAE loaded successfully")
        except Exception as e:
            print(f"Failed to load VAE: {e}")

    # Diffusion params
    if config.diffusion.beta_schedule == "cosine":
        betas = cosine_beta_schedule(config.diffusion.num_timesteps)
    else:
        betas = linear_beta_schedule(
            config.diffusion.num_timesteps,
            config.diffusion.beta_start,
            config.diffusion.beta_end,
        )
    diffusion_params = get_diffusion_params(betas)

    # Model
    model = UNet(
        ch=config.model.ch,
        ch_mult=config.model.ch_mult,
        num_res_blocks=config.model.num_res_blocks,
        attn_resolutions=config.model.attn_resolutions,
        num_classes=num_classes,
        dropout=config.model.dropout,
        in_channels=config.data.channels,
        out_channels=config.data.channels,
    )

    # Initialize state
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, model)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(config.checkpoint_dir, max_to_keep=5)

    # Resume if available
    latest_step = ckpt_manager.latest_step()
    if latest_step is not None:
        restored = ckpt_manager.restore(latest_step)
        if restored is not None:
            state = restored
            print(f"Resumed from step {latest_step}")

    # Replicate state for pmap
    state = flax.jax_utils.replicate(state)

    # Replicate diffusion params
    diffusion_params_rep = flax.jax_utils.replicate(diffusion_params)

    # Config dict for pmap (needs to be a pytree)
    config_dict = {
        'class_dropout': config.model.class_dropout,
        'num_classes': num_classes,
        'num_timesteps': config.diffusion.num_timesteps,
    }
    config_dict_rep = flax.jax_utils.replicate(config_dict)

    # Training loop
    print(f"\nStarting training for {config.training.num_steps} steps...")
    print(f"Batch size: {config.training.batch_size} (global), {config.training.batch_size // num_devices} (per device)")

    step = latest_step or 0
    train_iter = iter(train_ds)

    while step < config.training.num_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)

        images, labels = prepare_batch_for_pmap(batch, num_devices)

        # Encode to latent (on CPU/single device for now)
        if vae is not None:
            # Flatten for VAE
            B = images.shape[0] * images.shape[1]
            images_flat = images.reshape(B, *images.shape[2:])
            latents_flat = vae.encode(jnp.array(images_flat))
            latents = latents_flat.reshape(num_devices, -1, *latents_flat.shape[1:])
        else:
            latents = jnp.array(images)

        # Class dropout for CFG
        rng, dropout_rng = jax.random.split(rng)
        mask = jax.random.uniform(dropout_rng, labels.shape) < config.model.class_dropout
        labels = jnp.where(mask, num_classes, labels)

        # Sample noise and timesteps
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, latents.shape)
        timesteps = jax.random.randint(
            t_rng, (num_devices, latents.shape[1]),
            0, config.diffusion.num_timesteps
        )

        # Split RNG for each device
        rng, *device_rngs = jax.random.split(rng, num_devices + 1)
        device_rngs = jnp.stack(device_rngs)

        # Train step
        state, loss = train_step(
            state, latents, labels, noise, timesteps,
            device_rngs, diffusion_params_rep, config_dict_rep
        )

        step += 1

        # Logging
        if step % config.training.log_freq == 0:
            loss_val = float(jax.device_get(loss[0]))
            print(f"Step {step}, Loss: {loss_val:.4f}")
            if use_wandb:
                wandb.log({"loss": loss_val, "step": step}, step=step)

        # Generate samples
        if step % config.training.sample_freq == 0:
            print(f"Step {step}: Generating samples...")
            rng, sample_rng = jax.random.split(rng)

            # Unreplicate for sampling
            state_single = flax.jax_utils.unreplicate(state)

            samples = generate_samples(
                state_single, vae, diffusion_params, config, sample_rng, num_samples=16
            )

            save_path = os.path.join(config.samples_dir, f"step_{step}.png")
            save_image_grid(samples, save_path)
            print(f"Saved samples to {save_path}")

            if use_wandb:
                wandb.log({"samples": wandb.Image(save_path)}, step=step)

        # Save checkpoint
        if step % config.training.snapshot_freq == 0:
            print(f"Step {step}: Saving checkpoint...")
            state_single = flax.jax_utils.unreplicate(state)

            # Update EMA before saving
            state_single = update_ema(state_single, config.training.ema_decay)

            ckpt_manager.save(step, state_single)

            # Re-replicate
            state = flax.jax_utils.replicate(state_single)

    print("\nTraining complete!")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
