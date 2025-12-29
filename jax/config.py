"""Configuration for JAX training."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DataConfig:
    data_root: str = "/kaggle/input/plantdisease/PlantVillage"
    image_size: int = 256
    latent_size: int = 32
    channels: int = 4
    num_classes: int = 15
    train_ratio: float = 0.85
    val_ratio: float = 0.075
    test_ratio: float = 0.075
    split_seed: int = 42
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    color_jitter_strength: float = 0.1
    use_weighted_sampler: bool = True


@dataclass
class ModelConfig:
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16, 32)
    dropout: float = 0.0
    class_dropout: float = 0.1  # For CFG training
    ema: bool = True
    ema_rate: float = 0.9999


@dataclass
class DiffusionConfig:
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    prediction_type: str = "epsilon"


@dataclass
class OptimConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    n_iters: int = 200000
    n_epochs: int = 10000
    warmup_steps: int = 2000

    # Logging & checkpoints
    log_freq: int = 100
    sample_freq: int = 10000
    snapshot_freq: int = 25000
    validation_freq: int = 25000
    num_devices: int = 8


@dataclass
class SamplingConfig:
    num_inference_steps: int = 50
    cfg_scale: float = 3.0
    batch_size: int = 16
    last_only: bool = True


@dataclass
class FIDConfig:
    num_samples: int = 500
    real_stats_path: Optional[str] = None


@dataclass
class VAEConfig:
    model_id: str = "stabilityai/sd-vae-ft-mse"


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "ddim-plantvillage-jax"
    entity: Optional[str] = None
    name: str = "class-conditional-tpu"
    tags: Tuple[str, ...] = ("plantvillage", "latent", "conditional", "tpu")
    notes: str = "Class-conditional latent diffusion on TPU"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fid: FIDConfig = field(default_factory=FIDConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Paths (TPU-specific defaults)
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    samples_dir: str = "/kaggle/working/samples"
    log_path: str = "/kaggle/working/logs"

    # Training precision
    training_precision: str = "bf16"


def _update_from_dict(obj, data):
    """Recursively update dataclass from dict."""
    for key, value in data.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if isinstance(value, dict) and hasattr(attr, '__dataclass_fields__'):
                # Recursively update nested dataclass
                _update_from_dict(attr, value)
            else:
                # Handle tuple conversion for ch_mult, attn_resolutions, tags
                if key in ['ch_mult', 'attn_resolutions', 'tags'] and isinstance(value, list):
                    setattr(obj, key, tuple(value))
                else:
                    setattr(obj, key, value)


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get config. If config_path is provided, load from YAML and merge with defaults.
    Otherwise return default config.
    """
    config = Config()

    if config_path is not None:
        # Handle both absolute and relative paths
        if not os.path.isabs(config_path):
            # Assume it's in configs/ directory
            config_path = os.path.join("configs", config_path)

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Update config from YAML
            _update_from_dict(config, yaml_config)
            print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found. Using defaults.")

    return config
