"""Configuration for JAX training."""

from dataclasses import dataclass, field
from typing import Tuple


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


@dataclass
class ModelConfig:
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16, 32)
    dropout: float = 0.0
    class_dropout: float = 0.1  # For CFG training


@dataclass
class DiffusionConfig:
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_timesteps: int = 1000


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_steps: int = 200000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999

    # Logging & checkpoints
    log_freq: int = 100
    sample_freq: int = 10000
    snapshot_freq: int = 25000
    num_devices: int = 8


@dataclass
class SamplingConfig:
    num_steps: int = 50
    cfg_scale: float = 3.0
    batch_size: int = 16


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Paths
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    samples_dir: str = "/kaggle/working/samples"

    # VAE
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"

    # Wandb
    wandb_project: str = "ddim-plantvillage-jax"
    wandb_name: str = "class-conditional-tpu"


def get_config() -> Config:
    """Get default config."""
    return Config()
