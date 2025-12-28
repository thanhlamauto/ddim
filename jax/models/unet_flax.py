"""U-Net model for diffusion in Flax/Linen."""

import math
from typing import Tuple, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


def get_timestep_embedding(timesteps: jnp.ndarray, embedding_dim: int) -> jnp.ndarray:
    """Sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    return emb


class ResnetBlock(nn.Module):
    """ResNet block with timestep conditioning."""
    out_channels: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, temb, train: bool = True):
        in_channels = x.shape[-1]
        h = x

        # First conv
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # Add timestep embedding
        temb = nn.Dense(self.out_channels)(nn.swish(temb))
        h = h + temb[:, None, None, :]

        # Second conv
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout, deterministic=not train)(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # Shortcut
        if in_channels != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)

        return x + h


class AttnBlock(nn.Module):
    """Self-attention block."""

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=32)(x)

        # QKV
        q = nn.Conv(C, kernel_size=(1, 1))(h)
        k = nn.Conv(C, kernel_size=(1, 1))(h)
        v = nn.Conv(C, kernel_size=(1, 1))(h)

        # Reshape for attention
        q = q.reshape(B, H * W, C)
        k = k.reshape(B, H * W, C)
        v = v.reshape(B, H * W, C)

        # Attention
        scale = C ** -0.5
        attn = jnp.einsum('bic,bjc->bij', q, k) * scale
        attn = nn.softmax(attn, axis=-1)

        # Apply attention
        h = jnp.einsum('bij,bjc->bic', attn, v)
        h = h.reshape(B, H, W, C)

        # Project out
        h = nn.Conv(C, kernel_size=(1, 1))(h)

        return x + h


class Downsample(nn.Module):
    """Downsample with conv."""

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        x = nn.Conv(C, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        return x


class Upsample(nn.Module):
    """Upsample with conv."""

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
        x = nn.Conv(C, kernel_size=(3, 3), padding='SAME')(x)
        return x


class UNet(nn.Module):
    """U-Net for diffusion models.

    Args:
        ch: Base channel count
        ch_mult: Channel multipliers per resolution
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        num_classes: Number of classes for conditioning (+1 for null class)
        dropout: Dropout rate
    """
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16, 32)
    num_classes: int = 15
    dropout: float = 0.0
    in_channels: int = 4
    out_channels: int = 4

    @nn.compact
    def __call__(self, x, t, y, train: bool = True):
        """Forward pass.

        Args:
            x: Input tensor [B, H, W, C]
            t: Timesteps [B]
            y: Class labels [B]
            train: Training mode
        """
        B, H, W, C = x.shape
        num_resolutions = len(self.ch_mult)
        temb_ch = self.ch * 4

        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = nn.Dense(temb_ch)(temb)
        temb = nn.swish(temb)
        temb = nn.Dense(temb_ch)(temb)

        # Class embedding (+1 for null class)
        class_emb = nn.Embed(num_embeddings=self.num_classes + 1, features=temb_ch)(y)
        temb = temb + class_emb

        # Downsampling
        hs = []
        h = nn.Conv(self.ch, kernel_size=(3, 3), padding='SAME')(x)
        hs.append(h)

        curr_res = H
        for i_level in range(num_resolutions):
            out_ch = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(out_ch, dropout=self.dropout)(h, temb, train)
                if curr_res in self.attn_resolutions:
                    h = AttnBlock()(h)
                hs.append(h)

            if i_level != num_resolutions - 1:
                h = Downsample()(h)
                curr_res = curr_res // 2
                hs.append(h)

        # Middle
        h = ResnetBlock(out_ch, dropout=self.dropout)(h, temb, train)
        h = AttnBlock()(h)
        h = ResnetBlock(out_ch, dropout=self.dropout)(h, temb, train)

        # Upsampling
        for i_level in reversed(range(num_resolutions)):
            out_ch = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                h = jnp.concatenate([h, hs.pop()], axis=-1)
                h = ResnetBlock(out_ch, dropout=self.dropout)(h, temb, train)
                if curr_res in self.attn_resolutions:
                    h = AttnBlock()(h)

            if i_level != 0:
                h = Upsample()(h)
                curr_res = curr_res * 2

        # Output
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        return h
