"""FID calculation utilities for TPU using InceptionV3."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import flax.linen as nn
from functools import partial
import os

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available for FID calculation")


def get_fid_network():
    """Load InceptionV3 model for FID calculation.

    Returns activations from pool_3 layer (2048-d features).
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for FID calculation")

    # Load InceptionV3 from TF Hub
    inception_model = hub.load("https://tfhub.dev/tensorflow/tfgan/eval/inception/1")

    @jax.jit
    def get_activations(images):
        """
        Get InceptionV3 activations for FID.

        Args:
            images: Images in [-1, 1] range, shape [B, 299, 299, 3]

        Returns:
            activations: Features from pool_3 layer, shape [B, 2048]
        """
        # Convert JAX array to numpy for TF
        images_np = np.array(images)

        # Get activations using TF model
        activations = inception_model(images_np)
        activations = np.array(activations)

        return jnp.array(activations)

    return get_activations


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussians.

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Epsilon for numerical stability

    Returns:
        fid: Frechet Inception Distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def fid_from_stats(mu1, sigma1, mu2, sigma2):
    """Calculate FID from pre-computed statistics.

    Args:
        mu1, sigma1: Mean and covariance of generated distribution
        mu2, sigma2: Mean and covariance of real distribution

    Returns:
        fid: Frechet Inception Distance
    """
    try:
        import scipy.linalg
    except ImportError:
        raise ImportError("scipy required for FID calculation")

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_statistics_from_activations(activations):
    """Compute mean and covariance from activations.

    Args:
        activations: Array of shape [N, 2048]

    Returns:
        mu: Mean vector of shape [2048]
        sigma: Covariance matrix of shape [2048, 2048]
    """
    activations = np.array(activations)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def save_fid_stats(activations, save_path):
    """Save FID statistics to file.

    Args:
        activations: Array of shape [N, 2048]
        save_path: Path to save .npz file
    """
    mu, sigma = compute_statistics_from_activations(activations)
    np.savez(save_path, mu=mu, sigma=sigma)
    print(f"Saved FID stats to {save_path}")


def load_fid_stats(stats_path):
    """Load FID statistics from file.

    Args:
        stats_path: Path to .npz file

    Returns:
        mu, sigma: Mean and covariance
    """
    stats = np.load(stats_path)
    return stats['mu'], stats['sigma']


def preprocess_images_for_fid(images):
    """Preprocess images for FID calculation.

    Args:
        images: Images in [0, 1] range, shape [B, H, W, 3]

    Returns:
        processed: Images in [-1, 1] range, shape [B, 299, 299, 3]
    """
    # Resize to 299x299 for InceptionV3
    images = jax.image.resize(
        images,
        (images.shape[0], 299, 299, 3),
        method='bilinear',
        antialias=False
    )

    # Scale to [-1, 1]
    images = 2 * images - 1

    return images


def compute_fid_from_images(real_images, fake_images, batch_size=128):
    """Compute FID between real and fake images.

    Args:
        real_images: Real images in [0, 1] range, shape [N, H, W, 3]
        fake_images: Fake images in [0, 1] range, shape [N, H, W, 3]
        batch_size: Batch size for processing

    Returns:
        fid: Frechet Inception Distance
    """
    get_activations = get_fid_network()

    # Process real images
    real_acts = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size]
        batch = preprocess_images_for_fid(batch)
        acts = get_activations(batch)
        real_acts.append(np.array(acts))
    real_acts = np.concatenate(real_acts, axis=0)

    # Process fake images
    fake_acts = []
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size]
        batch = preprocess_images_for_fid(batch)
        acts = get_activations(batch)
        fake_acts.append(np.array(acts))
    fake_acts = np.concatenate(fake_acts, axis=0)

    # Compute statistics
    mu_real, sigma_real = compute_statistics_from_activations(real_acts)
    mu_fake, sigma_fake = compute_statistics_from_activations(fake_acts)

    # Compute FID
    fid = fid_from_stats(mu_fake, sigma_fake, mu_real, sigma_real)

    return fid
