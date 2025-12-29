"""
Compute and save FID statistics for PlantVillage dataset.

Usage:
    python compute_fid_stats.py --config plantvillage_latent.yml --split val --num_samples 500
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['JAX_PLATFORMS'] = 'tpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import jax
import jax.numpy as jnp
import numpy as np

from config import get_config
from data.dataset import get_plantvillage_dataset
from utils.fid import (
    get_fid_network,
    preprocess_images_for_fid,
    save_fid_stats,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="plantvillage_latent.yml",
        help="Config file name (in configs/ directory)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to use for FID stats",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for FID stats (default: auto-generated)",
    )
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    # Auto-generate output path if not specified
    if args.output is None:
        output_dir = "/kaggle/working/fid_stats"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"plantvillage_{args.split}_fid_stats.npz")

    print(f"Computing FID stats for PlantVillage {args.split} split...")
    print(f"Using {args.num_samples} samples")
    print(f"Output path: {args.output}")

    # Load dataset
    print("\nLoading dataset...")
    dataset, num_samples, num_classes = get_plantvillage_dataset(
        data_root=config.data.data_root,
        split=args.split,
        image_size=config.data.image_size,
        batch_size=min(64, args.num_samples),
        num_devices=1,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
        random_crop=False,
        random_flip=False,
        color_jitter=False,
    )

    print(f"Dataset loaded: {num_samples} samples, {num_classes} classes")

    # Get InceptionV3 model
    print("\nLoading InceptionV3 model...")
    get_activations = get_fid_network()

    # Collect activations
    print("\nProcessing images...")
    all_activations = []
    collected = 0

    for batch in dataset.take(args.num_samples // 64 + 1):
        if collected >= args.num_samples:
            break

        images, _ = batch
        images = jnp.array(images.numpy())

        # Preprocess for InceptionV3
        images_processed = preprocess_images_for_fid(images)

        # Get activations
        acts = get_activations(images_processed)
        all_activations.append(np.array(acts))

        collected += images.shape[0]
        print(f"Processed {collected}/{args.num_samples} images...")

    # Concatenate and save stats
    all_activations = np.concatenate(all_activations, axis=0)[:args.num_samples]
    save_fid_stats(all_activations, args.output)

    print(f"\n✓ FID stats saved to {args.output}")
    print(f"  Shape: {all_activations.shape}")
    print(f"  Samples: {len(all_activations)}")


if __name__ == "__main__":
    main()
