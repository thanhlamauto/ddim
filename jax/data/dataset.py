"""TensorFlow data pipeline for PlantVillage dataset."""

import os
from typing import Tuple, Optional
import tensorflow as tf
import numpy as np


def get_plantvillage_dataset(
    data_root: str,
    split: str = 'train',
    image_size: int = 256,
    batch_size: int = 64,
    num_devices: int = 8,
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    seed: int = 42,
    random_crop: bool = True,
    random_flip: bool = True,
    color_jitter: bool = True,
) -> Tuple[tf.data.Dataset, int, int]:
    """Create TensorFlow dataset for PlantVillage.

    Args:
        data_root: Path to PlantVillage folder
        split: 'train', 'val', or 'test'
        image_size: Target image size
        batch_size: Global batch size (will be split across devices)
        num_devices: Number of TPU cores
        train_ratio, val_ratio: Split ratios
        seed: Random seed
        random_crop, random_flip, color_jitter: Augmentation flags

    Returns:
        dataset: tf.data.Dataset
        num_samples: Number of samples in split
        num_classes: Number of classes
    """
    # Get all classes (subfolders)
    classes = sorted([d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    num_classes = len(classes)

    # Collect all image paths and labels
    all_paths = []
    all_labels = []
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(cls_dir, img_name))
                all_labels.append(class_to_idx[cls])

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    n_train = int(len(all_paths) * train_ratio)
    n_val = int(len(all_paths) * val_ratio)

    if split == 'train':
        paths = all_paths[:n_train]
        labels = all_labels[:n_train]
    elif split == 'val':
        paths = all_paths[n_train:n_train + n_val]
        labels = all_labels[n_train:n_train + n_val]
    else:  # test
        paths = all_paths[n_train + n_val:]
        labels = all_labels[n_train + n_val:]

    num_samples = len(paths)
    print(f"PlantVillage {split}: {num_samples} images, {num_classes} classes")

    # Compute class weights for weighted sampling
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sample_weights = sample_weights / sample_weights.sum() * len(labels)

    # Create dataset
    def load_and_preprocess(path, label, weight):
        # Load image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0

        if split == 'train':
            # Random crop
            if random_crop:
                img = tf.image.resize(img, [int(image_size * 1.1), int(image_size * 1.1)])
                img = tf.image.random_crop(img, [image_size, image_size, 3])
            else:
                img = tf.image.resize(img, [image_size, image_size])

            # Random flip
            if random_flip:
                img = tf.image.random_flip_left_right(img)

            # Color jitter
            if color_jitter:
                img = tf.image.random_brightness(img, 0.1)
                img = tf.image.random_contrast(img, 0.9, 1.1)
                img = tf.image.random_saturation(img, 0.9, 1.1)
                img = tf.clip_by_value(img, 0.0, 1.0)
        else:
            img = tf.image.resize(img, [image_size, image_size])

        return img, label

    # Create dataset with weighted sampling for training
    if split == 'train':
        # Use weighted random sampling
        paths_ds = tf.data.Dataset.from_tensor_slices(paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        weights_ds = tf.data.Dataset.from_tensor_slices(sample_weights.astype(np.float32))

        dataset = tf.data.Dataset.zip((paths_ds, labels_ds, weights_ds))
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

        # Rejection sampling for weighted sampling
        dataset = dataset.map(
            lambda p, l, w: (load_and_preprocess(p, l, w)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        paths_ds = tf.data.Dataset.from_tensor_slices(paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        weights_ds = tf.data.Dataset.from_tensor_slices(np.ones(len(paths), dtype=np.float32))

        dataset = tf.data.Dataset.zip((paths_ds, labels_ds, weights_ds))
        dataset = dataset.map(
            lambda p, l, w: load_and_preprocess(p, l, w),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Batch for TPU (batch must be divisible by num_devices)
    per_device_batch = batch_size // num_devices
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if split == 'train':
        dataset = dataset.repeat()

    return dataset, num_samples, num_classes


def prepare_batch_for_pmap(batch, num_devices: int = 8):
    """Reshape batch for pmap: [B, ...] -> [num_devices, B//num_devices, ...]"""
    images, labels = batch
    batch_size = images.shape[0]
    per_device = batch_size // num_devices

    images = images.numpy().reshape(num_devices, per_device, *images.shape[1:])
    labels = labels.numpy().reshape(num_devices, per_device)

    return images, labels
