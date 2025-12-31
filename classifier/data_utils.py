"""
Data utilities for PlantVillage dataset balancing and augmentation.
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BalancedPlantVillageDataset(Dataset):
    """
    PlantVillage dataset with balancing options.

    Supports:
    - Using dataset as-is (real_only)
    - Downsampling majority classes (real_balanced with downsample)
    - On-the-fly augmentation for minority classes (real_balanced with augment)
    - Hybrid real + synthetic (hybrid)
    """

    def __init__(self, root, split='train', scenario='real_only',
                 synth_root=None, target_count=1024, transform=None,
                 train_ratio=0.85, val_ratio=0.075, test_ratio=0.075, seed=42):
        """
        Args:
            root: Path to real PlantVillage dataset
            split: One of 'train', 'val', 'test'
            scenario: One of 'real_only', 'real_balanced', 'hybrid', 'synth_only'
            synth_root: Path to synthetic dataset (for hybrid or synth_only)
            target_count: Target number of images per class for balancing
            transform: Torchvision transforms
            train_ratio, val_ratio, test_ratio: Dataset split ratios
            seed: Random seed for reproducibility
        """
        self.root = root
        self.synth_root = synth_root
        self.split = split
        self.scenario = scenario
        self.target_count = target_count
        self.transform = transform
        self.seed = seed

        # Get class names
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Build dataset based on scenario
        self.samples = self._build_dataset(train_ratio, val_ratio, test_ratio)

        print(f"[{scenario}] {split} split: {len(self.samples)} images across {self.num_classes} classes")

    def _build_dataset(self, train_ratio, val_ratio, test_ratio):
        """Build dataset based on scenario."""
        if self.scenario == 'synth_only':
            return self._build_synth_only()
        elif self.scenario == 'real_only':
            return self._build_real_only(train_ratio, val_ratio, test_ratio)
        elif self.scenario == 'real_balanced':
            return self._build_real_balanced(train_ratio, val_ratio, test_ratio)
        elif self.scenario == 'hybrid':
            return self._build_hybrid(train_ratio, val_ratio, test_ratio)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def _get_split_indices(self, total_samples, train_ratio, val_ratio, test_ratio):
        """Get train/val/test split indices."""
        np.random.seed(self.seed)
        indices = np.random.permutation(total_samples)

        n_train = int(total_samples * train_ratio)
        n_val = int(total_samples * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        if self.split == 'train':
            return set(train_indices)
        elif self.split == 'val':
            return set(val_indices)
        else:  # test
            return set(test_indices)

    def _build_synth_only(self):
        """Build dataset from synthetic images only."""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.synth_root, class_name)
            if not os.path.exists(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

            # Split into train/val/test
            selected_indices = self._get_split_indices(len(images), 0.85, 0.075, 0.075)

            for i, img_name in enumerate(images):
                if i in selected_indices:
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))

        return samples

    def _build_real_only(self, train_ratio, val_ratio, test_ratio):
        """Build dataset from real images only (unbalanced)."""
        all_samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]

            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                all_samples.append((img_path, class_idx))

        # Global split across all classes
        selected_indices = self._get_split_indices(len(all_samples), train_ratio, val_ratio, test_ratio)

        samples = [all_samples[i] for i in range(len(all_samples)) if i in selected_indices]
        return samples

    def _build_real_balanced(self, train_ratio, val_ratio, test_ratio):
        """
        Build balanced dataset from real images only.
        - Classes with >target_count: random downsample
        - Classes with <target_count: augment during training (via repeated sampling)
        """
        samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]

            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            n_images = len(images)

            # Downsample if too many
            if n_images > self.target_count:
                random.seed(self.seed)
                images = random.sample(images, self.target_count)

            # Split into train/val/test
            selected_indices = self._get_split_indices(len(images), train_ratio, val_ratio, test_ratio)

            class_samples = []
            for i, img_name in enumerate(images):
                if i in selected_indices:
                    img_path = os.path.join(class_dir, img_name)
                    class_samples.append((img_path, class_idx))

            # If training split and class is minority, repeat samples to reach target
            if self.split == 'train' and len(class_samples) < self.target_count * train_ratio:
                target_train_count = int(self.target_count * train_ratio)
                while len(class_samples) < target_train_count:
                    class_samples.append(random.choice(class_samples))

            samples.extend(class_samples)

        return samples

    def _build_hybrid(self, train_ratio, val_ratio, test_ratio):
        """
        Build hybrid dataset (real + synthetic).
        - Classes with ≥target_count real: use only real (downsampled)
        - Classes with <target_count real: use all real + synthetic to reach target
        """
        samples = []

        for class_name in self.classes:
            real_class_dir = os.path.join(self.root, class_name)
            synth_class_dir = os.path.join(self.synth_root, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get real images
            real_images = [os.path.join(real_class_dir, f)
                           for f in os.listdir(real_class_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
            n_real = len(real_images)

            class_images = []

            if n_real >= self.target_count:
                # Use only real (downsample)
                random.seed(self.seed)
                class_images = random.sample(real_images, self.target_count)
            else:
                # Use all real + synthetic to fill
                class_images = real_images.copy()

                # Add synthetic images
                if os.path.exists(synth_class_dir):
                    synth_images = [os.path.join(synth_class_dir, f)
                                    for f in os.listdir(synth_class_dir)
                                    if f.endswith(('.jpg', '.png', '.jpeg'))]
                    n_needed = self.target_count - n_real
                    random.seed(self.seed)
                    class_images.extend(random.sample(synth_images, min(n_needed, len(synth_images))))

            # Split into train/val/test
            selected_indices = self._get_split_indices(len(class_images), train_ratio, val_ratio, test_ratio)

            for i, img_path in enumerate(class_images):
                if i in selected_indices:
                    samples.append((img_path, class_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get item with error handling for corrupted images."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert('RGB')

                # Verify image can be loaded (triggers lazy loading)
                image.load()

                if self.transform:
                    image = self.transform(image)

                return image, label

            except (OSError, IOError, Image.UnidentifiedImageError) as e:
                if attempt == 0:  # Only log once per corrupted image
                    print(f"Warning: Corrupted image {img_path}, trying next image...")

                # Try next sample
                idx = (idx + 1) % len(self.samples)

        # If all attempts failed, raise error
        raise RuntimeError(f"Failed to load valid image after {max_attempts} attempts")


def get_transforms(split='train', image_size=256):
    """
    Get data transforms for PlantVillage classification.

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size

    Returns:
        torchvision.transforms.Compose
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <scenario>")
        print("Scenarios: real_only, real_balanced, hybrid, synth_only")
        sys.exit(1)

    scenario = sys.argv[1]
    root = "/workspace/PlantVillage"
    synth_root = "/workspace/PlantVillage_Synthetic"

    dataset = BalancedPlantVillageDataset(
        root=root,
        split='train',
        scenario=scenario,
        synth_root=synth_root,
        target_count=1024,
        transform=get_transforms('train')
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    # Check class distribution
    from collections import Counter
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for cls_idx, count in sorted(class_counts.items()):
        print(f"  {dataset.classes[cls_idx]}: {count}")
