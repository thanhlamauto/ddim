import os
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
from collections import Counter


class PlantVillageLatent(Dataset):
    """Full PlantVillage dataset with augmentation and weighted sampling support.

    Args:
        root: Path to PlantVillage folder containing class subfolders
        split: One of {'train', 'val', 'test'}
        config: Config namespace with augmentation settings
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
    """

    def __init__(self, root, split='train', config=None,
                 train_ratio=0.85, val_ratio=0.075, test_ratio=0.075, seed=42):
        super().__init__()

        assert split in ['train', 'val', 'test']
        self.root = root
        self.split = split
        self.config = config

        # Get all classes (subfolders)
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect all samples
        all_samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            cls_idx = self.class_to_idx[cls]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if self._is_image(img_path):
                    all_samples.append((img_path, cls_idx))

        # Shuffle and split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))

        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)

        if split == 'train':
            selected = indices[:n_train]
        elif split == 'val':
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        self.samples = [all_samples[i] for i in selected]

        # Compute class weights for weighted sampling
        self.class_counts = Counter([s[1] for s in self.samples])
        self.sample_weights = self._compute_sample_weights()

        # Build transforms
        self.transform = self._build_transform()

        print(f"PlantVillageLatent {split}: {len(self.samples)} images, {self.num_classes} classes")
        print(f"  Class distribution: {dict(self.class_counts)}")

    def _is_image(self, path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts

    def _compute_sample_weights(self):
        """Compute weights for WeightedRandomSampler"""
        total = len(self.samples)
        weights = []
        for _, cls_idx in self.samples:
            # Inverse frequency weighting
            weight = total / (self.num_classes * self.class_counts[cls_idx])
            weights.append(weight)
        return weights

    def get_sampler(self):
        """Get weighted random sampler for balanced training"""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )

    def _build_transform(self):
        """Build augmentation pipeline"""
        transform_list = []
        img_size = 256  # Default
        if self.config:
            img_size = getattr(self.config.data, 'image_size', 256)

        if self.split == 'train':
            # Random crop
            if self.config and getattr(self.config.data, 'random_crop', False):
                transform_list.append(transforms.Resize(int(img_size * 1.1)))
                transform_list.append(transforms.RandomCrop(img_size))
            else:
                transform_list.append(transforms.Resize((img_size, img_size)))

            # Random horizontal flip
            if self.config and getattr(self.config.data, 'random_flip', True):
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

            # Color jitter (nhẹ)
            if self.config and getattr(self.config.data, 'color_jitter', False):
                strength = getattr(self.config.data, 'color_jitter_strength', 0.1)
                transform_list.append(transforms.ColorJitter(
                    brightness=strength,
                    contrast=strength,
                    saturation=strength,
                    hue=strength * 0.5  # Hue nhẹ hơn
                ))
        else:
            # Val/test: chỉ resize
            transform_list.append(transforms.Resize((img_size, img_size)))

        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

    def __getitem__(self, idx):
        """Get item with error handling for corrupted images."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                img_path, cls_idx = self.samples[idx]
                img = Image.open(img_path).convert('RGB')

                # Verify image can be loaded (triggers lazy loading)
                img.load()

                img = self.transform(img)
                return img, cls_idx

            except (OSError, IOError, Image.UnidentifiedImageError) as e:
                if attempt == 0:  # Only log once per corrupted image
                    print(f"Warning: Corrupted image {img_path}, trying next image...")

                # Try next sample
                idx = (idx + 1) % len(self.samples)

        # If all attempts failed, raise error
        raise RuntimeError(f"Failed to load valid image after {max_attempts} attempts")

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, idx):
        return self.classes[idx]
