"""
PlantDoc Dataset Loader with Augmentation
Supports class-conditional training with classifier-free guidance
"""

import os
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from collections import Counter
import random


class PlantDoc(Dataset):
    """PlantDoc dataset with augmentation and weighted sampling.

    Args:
        root: Path to PlantDoc folder with train/test structure
        split: One of {'train', 'test'}
        config: Config namespace with augmentation settings
        augment_prob: Probability of applying augmentation (default: 0.25 for 25%)
    """

    def __init__(self, root, split='train', config=None, augment_prob=0.25):
        super().__init__()

        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.config = config
        self.augment_prob = augment_prob

        # Get augmentation strength
        self.augment_strength = getattr(config.data, 'augment_strength', 'normal') if config else 'normal'

        # Path to split folder
        split_path = os.path.join(root, split)

        # Get all classes (subfolders)
        self.classes = sorted([d for d in os.listdir(split_path)
                               if os.path.isdir(os.path.join(split_path, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect all samples
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(split_path, cls)
            cls_idx = self.class_to_idx[cls]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if self._is_image(img_path):
                    self.samples.append((img_path, cls_idx))

        # Compute class weights for weighted sampling
        self.class_counts = Counter([s[1] for s in self.samples])
        self.sample_weights = self._compute_sample_weights()

        # Build transforms
        self.base_transform = self._build_base_transform()

        print(f"PlantDoc {split}: {len(self.samples)} images, {self.num_classes} classes")
        print(f"  Augmentation probability: {self.augment_prob*100}%")
        print(f"  Augmentation strength: {self.augment_strength}")
        if self.split == 'train':
            if self.augment_strength == 'balanced':
                print(f"  → On-the-fly augmentation (infinite variations per epoch)")
                print(f"  → Color-preserving for disease symptoms")
            elif self.augment_prob >= 0.8 and self.augment_strength == 'strong':
                effective = len(self.samples) * 2.5
                print(f"  Effective dataset size: ~{int(effective)} samples (250% augmentation)")
        print(f"  Top 5 classes: {dict(sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"  Bottom 5 classes: {dict(sorted(self.class_counts.items(), key=lambda x: x[1])[:5])}")

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

    def _build_base_transform(self):
        """Build base transform (resize + tensor)"""
        img_size = 256  # Default
        if self.config:
            img_size = getattr(self.config.data, 'image_size', 256)

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def _apply_augmentation(self, img):
        """Apply augmentation with configurable strength

        Normal strength (light):
        - Random horizontal flip (50%)
        - Random rotation (±15 degrees, 30%)
        - Color jitter (brightness, contrast, saturation: ±10%, 40%)
        - Random affine (translate ±5%, 20%)

        Balanced strength (recommended for plant diseases):
        - Preserve color symptoms (light brightness/contrast only)
        - Reasonable geometric transforms
        - No perspective/warp that distorts leaf shapes
        - Suitable for on-the-fly infinite variations

        Strong strength (aggressive, may alter disease symptoms):
        - Use with caution for medical/agricultural datasets
        """
        img_size = img.size[0]  # Assume square after resize

        if self.augment_strength == 'balanced':
            # BALANCED AUGMENTATION for plant diseases
            # Key: preserve color info, reasonable geometry

            # Random horizontal flip (60%)
            if random.random() < 0.6:
                img = TF.hflip(img)

            # Random rotation ±15 degrees (50%) - moderate, realistic
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # LIGHT color jitter (60%) - preserve disease color
            if random.random() < 0.6:
                brightness = random.uniform(0.9, 1.1)   # ±10% only
                contrast = random.uniform(0.9, 1.1)     # ±10% only
                saturation = random.uniform(0.95, 1.05) # ±5% very light
                # NO hue shift - preserve disease color!

                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)

            # Random affine (40%) - light translation + scale
            if random.random() < 0.4:
                angle_affine = random.uniform(-5, 5)  # Small rotation
                translate = (
                    random.uniform(-0.08, 0.08) * img_size,
                    random.uniform(-0.08, 0.08) * img_size
                )
                scale = random.uniform(0.95, 1.05)  # ±5% scale
                # No shear - keeps leaf shape natural

                img = TF.affine(img, angle=angle_affine, translate=translate,
                               scale=scale, shear=0,
                               interpolation=TF.InterpolationMode.BILINEAR)

            # Very light Gaussian blur (10%) - simulate focus
            if random.random() < 0.1:
                img = TF.gaussian_blur(img, kernel_size=3)

            # Random crop and resize (30%) - simulate different framing
            if random.random() < 0.3:
                crop_size = int(img_size * random.uniform(0.9, 1.0))
                i = random.randint(0, img_size - crop_size)
                j = random.randint(0, img_size - crop_size)
                img = TF.crop(img, i, j, crop_size, crop_size)
                img = TF.resize(img, (img_size, img_size))

        elif self.augment_strength == 'strong':
            # STRONG AUGMENTATION for small datasets

            # Random horizontal flip (70%)
            if random.random() < 0.7:
                img = TF.hflip(img)

            # Random vertical flip (30%) - useful for plant diseases
            if random.random() < 0.3:
                img = TF.vflip(img)

            # Random rotation ±25 degrees (60%)
            if random.random() < 0.6:
                angle = random.uniform(-25, 25)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # Strong color jitter (70%)
            if random.random() < 0.7:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue = random.uniform(-0.1, 0.1)

                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)

            # Random affine with scale (50%)
            if random.random() < 0.5:
                angle_affine = random.uniform(-10, 10)
                translate = (
                    random.uniform(-0.1, 0.1) * img_size,
                    random.uniform(-0.1, 0.1) * img_size
                )
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5, 5)

                img = TF.affine(img, angle=angle_affine, translate=translate,
                               scale=scale, shear=shear,
                               interpolation=TF.InterpolationMode.BILINEAR)

            # Random perspective (20%)
            if random.random() < 0.2:
                width, height = img.size
                startpoints = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]
                endpoints = []
                for point in startpoints:
                    x = point[0] + random.uniform(-0.05, 0.05) * width
                    y = point[1] + random.uniform(-0.05, 0.05) * height
                    endpoints.append([int(x), int(y)])
                img = TF.perspective(img, startpoints, endpoints,
                                    interpolation=TF.InterpolationMode.BILINEAR)

            # Gaussian blur (15%)
            if random.random() < 0.15:
                kernel_size = random.choice([3, 5])
                img = TF.gaussian_blur(img, kernel_size)

        else:
            # NORMAL AUGMENTATION

            # Random horizontal flip (50%)
            if random.random() < 0.5:
                img = TF.hflip(img)

            # Random rotation ±15 degrees (30%)
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # Color jitter (40%)
            if random.random() < 0.4:
                brightness = random.uniform(0.9, 1.1)
                contrast = random.uniform(0.9, 1.1)
                saturation = random.uniform(0.9, 1.1)
                hue = random.uniform(-0.05, 0.05)

                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)

            # Random affine (translate ±5%, 20%)
            if random.random() < 0.2:
                translate = (
                    random.uniform(-0.05, 0.05) * img_size,
                    random.uniform(-0.05, 0.05) * img_size
                )
                img = TF.affine(img, angle=0, translate=translate, scale=1.0, shear=0,
                               interpolation=TF.InterpolationMode.BILINEAR)

        return img

    def __getitem__(self, idx):
        img_path, cls_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # Resize first
        img_size = 256
        if self.config:
            img_size = getattr(self.config.data, 'image_size', 256)
        img = TF.resize(img, (img_size, img_size))

        # Apply augmentation with probability
        if self.split == 'train' and random.random() < self.augment_prob:
            img = self._apply_augmentation(img)

        # Convert to tensor
        img = TF.to_tensor(img)

        return img, cls_idx

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, idx):
        return self.classes[idx]
