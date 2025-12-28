import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class PlantVillage(Dataset):
    """PlantVillage Dataset for plant disease classification.

    The dataset structure should be:
    root/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        ...

    Args:
        root (string): Root directory path containing class folders
        split (string): One of {'train', 'val', 'test'}. Default: 'train'
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        train_ratio (float): Ratio of training data. Default: 0.7
        val_ratio (float): Ratio of validation data. Default: 0.15
        test_ratio (float): Ratio of test data. Default: 0.15
        seed (int): Random seed for reproducible splits. Default: 42
    """

    def __init__(self, root, split='train', transform=None,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        super(PlantVillage, self).__init__()

        assert split in ['train', 'val', 'test'], \
            f"split must be one of 'train', 'val', 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"

        self.root = root
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect all image paths and labels
        all_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get all image files
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if self._is_image_file(img_path):
                    all_samples.append((img_path, class_idx))

        # Shuffle and split dataset
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))

        # Calculate split indices
        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Select samples for current split
        if split == 'train':
            selected_indices = train_indices
        elif split == 'val':
            selected_indices = val_indices
        else:  # test
            selected_indices = test_indices

        self.samples = [all_samples[i] for i in selected_indices]

        print(f"PlantVillage {split} split: {len(self.samples)} images across {self.num_classes} classes")

    def _is_image_file(self, filename):
        """Check if a file is an image"""
        img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        return os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in img_extensions

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path, target = self.samples[index]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, idx):
        """Get class name from class index"""
        return self.classes[idx]
