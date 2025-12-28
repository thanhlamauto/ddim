import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class PotatoLateBlight(Dataset):
    """Single class dataset for Potato Late Blight images.

    Args:
        root: Path to Potato___Late_blight folder
        split: One of {'train', 'val', 'test'}
        transform: Image transforms
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed for reproducible splits
    """

    def __init__(self, root, split='train', transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        super().__init__()

        assert split in ['train', 'val', 'test']
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        self.root = root
        self.split = split
        self.transform = transform

        # Collect all image paths
        all_images = []
        for img_name in os.listdir(root):
            img_path = os.path.join(root, img_name)
            if self._is_image_file(img_path):
                all_images.append(img_path)

        all_images = sorted(all_images)

        # Shuffle and split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))

        n_train = int(len(all_images) * train_ratio)
        n_val = int(len(all_images) * val_ratio)

        if split == 'train':
            selected = indices[:n_train]
        elif split == 'val':
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        self.samples = [all_images[i] for i in selected]
        print(f"PotatoLateBlight {split}: {len(self.samples)} images")

    def _is_image_file(self, path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # Return dummy label for compatibility

    def __len__(self):
        return len(self.samples)
