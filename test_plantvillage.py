#!/usr/bin/env python3
"""
Test script to verify PlantVillage dataset loading and configuration
"""

import os
import sys
import yaml
import argparse
import torch
from datasets.plantvillage import PlantVillage
import torchvision.transforms as transforms


def test_dataset_loading(data_root):
    """Test loading the PlantVillage dataset"""
    print("=" * 60)
    print("Testing PlantVillage Dataset Loading")
    print("=" * 60)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    print(f"\nDataset path: {data_root}")

    if not os.path.exists(data_root):
        print(f"ERROR: Dataset path does not exist: {data_root}")
        return False

    # Test train split
    print("\n1. Testing train split...")
    train_dataset = PlantVillage(
        root=data_root,
        split='train',
        transform=transform,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Number of classes: {train_dataset.num_classes}")
    print(f"   Classes: {train_dataset.classes[:5]}... (showing first 5)")

    # Test val split
    print("\n2. Testing val split...")
    val_dataset = PlantVillage(
        root=data_root,
        split='val',
        transform=transform,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    print(f"   Val samples: {len(val_dataset)}")

    # Test test split
    print("\n3. Testing test split...")
    test_dataset = PlantVillage(
        root=data_root,
        split='test',
        transform=transform,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    print(f"   Test samples: {len(test_dataset)}")

    # Verify split ratios
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\n4. Verifying split ratios...")
    print(f"   Total samples: {total}")
    print(f"   Train ratio: {len(train_dataset)/total:.2%} (expected: 70%)")
    print(f"   Val ratio: {len(val_dataset)/total:.2%} (expected: 15%)")
    print(f"   Test ratio: {len(test_dataset)/total:.2%} (expected: 15%)")

    # Test loading a sample
    print("\n5. Testing sample loading...")
    img, label = train_dataset[0]
    print(f"   Image shape: {img.shape}")
    print(f"   Image type: {img.dtype}")
    print(f"   Label: {label} ({train_dataset.get_class_name(label)})")
    print(f"   Image range: [{img.min():.3f}, {img.max():.3f}]")

    # Test dataloader
    print("\n6. Testing DataLoader...")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_img, batch_label = next(iter(train_loader))
    print(f"   Batch image shape: {batch_img.shape}")
    print(f"   Batch label shape: {batch_label.shape}")
    print(f"   Batch labels: {batch_label.tolist()}")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    return True


def test_config():
    """Test loading and validating the config file"""
    print("\n" + "=" * 60)
    print("Testing Configuration File")
    print("=" * 60)

    config_path = "configs/plantvillage.yml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig file: {config_path}")
    print("\nKey settings:")
    print(f"  Dataset: {config['data']['dataset']}")
    print(f"  Image size: {config['data']['image_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Conditional: {config['model']['conditional']}")
    print(f"  Num classes: {config['model']['num_classes']}")
    print(f"  Wandb enabled: {config['wandb']['enabled']}")
    print(f"  Wandb project: {config['wandb']['project']}")

    print("\n✓ Configuration file loaded successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test PlantVillage dataset and configuration')
    parser.add_argument('--data_root', type=str,
                        default='/kaggle/input/plantdisease/PlantVillage',
                        help='Path to PlantVillage dataset')
    parser.add_argument('--skip_dataset', action='store_true',
                        help='Skip dataset loading test (useful if dataset not available)')

    args = parser.parse_args()

    # Test config
    config_ok = test_config()

    # Test dataset
    if not args.skip_dataset:
        dataset_ok = test_dataset_loading(args.data_root)
        if not dataset_ok:
            print("\n❌ Dataset test failed!")
            sys.exit(1)
    else:
        print("\nSkipping dataset test (--skip_dataset flag set)")

    if config_ok:
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        print("\nYou can now run training with:")
        print("  python main.py --config plantvillage.yml --doc plantvillage_run --ni")
        print("\nOr use the Kaggle script:")
        print("  bash train_kaggle.sh")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
