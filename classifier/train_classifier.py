"""
Train ResNet-18 classifier on PlantVillage dataset.
Supports multiple scenarios for ablation studies.
"""

import os
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models import create_resnet18
from data_utils import BalancedPlantVillageDataset, get_transforms


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on PlantVillage')
    parser.add_argument('--data_root', type=str, default='/workspace/PlantVillage',
                        help='Path to real PlantVillage dataset')
    parser.add_argument('--synth_root', type=str, default='/workspace/PlantVillage_Synthetic',
                        help='Path to synthetic PlantVillage dataset')
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['synth_only', 'real_only', 'real_balanced', 'hybrid'],
                        help='Training scenario')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--target_count', type=int, default=1024,
                        help='Target number of images per class for balancing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Training set ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print(f"\nCreating datasets for scenario: {args.scenario}")
    print(f"Split ratios: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, Test={args.test_ratio:.1%}")

    train_dataset = BalancedPlantVillageDataset(
        root=args.data_root,
        split='train',
        scenario=args.scenario,
        synth_root=args.synth_root,
        target_count=args.target_count,
        transform=get_transforms('train'),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    val_dataset = BalancedPlantVillageDataset(
        root=args.data_root,
        split='val',
        scenario=args.scenario,
        synth_root=args.synth_root,
        target_count=args.target_count,
        transform=get_transforms('val'),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")

    # Create model
    model = create_resnet18(num_classes=train_dataset.num_classes, pretrained=False)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best model saved! Val Acc: {val_acc:.2f}%")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Training complete
    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(args.output_dir, 'final_model.pth'))

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
