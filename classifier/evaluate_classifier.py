"""
Evaluate trained ResNet-18 classifier on PlantVillage test set.
"""

import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from models import create_resnet18
from data_utils import BalancedPlantVillageDataset, get_transforms


def evaluate(model, loader, device, class_names):
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Overall accuracy
    acc = 100. * correct / total

    # Per-class metrics
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds,
                                    target_names=class_names,
                                    output_dict=True,
                                    zero_division=0)

    return acc, cm, report, all_preds, all_targets


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 on PlantVillage')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/workspace/PlantVillage',
                        help='Path to PlantVillage test set (always use real data)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Training set ratio (must match training)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (must match training)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (must match training)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test dataset (always from real data)
    print(f"\nCreating test dataset from real data...")
    print(f"Using split ratios: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, Test={args.test_ratio:.1%}")

    test_dataset = BalancedPlantVillageDataset(
        root=args.data_root,
        split='test',
        scenario='real_only',  # Always test on real data
        synth_root=None,
        transform=get_transforms('test'),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test set: {len(test_dataset)} images")
    print(f"Number of classes: {test_dataset.num_classes}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = create_resnet18(num_classes=test_dataset.num_classes, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # Evaluate
    print("\nEvaluating on test set...")
    test_acc, cm, report, preds, targets = evaluate(
        model, test_loader, device, test_dataset.classes
    )

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Print per-class results
    print("\nPer-class metrics:")
    print("=" * 80)
    print(f"{'Class':<50} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("=" * 80)

    for class_name in test_dataset.classes:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<50} "
                  f"{metrics['precision']*100:>9.2f}% "
                  f"{metrics['recall']*100:>9.2f}% "
                  f"{metrics['f1-score']*100:>9.2f}%")

    print("=" * 80)
    print(f"{'Macro Average':<50} "
          f"{report['macro avg']['precision']*100:>9.2f}% "
          f"{report['macro avg']['recall']*100:>9.2f}% "
          f"{report['macro avg']['f1-score']*100:>9.2f}%")
    print(f"{'Weighted Average':<50} "
          f"{report['weighted avg']['precision']*100:>9.2f}% "
          f"{report['weighted avg']['recall']*100:>9.2f}% "
          f"{report['weighted avg']['f1-score']*100:>9.2f}%")
    print("=" * 80)

    # Save results
    results = {
        'test_accuracy': test_acc,
        'checkpoint': args.checkpoint,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': test_dataset.classes
    }

    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, test_dataset.classes, cm_path)

    # Save predictions
    np.save(os.path.join(args.output_dir, 'predictions.npy'), np.array(preds))
    np.save(os.path.join(args.output_dir, 'targets.npy'), np.array(targets))

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
