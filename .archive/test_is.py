#!/usr/bin/env python3
"""
Test Inception Score calculation on sample directories
"""

import sys
import os

sys.path.insert(0, '/workspace/ddim')

from utils.inception_score import calculate_inception_score

# Test directories
test_dirs = [
    '/workspace/ddim/exp/logs/plantvillage_100steps/real_val',
]

print("=" * 60)
print("Testing Inception Score Calculation")
print("=" * 60)
print()

for test_dir in test_dirs:
    if os.path.exists(test_dir):
        print(f"Testing: {test_dir}")
        is_mean, is_std = calculate_inception_score(
            test_dir,
            batch_size=50,
            device='cuda'
        )
        print(f"✓ IS = {is_mean:.2f} ± {is_std:.2f}")
        print()
    else:
        print(f"✗ Directory not found: {test_dir}")
        print()

print("=" * 60)
print("Test complete!")
print()
print("When training resumes, IS will be calculated automatically")
print("and logged to wandb every 20k steps.")
print("=" * 60)
