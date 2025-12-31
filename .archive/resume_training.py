#!/usr/bin/env python3
"""
Resume training script for PlantVillage diffusion model.
Updates config and resumes from checkpoint.
"""

import os
import yaml
import argparse
import subprocess
import sys


def update_config_iters(config_path, new_iters):
    """Update n_iters in config.yml file"""
    # Load the config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Get old value
    old_iters = config_dict['training'].__dict__.get('n_iters', 'unknown')

    # Update n_iters
    config_dict['training'].__dict__['n_iters'] = new_iters

    # Save back
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"✓ Updated config: n_iters {old_iters} -> {new_iters}")
    return old_iters, new_iters


def check_checkpoint(log_dir):
    """Verify checkpoint exists and get current step"""
    ckpt_path = os.path.join(log_dir, "ckpt.pth")

    if not os.path.exists(ckpt_path):
        print(f"✗ ERROR: Checkpoint not found at {ckpt_path}")
        return None

    # Try to load and check step
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        current_step = ckpt[2]  # Step is 3rd element in checkpoint
        print(f"✓ Checkpoint found at step {current_step}")
        return current_step
    except Exception as e:
        print(f"✓ Checkpoint found (step unknown: {e})")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default='/workspace/ddim/exp/logs/plantvillage_100steps',
                        help='Path to log directory with checkpoint')
    parser.add_argument('--target_steps', type=int, default=160000,
                        help='Target number of training steps')
    parser.add_argument('--config_name', type=str, default='plantvillage_latent.yml',
                        help='Original config file name')
    parser.add_argument('--doc_name', type=str, default='plantvillage_100steps',
                        help='Experiment doc name')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only update config, do not start training')

    args = parser.parse_args()

    print("=" * 60)
    print("Resume Training Script")
    print("=" * 60)
    print(f"Log directory: {args.log_dir}")
    print(f"Target steps: {args.target_steps}")
    print()

    # Check checkpoint exists
    current_step = check_checkpoint(args.log_dir)
    if current_step is None:
        return 1

    # Update config
    config_path = os.path.join(args.log_dir, "config.yml")
    if not os.path.exists(config_path):
        print(f"✗ ERROR: Config not found at {config_path}")
        return 1

    old_iters, new_iters = update_config_iters(config_path, args.target_steps)

    if args.dry_run:
        print("\n✓ Dry run complete. Config updated but training not started.")
        return 0

    # Start training
    print()
    print("=" * 60)
    print(f"Starting training: step {current_step} -> {args.target_steps}")
    print("=" * 60)
    print()

    cmd = [
        sys.executable, "main.py",
        "--config", args.config_name,
        "--doc", args.doc_name,
        "--exp", "exp",
        "--latent_cond",
        "--resume_training",
        "--ni"
    ]

    # Change to ddim directory
    os.chdir('/workspace/ddim')

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
