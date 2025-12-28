#!/usr/bin/env python3
"""
One-click installation script for Kaggle
Run this first before training
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   PlantVillage DDIM - Kaggle Installation Script      ║
    ╚════════════════════════════════════════════════════════╝
    """)

    # Step 1: Install core dependencies
    if not run_command(
        "pip install -q torch torchvision numpy Pillow PyYAML tqdm pandas",
        "Installing core dependencies"
    ):
        print("❌ Failed to install core dependencies")
        return False

    # Step 2: Install additional required packages
    if not run_command(
        'pip install -q lmdb "protobuf>=3.20.0,<4.0.0"',
        "Installing lmdb and protobuf"
    ):
        print("❌ Failed to install lmdb/protobuf")
        return False

    # Step 3: Install tensorboard with fixed version
    if not run_command(
        'pip install -q "tensorboard>=2.4.0"',
        "Installing tensorboard"
    ):
        print("⚠️  Tensorboard installation failed (non-critical)")

    # Step 4: Install wandb (optional)
    if not run_command(
        "pip install -q wandb",
        "Installing wandb"
    ):
        print("⚠️  Wandb installation failed (optional)")

    # Step 5: Verify installations
    print(f"\n{'='*60}")
    print("[Verifying installations]")
    print(f"{'='*60}")

    packages_to_check = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'lmdb': 'LMDB',
    }

    all_ok = True
    for package, name in packages_to_check.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING!")
            all_ok = False

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA ({torch.cuda.get_device_name(0)})")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available (make sure GPU is enabled in Kaggle settings)")
    except:
        pass

    # Step 6: Check dataset
    print(f"\n{'='*60}")
    print("[Checking dataset]")
    print(f"{'='*60}")

    dataset_path = "/kaggle/input/plantdisease/PlantVillage"
    if os.path.exists(dataset_path):
        try:
            classes = sorted([d for d in os.listdir(dataset_path)
                            if os.path.isdir(os.path.join(dataset_path, d))])
            print(f"✓ Dataset found at {dataset_path}")
            print(f"  Number of classes: {len(classes)}")
            if len(classes) > 0:
                print(f"  Sample classes: {', '.join(classes[:3])}...")
        except Exception as e:
            print(f"⚠️  Dataset exists but couldn't read: {e}")
    else:
        print(f"✗ Dataset NOT found at {dataset_path}")
        print("  Please add the PlantVillage dataset in Kaggle")
        all_ok = False

    # Step 7: Check config file
    print(f"\n{'='*60}")
    print("[Checking configuration]")
    print(f"{'='*60}")

    if os.path.exists("configs/plantvillage.yml"):
        print("✓ Config file found: configs/plantvillage.yml")
    else:
        print("✗ Config file NOT found")
        all_ok = False

    # Final summary
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ Installation successful!")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Run training:")
        print("     !python main.py --config plantvillage.yml --doc my_run --ni")
        print("\n  2. Or use the quick script:")
        print("     !bash train_kaggle.sh")
        print("\n  3. Or test first:")
        print("     !python test_plantvillage.py --data_root /kaggle/input/plantdisease/PlantVillage")
    else:
        print("⚠️  Installation completed with warnings")
        print(f"{'='*60}")
        print("\nPlease fix the issues above before training")
        print("See KAGGLE_FIXES.md for troubleshooting")

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
