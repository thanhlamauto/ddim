#!/bin/bash
# Quick setup script for Kaggle TPU environment
# Run this after starting the notebook

set -e

echo "=========================================="
echo "DDIM Training Setup for Kaggle TPU"
echo "=========================================="

# Step 1: Create directories
echo ""
echo "Step 1: Creating directories..."
mkdir -p /kaggle/working/logs
mkdir -p /kaggle/working/checkpoints
mkdir -p /kaggle/working/samples
mkdir -p /kaggle/working/fid_stats
echo "✓ Directories created"

# Step 2: Verify JAX TPU
echo ""
echo "Step 2: Verifying JAX TPU setup..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"
echo "✓ JAX TPU verified"

# Step 3: Test VAE loading (if code is ready)
if [ -f "utils/vae.py" ]; then
    echo ""
    echo "Step 3: Testing VAE loading..."
    python3 -c "from utils.vae import create_vae; vae = create_vae('stabilityai/sd-vae-ft-mse'); print('✓ VAE loaded successfully')"
else
    echo "⚠ Skipping VAE test (utils/vae.py not found)"
fi

# Step 4: Check dataset
echo ""
echo "Step 4: Checking dataset..."
if [ -d "/kaggle/input/plantdisease/PlantVillage" ]; then
    echo "✓ PlantVillage dataset found"
    echo "  Classes: $(ls /kaggle/input/plantdisease/PlantVillage | wc -l)"
    echo "  Sample class: $(ls /kaggle/input/plantdisease/PlantVillage | head -n 1)"
else
    echo "⚠ PlantVillage dataset not found at /kaggle/input/plantdisease/PlantVillage"
    echo "  Please add the 'emmarex/plantdisease' dataset to your notebook"
fi

# Step 5: Check config file
echo ""
echo "Step 5: Checking config..."
if [ -f "../configs/plantvillage_latent.yml" ]; then
    echo "✓ Config file found"
else
    echo "⚠ Config file not found at ../configs/plantvillage_latent.yml"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. (Optional) Compute FID stats:"
echo "   python compute_fid_stats.py --config plantvillage_latent.yml --split val --num_samples 500"
echo ""
echo "2. Start training:"
echo "   python train_tpu.py --config plantvillage_latent.yml --doc plantvillage_tpu_v1"
echo ""
