#!/bin/bash

# Resume training script for PlantVillage 100steps experiment
# Extends training from 100k to 160k steps

LOG_DIR="/workspace/ddim/exp/logs/plantvillage_100steps"
CONFIG_FILE="${LOG_DIR}/config.yml"

echo "=== Resuming Training ==="
echo "Checkpoint directory: ${LOG_DIR}"
echo ""

# Check if checkpoint exists
if [ ! -f "${LOG_DIR}/ckpt.pth" ]; then
    echo "ERROR: Checkpoint not found at ${LOG_DIR}/ckpt.pth"
    exit 1
fi

echo "Checkpoint found. Current status:"
tail -n 5 "${LOG_DIR}/stdout.txt"
echo ""

# Update n_iters in config.yml to 160000
echo "Updating config to extend training to 160k steps..."
python3 << 'EOF'
import yaml
import argparse

config_path = "/workspace/ddim/exp/logs/plantvillage_100steps/config.yml"

# Load config
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# Update n_iters
if hasattr(config_dict['training'], 'n_iters'):
    old_iters = config_dict['training'].n_iters
else:
    old_iters = config_dict['training'].__dict__.get('n_iters', 'unknown')

# Modify the namespace object
config_dict['training'].__dict__['n_iters'] = 160000

# Save back
with open(config_path, 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False)

print(f"Updated n_iters: {old_iters} -> 160000")
EOF

echo ""
echo "Starting training resume..."
echo "This will train from step 100k to 160k (60k additional steps)"
echo ""

# Run training with resume flag
cd /workspace/ddim
python main.py \
    --config plantvillage_latent.yml \
    --doc plantvillage_100steps \
    --exp exp \
    --latent_cond \
    --resume_training \
    --ni

echo ""
echo "Training complete!"
