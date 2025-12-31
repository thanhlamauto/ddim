#!/bin/bash
# Resume training - Simple version
# Config đã được update: 160k steps

cd /workspace/ddim

echo "=== Resume Training ==="
echo "Checkpoint: exp/logs/plantvillage_100steps"
echo "Config: configs/plantvillage_latent.yml"
echo "Target: 100k -> 160k steps (60k more)"
echo ""

# Verify config
ITERS=$(grep "n_iters:" configs/plantvillage_latent.yml | grep -o '[0-9]*')
echo "n_iters in config: $ITERS"

if [ "$ITERS" != "160000" ]; then
    echo "✗ ERROR: Config n_iters = $ITERS, expected 160000"
    echo "Updating config..."
    sed -i 's/n_iters: [0-9]*/n_iters: 160000/' configs/plantvillage_latent.yml
    echo "✓ Config updated"
fi

echo ""
echo "Starting training..."
echo ""

python main.py \
    --config plantvillage_latent.yml \
    --doc plantvillage_100steps \
    --exp exp \
    --latent_cond \
    --resume_training \
    --ni

echo ""
echo "Training finished!"
