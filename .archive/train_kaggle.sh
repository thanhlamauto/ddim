#!/bin/bash

# Kaggle training script for PlantVillage DDIM
# This script is designed to run on Kaggle with minimal setup

set -e  # Exit on error

echo "================================================"
echo "PlantVillage Conditional DDIM Training - Kaggle"
echo "================================================"

# Set environment variables
export PYTHONUNBUFFERED=1

# Install additional dependencies if needed
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Fix tensorboard protobuf compatibility
echo "Fixing protobuf compatibility..."
pip install -q --upgrade "protobuf>=3.20.0,<4.0.0"

# Login to wandb (optional - set your API key in Kaggle secrets)
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Logging in to Wandb..."
    wandb login $WANDB_API_KEY
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p exp/logs
mkdir -p exp/tensorboard

# Check dataset path
DATASET_PATH="/kaggle/input/plantdisease/PlantVillage"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please ensure the PlantVillage dataset is mounted in Kaggle"
    exit 1
fi

echo "Dataset found at: $DATASET_PATH"
echo "Number of classes: $(ls -d $DATASET_PATH/*/ | wc -l)"

# Run training
echo "Starting training..."
python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run \
    --ni \
    --exp /kaggle/working/exp

echo "Training completed!"
