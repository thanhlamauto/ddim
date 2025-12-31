#!/bin/bash

echo "=== Setting up DDIM environment ==="

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy>=1.19.0
pip install Pillow>=8.0.0
pip install PyYAML>=5.3.0
pip install tqdm>=4.50.0
pip install tensorboard>=2.4.0
pip install wandb>=0.12.0
pip install pandas>=1.1.0
pip install lmdb>=1.2.0
pip install "protobuf>=3.20.0,<4.0.0"
pip install huggingface_hub

echo "=== Downloading datasets from HuggingFace ==="

# Download datasets
huggingface-cli download LamTNguyen/ddim-plantvillage exp.zip --repo-type dataset --local-dir /workspace
huggingface-cli download LamTNguyen/ddim-plantvillage PlantVillage_Synthetic.zip --repo-type dataset --local-dir /workspace

# Unzip datasets
echo "=== Extracting datasets ==="
cd /workspace
unzip -o exp.zip -d /workspace/ddim/
unzip -o PlantVillage_Synthetic.zip -d /workspace/

# Cleanup zip files
rm -f exp.zip PlantVillage_Synthetic.zip

echo "=== Setup complete! ==="
