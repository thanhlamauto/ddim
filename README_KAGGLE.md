# PlantVillage Conditional DDIM Training on Kaggle

This guide explains how to train a conditional DDIM model on the PlantVillage dataset using Kaggle.

## Quick Start

### 1. Setup Kaggle Notebook

1. Create a new Kaggle notebook
2. Add the PlantVillage dataset:
   - Go to "Add data" in your Kaggle notebook
   - Search for "plantdisease" or add the dataset path: `/kaggle/input/plantdisease/PlantVillage`

### 2. Clone Repository and Install

```bash
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/ddim.git
%cd ddim

# Install dependencies
!pip install -r requirements.txt
```

### 3. Configure Wandb (Optional)

To enable Wandb logging:

1. Get your Wandb API key from https://wandb.ai/authorize
2. In Kaggle, go to Settings → Secrets and add `WANDB_API_KEY`
3. Or run manually in notebook:
   ```python
   import wandb
   wandb.login()
   ```

### 4. Run Training

**Option A: Using the training script (recommended)**
```bash
!bash train_kaggle.sh
```

**Option B: Direct Python command**
```bash
!python main.py --config plantvillage.yml --doc plantvillage_run --ni --exp /kaggle/working/exp
```

## Configuration

The default configuration is in `configs/plantvillage.yml`:

- **Image size**: 64x64 (adjust based on GPU memory)
- **Batch size**: 64 (reduce if OOM)
- **Classes**: 38 plant disease classes
- **Split**: 70% train / 15% val / 15% test
- **Wandb**: Enabled by default

### Key Configuration Parameters

```yaml
data:
  dataset: "PlantVillage"
  data_root: "/kaggle/input/plantdisease/PlantVillage"
  image_size: 64

model:
  conditional: True
  num_classes: 38

training:
  batch_size: 64
  n_epochs: 1000

wandb:
  enabled: True
  project: "ddim-plantvillage"
```

## Dataset Structure

The PlantVillage dataset should have this structure:
```
/kaggle/input/plantdisease/PlantVillage/
├── Pepper__bell___Bacterial_spot/
│   ├── image1.jpg
│   └── image2.jpg
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
└── ... (38 classes total)
```

## Expected Output

The training will produce:
- Checkpoints saved to `/kaggle/working/exp/logs/plantvillage_run/`
- Tensorboard logs in `/kaggle/working/exp/tensorboard/plantvillage_run/`
- Wandb logs at https://wandb.ai (if configured)

## Monitoring Training

### View Tensorboard
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/exp/tensorboard/plantvillage_run
```

### View Wandb
Visit your Wandb project page at: https://wandb.ai/YOUR_USERNAME/ddim-plantvillage

## Troubleshooting

### GPU Out of Memory
Reduce batch size in `configs/plantvillage.yml`:
```yaml
training:
  batch_size: 32  # or 16
```

### Dataset Not Found
Ensure the dataset is mounted at `/kaggle/input/plantdisease/PlantVillage`

Check with:
```bash
!ls /kaggle/input/plantdisease/PlantVillage
```

### Wandb Not Working
If Wandb fails, you can disable it in the config:
```yaml
wandb:
  enabled: False
```

## Advanced Usage

### Resume Training
```bash
!python main.py --config plantvillage.yml --doc plantvillage_run --resume_training --ni --exp /kaggle/working/exp
```

### Adjust Number of Workers
If data loading is slow:
```yaml
data:
  num_workers: 2  # Reduce for Kaggle
```

## Model Details

This implementation trains a **conditional DDIM** (Denoising Diffusion Implicit Models) on plant disease images. The model learns to generate images conditioned on disease class labels.

### Architecture
- U-Net based diffusion model
- Class-conditional generation via label embeddings
- EMA for stable training
- 1000 diffusion timesteps

### Training Features
- 70/15/15 train/val/test split
- Random horizontal flips for data augmentation
- Gradient clipping for stability
- Automatic checkpoint saving every 5000 steps
- Wandb integration for experiment tracking

## Sampling (After Training)

To generate samples from the trained model:
```bash
!python main.py --config plantvillage.yml --doc plantvillage_run --sample --ni --exp /kaggle/working/exp
```

## Citation

If you use this code, please cite the original DDIM paper:
```
@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2010.02502},
  year={2020}
}
```
