# Training on Kaggle TPU v5e-8

Complete guide for training class-conditional latent diffusion on Kaggle TPU.

## Quick Start

### 1. Create Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com)
2. Create new notebook
3. **IMPORTANT**: Enable TPU in Settings
   - Click Settings (gear icon)
   - Accelerator → TPU v3-8 or TPU v5-8
   - Language → Python

### 2. Add Dataset

Add PlantVillage dataset:
- Search for: `emmarex/plantdisease`
- Click "Add Data"
- Dataset will be at: `/kaggle/input/plantdisease/PlantVillage`

### 3. Upload Code

**Option A: From GitHub**
```python
!git clone https://github.com/YOUR_USERNAME/ddim.git /kaggle/working/ddim
```

**Option B: As Kaggle Dataset**
1. Zip your `ddim` folder
2. Upload to Kaggle as dataset
3. Add to notebook
4. Copy: `!cp -r /kaggle/input/your-dataset/ddim /kaggle/working/`

### 4. Run Setup

Upload and run `kaggle_notebook.ipynb` or run cells manually.

## Files Overview

```
jax/
├── requirements_kaggle.txt     # Dependencies for pip install
├── kaggle_notebook.ipynb       # Ready-to-use notebook
├── kaggle_setup.md            # Detailed setup guide
├── kaggle_quick_setup.sh      # Automated setup script
└── README_KAGGLE.md           # This file
```

## Installation

### Method 1: Jupyter Notebook (Recommended)

Upload `kaggle_notebook.ipynb` to Kaggle and run cells in order.

### Method 2: Manual Setup

```python
# Cell 1: Environment (MUST BE FIRST!)
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['JAX_PLATFORMS'] = 'tpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Cell 2: Install
!pip install -q -r requirements_kaggle.txt

# Cell 3: Clone code
!git clone YOUR_REPO /kaggle/working/ddim
%cd /kaggle/working/ddim/jax

# Cell 4: Verify TPU
import jax
print(jax.devices())

# Cell 5: Train
!python train_tpu.py --config plantvillage_latent.yml --doc experiment_v1
```

## Configuration

Edit `configs/plantvillage_latent.yml` for your needs:

```yaml
data:
  data_root: "/kaggle/input/plantdisease/PlantVillage"  # Kaggle dataset path
  batch_size: 64  # Adjust based on memory

training:
  n_iters: 200000      # Total training steps
  snapshot_freq: 25000  # Checkpoint frequency
  sample_freq: 10000    # Sample generation frequency
  validation_freq: 25000 # FID evaluation frequency

wandb:
  enabled: true
  project: "ddim-plantvillage-tpu"
  name: "experiment_v1"
```

## Training

### Start Training
```bash
python train_tpu.py --config plantvillage_latent.yml --doc my_experiment
```

### Resume Training
Training auto-resumes from last checkpoint. Just run the same command again:
```bash
python train_tpu.py --config plantvillage_latent.yml --doc my_experiment
```

### Monitor Progress

**View Samples:**
```python
from IPython.display import Image, display
display(Image('/kaggle/working/logs/my_experiment/samples/step_10000.png'))
```

**Check Logs:**
```bash
!tail -n 50 /kaggle/working/logs/my_experiment/stdout.txt
```

**List Checkpoints:**
```bash
!ls -lh /kaggle/working/logs/my_experiment/checkpoints/
```

## Expected Timeline (TPU v5e-8)

- **Setup**: ~5 minutes
- **FID stats**: ~10 minutes (one-time)
- **Training**:
  - 10k steps: ~1 hour
  - 50k steps: ~5 hours
  - 200k steps: ~20 hours

## Memory & Performance

### Default Config (Batch Size 64)
- Memory per core: ~4-6 GB
- Throughput: ~10 steps/sec
- Samples: Every 10k steps (~1 hour)

### High Memory Usage?
Reduce batch size in config:
```yaml
training:
  batch_size: 32  # or 48
```

### Slow Training?
- Check TPU utilization: `!nvidia-smi` won't work, use JAX profiling
- Reduce data augmentation
- Use `pcuenq/sd-vae-ft-mse-flax` for faster VAE

## Outputs

All outputs saved to `/kaggle/working/logs/{experiment_name}/`:

```
logs/my_experiment/
├── checkpoints/
│   ├── ckpt_25000.pkl      # Checkpoint at step 25k
│   ├── ckpt_50000.pkl      # Checkpoint at step 50k
│   └── ckpt.pth            # Latest checkpoint
├── samples/
│   ├── step_10000.png      # Training samples
│   ├── step_20000.png
│   └── validation_step_25000.png  # Validation samples
├── real_fid_stats.npz      # FID statistics
└── config.yml              # Training config (saved copy)
```

## Download Results

Before notebook ends, download:

```python
# Option 1: Zip and download
!zip -r /kaggle/working/results.zip /kaggle/working/logs

# Option 2: Copy to Google Drive (if mounted)
!cp -r /kaggle/working/logs /content/drive/MyDrive/
```

## Troubleshooting

### "Device or resource busy"
```python
# Restart kernel and run setup cells again
# The code has auto-retry logic built-in
```

### "Out of memory"
```yaml
# Reduce batch size in config
training:
  batch_size: 32
```

### "Dataset not found"
```python
# Check dataset path
!ls /kaggle/input/plantdisease/
# Should show: PlantVillage/

# Update config if needed
# data:
#   data_root: "/kaggle/input/plantdisease/PlantVillage"
```

### "JAX not found" or "No TPU devices"
```python
# Check TPU enabled in Settings
# Restart kernel
# Run Cell 1 (environment setup) first
```

### FID calculation fails
```python
# FID is optional, you can disable it
# Or reduce num_samples in config:
# fid:
#   num_samples: 200  # instead of 500
```

## Wandb Integration

### Setup Wandb
```python
import wandb
wandb.login(key="YOUR_API_KEY")
```

### In Config
```yaml
wandb:
  enabled: true
  project: "ddim-plantvillage"
  name: "experiment_v1"
  tags: ["tpu", "plantvillage", "v5e-8"]
```

### View Training
Go to: `https://wandb.ai/YOUR_USERNAME/ddim-plantvillage`

## Cost & Limits

- **TPU v5e-8**: Free tier includes 30 hours/week
- **Storage**: 20 GB in `/kaggle/working/`
- **Runtime**: Max 12 hours per session (can resume)
- **Checkpoints**: ~500 MB each

## Best Practices

1. ✅ **Save frequently**: Set `snapshot_freq: 5000` for more checkpoints
2. ✅ **Monitor memory**: Check logs for OOM warnings
3. ✅ **Use Wandb**: Track experiments online
4. ✅ **Test first**: Run 1000 steps to verify setup
5. ✅ **Download results**: Before session ends

## Example: Quick Test Run

```python
# 1. Setup environment
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['JAX_PLATFORMS'] = 'tpu'

# 2. Install dependencies
!pip install -q -r requirements_kaggle.txt

# 3. Get code
!git clone YOUR_REPO /kaggle/working/ddim
%cd /kaggle/working/ddim/jax

# 4. Quick test (1000 steps)
# Edit config: n_iters: 1000
!python train_tpu.py --config plantvillage_latent.yml --doc test_run

# Should complete in ~10 minutes
# Verify samples are generated
```

## Support

- **Issues**: https://github.com/YOUR_USERNAME/ddim/issues
- **Kaggle Forums**: https://www.kaggle.com/discussions
- **Documentation**: See `kaggle_setup.md`

## Credits

- JAX/Flax: https://github.com/google/jax
- Diffusers: https://github.com/huggingface/diffusers
- PlantVillage Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
