# Kaggle TPU Setup Guide for DDIM Training

## Kaggle Notebook Setup

### Cell 1: Environment Variables (MUST RUN FIRST!)
```python
# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['JAX_PLATFORMS'] = 'tpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("✓ Environment variables set")
```

### Cell 2: Install Dependencies
```python
# Install required packages
!pip install -q flax==0.8.0 optax==0.1.9
!pip install -q diffusers==0.25.1 transformers
!pip install -q tensorflow-hub scipy einops
!pip install -q wandb PyYAML tqdm

print("✓ Dependencies installed")
```

### Cell 3: Clone Repository
```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/ddim.git /kaggle/working/ddim

# Or upload your code as a dataset and copy it
# !cp -r /kaggle/input/your-code-dataset/* /kaggle/working/ddim/

%cd /kaggle/working/ddim/jax
print("✓ Repository cloned")
```

### Cell 4: Verify TPU Setup
```python
import jax
import jax.numpy as jnp

# Verify TPU devices
devices = jax.devices()
print(f"JAX devices: {devices}")
print(f"Device count: {jax.device_count()}")
print(f"Local devices: {jax.local_devices()}")

# Test TPU
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"✓ TPU test passed: {y.shape}")
```

### Cell 5: Setup Directories
```python
import os

# Create necessary directories
dirs = [
    "/kaggle/working/logs",
    "/kaggle/working/checkpoints",
    "/kaggle/working/samples",
    "/kaggle/working/fid_stats"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✓ Created {d}")
```

### Cell 6: Configure Wandb (Optional)
```python
import wandb

# Login to wandb
wandb.login(key="YOUR_WANDB_API_KEY")

# Or use anonymous mode
# wandb.init(anonymous="allow")

print("✓ Wandb configured")
```

### Cell 7: Verify VAE Loading
```python
from utils.vae import create_vae

# Test VAE loading
print("Loading VAE...")
vae = create_vae("stabilityai/sd-vae-ft-mse")
print("✓ VAE loaded successfully")
```

### Cell 8: Compute FID Stats (One-time)
```python
# Compute FID statistics for PlantVillage dataset
# This only needs to be run once

!python compute_fid_stats.py \
    --config plantvillage_latent.yml \
    --split val \
    --num_samples 500 \
    --output /kaggle/working/fid_stats/plantvillage_val_fid_stats.npz

print("✓ FID stats computed")
```

### Cell 9: Start Training
```python
# Start training with default config
!python train_tpu.py \
    --config plantvillage_latent.yml \
    --doc plantvillage_tpu_experiment

# Or with custom config
# !python train_tpu.py \
#     --config custom_config.yml \
#     --doc my_experiment
```

### Cell 10: Monitor Training (Optional)
```python
# View latest samples
from IPython.display import Image, display
import glob

sample_files = sorted(glob.glob("/kaggle/working/logs/*/samples/*.png"))
if sample_files:
    latest = sample_files[-1]
    print(f"Latest sample: {latest}")
    display(Image(latest))
else:
    print("No samples yet")
```

### Cell 11: Resume Training (If needed)
```python
# Training will auto-resume if checkpoints exist
# Just run the same command again
!python train_tpu.py \
    --config plantvillage_latent.yml \
    --doc plantvillage_tpu_experiment
```

---

## Kaggle Dataset Setup

### Option 1: Use Public PlantVillage Dataset
Add this Kaggle dataset to your notebook:
- **Dataset**: `emmarex/plantdisease`
- **Path**: `/kaggle/input/plantdisease/PlantVillage`

Then update `configs/plantvillage_latent.yml`:
```yaml
data:
  data_root: "/kaggle/input/plantdisease/PlantVillage"
```

### Option 2: Upload Your Own Dataset
1. Create a new Kaggle dataset with your data
2. Add it to your notebook
3. Update `data_root` in config

---

## Important Notes

### TPU Considerations
1. **Memory**: TPU v5e-8 has 16GB HBM per core (128GB total)
2. **Batch Size**: Adjust based on memory usage
3. **Restart**: If TPU gets stuck, restart the notebook

### File Persistence
- `/kaggle/working/`: Saved after notebook run (up to 20GB)
- `/kaggle/temp/`: Not saved
- Checkpoints and logs will be saved in `/kaggle/working/`

### Wandb Integration
- Set `wandb.enabled: true` in config
- Add your API key in Cell 6
- Or use environment variable: `WANDB_API_KEY`

### Debugging
If you encounter errors:
```python
# Check JAX installation
!python -c "import jax; print(jax.__version__)"

# Check devices
!python -c "import jax; print(jax.devices())"

# View logs
!tail -n 50 /kaggle/working/logs/plantvillage_tpu_experiment/stdout.txt
```

---

## Recommended Workflow

1. **First Run** (Setup):
   - Run Cells 1-8 to setup environment
   - Compute FID stats (Cell 8)

2. **Training**:
   - Run Cell 9 to start training
   - Monitor with Cell 10

3. **Resume** (if interrupted):
   - Just run Cell 9 again
   - Training auto-resumes from last checkpoint

4. **Save Results**:
   - Download checkpoints from `/kaggle/working/logs/`
   - Download samples from `/kaggle/working/logs/*/samples/`
