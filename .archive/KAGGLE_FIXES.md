# Kaggle Common Errors & Quick Fixes

## Error 1: ModuleNotFoundError: No module named 'lmdb'

**Error message:**
```
ModuleNotFoundError: No module named 'lmdb'
```

**Quick Fix:**
```python
# Run this in a Kaggle cell
!pip install -q lmdb protobuf
```

**Permanent Fix:**
The updated `requirements.txt` already includes `lmdb`. Just run:
```bash
!pip install -r requirements.txt
```

---

## Error 2: Tensorboard/Protobuf Compatibility Warnings

**Error message:**
```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
ImportError: cannot import name 'notf' from 'tensorboard.compat'
```

**Quick Fix:**
```python
# Downgrade protobuf to compatible version
!pip install -q "protobuf>=3.20.0,<4.0.0"
```

---

## Error 3: CUDA Warnings (Can Ignore)

**Warning messages:**
```
E0000 00:00:xxx Unable to register cuFFT factory
E0000 00:00:xxx Unable to register cuDNN factory
E0000 00:00:xxx Unable to register cuBLAS factory
```

**Status:** These are **warnings only**, not errors. They don't affect training.

**If you want to suppress them:**
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

## Error 4: Dataset Not Found

**Error message:**
```
ERROR: Dataset not found at /kaggle/input/plantdisease/PlantVillage
```

**Fix:**
1. In your Kaggle notebook, click "Add data"
2. Search for "plantdisease" or "PlantVillage"
3. Add the dataset
4. Verify path:
```python
!ls /kaggle/input/
!ls /kaggle/input/plantdisease/
```

---

## Error 5: Config File Not Found

**Error message:**
```
FileNotFoundError: configs/plantvillage.yml
```

**Fix:**
```python
# Make sure you're in the right directory
%cd /kaggle/working/ddim
!pwd
!ls configs/
```

---

## Complete Installation Script

If you encounter multiple errors, run this complete setup:

```python
# Cell 1: Complete Setup
import os

# Change to working directory
os.chdir('/kaggle/working')

# Clone if not already cloned
if not os.path.exists('ddim'):
    !git clone https://github.com/YOUR_USERNAME/ddim.git

os.chdir('ddim')

# Install all dependencies with fixes
!pip install -q torch torchvision numpy Pillow PyYAML tqdm pandas wandb
!pip install -q lmdb "protobuf>=3.20.0,<4.0.0" "tensorboard>=2.4.0"

# Verify installation
print("\n✓ Installation complete!")
print(f"Current directory: {os.getcwd()}")
print(f"Config exists: {os.path.exists('configs/plantvillage.yml')}")
```

---

## Minimal Dependencies (If Full Install Fails)

```python
# Absolute minimum to run training
!pip install -q torch torchvision numpy Pillow PyYAML tqdm lmdb
!pip install -q "protobuf>=3.20.0,<4.0.0"

# Without wandb (disable in config)
# Without tensorboard (will use basic logging)
```

Then edit `configs/plantvillage.yml`:
```yaml
wandb:
  enabled: False
```

---

## Verify Everything Works

```python
# Test script
!python test_plantvillage.py --data_root /kaggle/input/plantdisease/PlantVillage
```

---

## Emergency: Run Without Script

If `train_kaggle.sh` fails, run directly:

```python
# Direct Python command
!python main.py \
    --config plantvillage.yml \
    --doc kaggle_run \
    --ni \
    --exp /kaggle/working/exp
```

---

## Check GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Expected output:**
```
CUDA available: True
CUDA device: Tesla T4
Number of GPUs: 2
```

---

## Common Issues Checklist

Before starting training, verify:

- [ ] GPU is enabled (Settings → Accelerator → GPU)
- [ ] Dataset is added and path is correct
- [ ] All dependencies installed (`pip list | grep -E "torch|lmdb|protobuf"`)
- [ ] In correct directory (`pwd` should show `/kaggle/working/ddim`)
- [ ] Config file exists (`ls configs/plantvillage.yml`)
- [ ] Test passes (`python test_plantvillage.py`)

---

## Still Having Issues?

### Debug Mode

Run with more verbose logging:

```python
!python main.py \
    --config plantvillage.yml \
    --doc debug_run \
    --verbose debug \
    --ni \
    --exp /kaggle/working/exp
```

### Check Logs

```python
# View detailed error
!cat /kaggle/working/exp/logs/*/stdout.txt
```

### Minimal Test

Create a minimal config to test:

```python
# Save minimal config
minimal_config = """
data:
    dataset: "PlantVillage"
    data_root: "/kaggle/input/plantdisease/PlantVillage"
    image_size: 32
    channels: 3
    random_flip: false
    rescaled: true
    num_workers: 2

model:
    type: "simple"
    in_channels: 3
    out_ch: 3
    ch: 64
    ch_mult: [1, 2]
    num_res_blocks: 1
    attn_resolutions: []
    dropout: 0.1
    var_type: fixedlarge
    ema: False
    resamp_with_conv: True
    conditional: True
    num_classes: 38

diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 100

training:
    batch_size: 4
    n_epochs: 1
    snapshot_freq: 10

sampling:
    batch_size: 4

optim:
    weight_decay: 0.0
    optimizer: "Adam"
    lr: 0.0002
    beta1: 0.9
    amsgrad: false
    eps: 0.00000001
    grad_clip: 1.0

wandb:
    enabled: False
"""

with open('configs/minimal_test.yml', 'w') as f:
    f.write(minimal_config)

# Run minimal test
!python main.py --config minimal_test.yml --doc minimal_test --ni --exp /tmp/test
```

If this works, gradually increase complexity (batch_size, epochs, model size, etc.)

---

## Summary of Fixes Applied

✅ Added `lmdb` to requirements.txt
✅ Fixed protobuf version compatibility
✅ Made FFHQ/LSUN imports lazy (only load when needed)
✅ Updated train_kaggle.sh with proper dependency installation
✅ Added error handling and verbose logging

**Latest code includes all fixes!** Just pull the latest version:
```bash
!git pull origin main
```
