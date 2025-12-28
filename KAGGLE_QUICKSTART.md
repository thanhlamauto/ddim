# PlantVillage DDIM - Kaggle Quickstart

Copy từng cell dưới đây vào Kaggle notebook của bạn và chạy tuần tự.

## Prerequisites
- Kaggle notebook với GPU enabled (T4 x2 recommended)
- Dataset "plantdisease" đã được add vào notebook

---

## Cell 1: Clone Repository
```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/ddim.git
%cd ddim
!ls -la
```

---

## Cell 2: Install Dependencies
```python
# Install required packages
!pip install -q torch torchvision numpy Pillow PyYAML tqdm tensorboard wandb pandas
```

---

## Cell 3: Verify Dataset
```python
# Check if dataset is available
import os

dataset_path = "/kaggle/input/plantdisease/PlantVillage"
if os.path.exists(dataset_path):
    classes = sorted([d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"✓ Dataset found!")
    print(f"  Path: {dataset_path}")
    print(f"  Number of classes: {len(classes)}")
    print(f"  Classes: {classes[:5]}... (showing first 5)")

    # Count total images
    total_images = 0
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        n_images = len([f for f in os.listdir(cls_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += n_images
    print(f"  Total images: {total_images}")
else:
    print(f"✗ Dataset not found at {dataset_path}")
    print("  Please add the PlantVillage dataset to your notebook")
```

---

## Cell 4: (Optional) Setup Wandb
```python
# Option 1: Login with API key
import wandb
wandb.login()

# Option 2: Use environment variable (if you set WANDB_API_KEY in Kaggle Secrets)
# import os
# os.environ['WANDB_API_KEY'] = 'your-api-key-here'
```

**Skip this cell if you don't want to use Wandb** (will still work, just no online logging)

---

## Cell 5: Test Configuration
```python
# Test the setup before training
!python test_plantvillage.py --data_root /kaggle/input/plantdisease/PlantVillage
```

---

## Cell 6: Start Training
```python
# Start training with default config
!python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run_$(date +%Y%m%d_%H%M%S) \
    --ni \
    --exp /kaggle/working/exp
```

**This will run for a long time!** Kaggle has session limits, so consider:
- Reducing epochs in config for testing
- Using checkpointing to resume later

---

## Cell 7: Monitor Training (Run in parallel)
```python
# View loss in real-time
!tail -f /kaggle/working/exp/logs/*/stdout.txt
```

---

## Cell 8: View Tensorboard (Run after training starts)
```python
# Load tensorboard
%load_ext tensorboard

# Find the log directory
import glob
log_dirs = glob.glob('/kaggle/working/exp/tensorboard/*')
if log_dirs:
    log_dir = log_dirs[0]
    print(f"Tensorboard log directory: {log_dir}")
    %tensorboard --logdir {log_dir}
else:
    print("No tensorboard logs found yet. Wait for training to start.")
```

---

## Cell 9: Check Training Progress
```python
# Check latest checkpoint
import glob
import os

ckpt_files = glob.glob('/kaggle/working/exp/logs/*/ckpt_*.pth')
if ckpt_files:
    latest_ckpt = max(ckpt_files, key=os.path.getctime)
    print(f"Latest checkpoint: {latest_ckpt}")
    print(f"File size: {os.path.getsize(latest_ckpt) / 1024 / 1024:.2f} MB")
    print(f"Last modified: {os.path.getctime(latest_ckpt)}")
else:
    print("No checkpoints found yet.")
```

---

## Cell 10: Resume Training (If interrupted)
```python
# Resume from last checkpoint
!python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run \
    --resume_training \
    --ni \
    --exp /kaggle/working/exp
```

---

## Alternative: Use Training Script
```python
# Instead of Cell 6, you can use the pre-made script
!bash train_kaggle.sh
```

---

## Quick Test Training (Small Config)
If you want to test quickly, create a test config:

```python
# Create a smaller config for testing
test_config = """
data:
    dataset: "PlantVillage"
    data_root: "/kaggle/input/plantdisease/PlantVillage"
    image_size: 32  # Smaller for faster training
    channels: 3
    random_flip: true
    rescaled: true
    num_workers: 2

model:
    type: "simple"
    in_channels: 3
    out_ch: 3
    ch: 64  # Smaller model
    ch_mult: [1, 2, 2]
    num_res_blocks: 1
    attn_resolutions: [16, ]
    dropout: 0.1
    var_type: fixedlarge
    ema_rate: 0.9999
    ema: True
    resamp_with_conv: True
    conditional: True
    num_classes: 38

diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000

training:
    batch_size: 32  # Smaller batch
    n_epochs: 10  # Just 10 epochs for testing
    snapshot_freq: 1000

sampling:
    batch_size: 16

optim:
    weight_decay: 0.000
    optimizer: "Adam"
    lr: 0.0002
    beta1: 0.9
    amsgrad: false
    eps: 0.00000001
    grad_clip: 1.0

wandb:
    enabled: False  # Disable for quick test
"""

# Save test config
with open('configs/plantvillage_test.yml', 'w') as f:
    f.write(test_config)

# Run with test config
!python main.py \
    --config plantvillage_test.yml \
    --doc test_run \
    --ni \
    --exp /kaggle/working/exp
```

---

## Tips for Kaggle

### 1. GPU Memory Issues
If you get OOM errors:
```python
# Edit configs/plantvillage.yml and reduce:
# - batch_size: 32 or 16
# - image_size: 32
# - ch: 64
```

### 2. Session Timeout
Kaggle sessions timeout after 12 hours. To handle this:
- Save checkpoints frequently (already configured)
- Use resume_training flag
- Download important checkpoints to your local machine

### 3. Download Checkpoints
```python
# Download the latest checkpoint
from IPython.display import FileLink
import glob

ckpt_files = glob.glob('/kaggle/working/exp/logs/*/ckpt.pth')
if ckpt_files:
    FileLink(ckpt_files[0])
```

### 4. Monitor GPU Usage
```python
# Check GPU memory
!nvidia-smi
```

### 5. View Wandb Dashboard
If you enabled Wandb:
```python
import wandb
print(f"View your run at: {wandb.run.get_url()}")
```

---

## What to Expect

**Training time:**
- Full training: 20-40 hours depending on GPU
- Quick test (10 epochs): 1-2 hours

**Output:**
- Checkpoints in `/kaggle/working/exp/logs/plantvillage_run/`
- Logs in `/kaggle/working/exp/logs/plantvillage_run/stdout.txt`
- Tensorboard logs in `/kaggle/working/exp/tensorboard/`
- Wandb dashboard (if enabled)

**First epoch should:**
- Take 5-15 minutes
- Show decreasing loss
- Use ~90% GPU
- Save checkpoint at step 5000

---

## Troubleshooting

### "Dataset not found"
```python
!ls -la /kaggle/input/plantdisease/
# Make sure PlantVillage folder exists
```

### "CUDA out of memory"
```python
# Reduce batch size in configs/plantvillage.yml
# Or restart kernel and run again
```

### "Module not found"
```python
# Reinstall dependencies
!pip install --force-reinstall -r requirements.txt
```

### "Config file not found"
```python
# Make sure you're in the ddim directory
%cd /kaggle/working/ddim
!pwd
!ls configs/
```

---

## Next Steps After Training

Once training completes:

1. **Generate samples:**
```python
!python main.py --config plantvillage.yml --doc plantvillage_run --sample --fid --ni
```

2. **Download model:**
```python
from IPython.display import FileLink
FileLink('/kaggle/working/exp/logs/plantvillage_run/ckpt.pth')
```

3. **Analyze results:**
- Check Wandb dashboard for loss curves
- View Tensorboard for detailed metrics
- Examine generated samples

---

**Ready to go!** 🚀

Start with Cell 1 and work your way down. Good luck with your training!
