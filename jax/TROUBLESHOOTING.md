# Troubleshooting Guide for Kaggle TPU

## Common Errors and Solutions

### 1. ImportError: cannot import name 'maps' from 'jax.experimental'

**Error:**
```
ImportError: cannot import name 'maps' from 'jax.experimental'
```

**Cause:** Version incompatibility between JAX and Flax.

**Solution:**

**Option A: Use Compatibility Fix (Recommended)**
```python
# Run BEFORE importing JAX/Flax
from fix_kaggle_imports import setup_kaggle_environment
setup_kaggle_environment()

# Now safe to import
import jax
import flax
```

**Option B: Use Compatible Versions**
```bash
# Check current versions
pip list | grep -E '(jax|flax|optax)'

# If needed, install compatible versions
pip install --upgrade 'flax>=0.7.5,<0.8.0' 'optax>=0.1.7'
```

**Option C: Don't Upgrade (Use Kaggle's Versions)**
```python
# Kaggle has pre-installed JAX/Flax that work together
# Just install other dependencies:
!pip install diffusers transformers tensorflow-hub scipy einops wandb
```

---

### 2. RuntimeWarning: os.fork() was called

**Warning:**
```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code
```

**Cause:** Jupyter/IPython uses fork() which conflicts with JAX.

**Solution:** This is just a warning, can be ignored. To suppress:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*os.fork.*')
```

Already handled in `train_tpu.py`.

---

### 3. UserWarning: Transparent hugepages are not enabled

**Warning:**
```
UserWarning: Transparent hugepages are not enabled
```

**Cause:** TPU optimization feature not enabled.

**Solution:** This is informational, can be ignored. To suppress:
```python
warnings.filterwarnings('ignore', category=UserWarning, message='.*hugepages.*')
```

Already handled in `train_tpu.py`.

---

### 4. "Device or resource busy" / TPU Not Found

**Error:**
```
RuntimeError: Device or resource busy
# or
RuntimeError: TPU initialization failed
```

**Solutions:**

1. **Check TPU is enabled:**
   - Settings → Accelerator → Select TPU v5e-8

2. **Restart Kernel:**
   - Kernel → Restart Kernel
   - Run setup cells again

3. **Clear JAX cache:**
   ```python
   !rm -rf /tmp/jax_cache
   ```

4. **Use retry logic (built-in):**
   ```python
   # train_tpu.py already has retry logic
   # Just run again after restart
   ```

---

### 5. Out of Memory (OOM)

**Error:**
```
RuntimeError: Resource exhausted: Out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   # In config
   training:
     batch_size: 32  # Instead of 64
   ```

2. **Clear memory:**
   ```python
   import jax
   jax.clear_caches()
   ```

3. **Reduce FID samples:**
   ```yaml
   fid:
     num_samples: 200  # Instead of 500
   ```

---

### 6. Dataset Not Found

**Error:**
```
FileNotFoundError: /kaggle/input/plantdisease/PlantVillage
```

**Solutions:**

1. **Add dataset to notebook:**
   - Add Data → Search "plantdisease" → Add

2. **Check dataset path:**
   ```python
   !ls /kaggle/input/
   !ls /kaggle/input/plantdisease/
   ```

3. **Update config:**
   ```yaml
   data:
     data_root: "/kaggle/input/plantdisease/PlantVillage"
   ```

---

### 7. Flax/JAX Version Conflicts

**Error:**
```
AttributeError: module 'flax' has no attribute 'XXX'
```

**Solution:**

**Check versions:**
```python
import jax, flax, optax
print(f"JAX: {jax.__version__}")
print(f"Flax: {flax.__version__}")
print(f"Optax: {optax.__version__}")
```

**Compatible combinations:**
- JAX 0.4.20-0.4.28 + Flax 0.7.5-0.8.4 + Optax 0.1.7-0.1.9
- JAX 0.4.23 + Flax 0.8.0 + Optax 0.1.9 (recommended)

**Reinstall compatible versions:**
```bash
pip install --force-reinstall jax[tpu]==0.4.23 flax==0.7.5 optax==0.1.7
```

---

### 8. VAE Loading Fails

**Error:**
```
OSError: Can't load tokenizer for 'stabilityai/sd-vae-ft-mse'
```

**Solutions:**

1. **Use Flax-native model:**
   ```python
   vae = create_vae("pcuenq/sd-vae-ft-mse-flax")
   ```

2. **Check internet connection:**
   ```python
   !curl -I https://huggingface.co
   ```

3. **Pre-download model:**
   ```python
   from diffusers import FlaxAutoencoderKL
   FlaxAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", from_pt=True)
   ```

---

### 9. Wandb Login Fails

**Error:**
```
wandb.errors.UsageError: api_key not configured
```

**Solutions:**

1. **Login with key:**
   ```python
   import wandb
   wandb.login(key="YOUR_API_KEY")
   ```

2. **Use environment variable:**
   ```python
   import os
   os.environ['WANDB_API_KEY'] = 'YOUR_KEY'
   ```

3. **Disable wandb:**
   ```yaml
   # In config
   wandb:
     enabled: false
   ```

---

### 10. Training Doesn't Resume

**Issue:** Training starts from step 0 instead of resuming.

**Solutions:**

1. **Check checkpoint exists:**
   ```python
   !ls -lh /kaggle/working/logs/*/checkpoints/
   ```

2. **Verify checkpoint manager:**
   ```python
   # CheckpointManager looks for .pkl files
   # Make sure they weren't deleted
   ```

3. **Force resume:**
   ```python
   # Just run the same training command again
   # It auto-detects checkpoints
   ```

---

## Version Compatibility Matrix

| Kaggle Environment | JAX | Flax | Optax | Status |
|-------------------|-----|------|-------|--------|
| Default (2024) | 0.4.23 | 0.7.5 | 0.1.7 | ✅ Recommended |
| Updated | 0.4.26 | 0.8.0 | 0.1.9 | ⚠️ May need fix |
| Latest | 0.4.28+ | 0.8.2+ | 0.1.9 | ❌ Needs compatibility fix |

---

## Quick Debug Commands

```python
# Check environment
!nvidia-smi  # Won't work on TPU
!python -c "import jax; print(jax.devices())"

# Check versions
!pip list | grep -E '(jax|flax|optax|diffusers)'

# Check dataset
!ls -lh /kaggle/input/plantdisease/PlantVillage/ | head

# Check disk space
!df -h /kaggle/working

# Check memory
!free -h  # May not work on TPU

# View logs
!tail -n 100 /kaggle/working/logs/*/stdout.txt

# Clear cache
!rm -rf /tmp/jax_cache
!rm -rf ~/.cache/huggingface
```

---

## Still Having Issues?

1. **Check Kaggle Forums:**
   - https://www.kaggle.com/discussions

2. **Open GitHub Issue:**
   - Include full error traceback
   - Include version info
   - Include steps to reproduce

3. **Try Clean Restart:**
   ```python
   # Kernel → Restart & Clear Output
   # Run all cells again
   ```

4. **Use CPU for Testing:**
   ```python
   # Change in Cell 1
   os.environ['JAX_PLATFORMS'] = 'cpu'
   ```
