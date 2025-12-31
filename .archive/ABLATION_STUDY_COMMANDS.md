# DDIM PlantVillage Ablation Study - Complete Commands

This document contains all commands needed to run the complete ablation study comparing synthetic data generation with traditional classification approaches.

## Setup Complete ✓

- [x] Config modified to 100 inference steps
- [x] Inception Score (IS) implementation added
- [x] IS integrated into validation
- [x] Dependencies installed (torch-fidelity, scikit-learn, seaborn)
- [x] Synthetic generation script modified (1024 images/class)
- [x] Classifier training scripts created
- [x] Evaluation scripts created
- [x] Comparison scripts created

---

## Phase 1: DDIM Training (8-12 hours)

### Start Training

```bash
cd /workspace/ddim

# Start in tmux so it runs in background
tmux new-session -s plantvillage_train

# Run training (will calculate FID and IS every 20k steps)
python3 main.py \
    --config plantvillage_latent.yml \
    --exp exp \
    --doc plantvillage_100steps \
    --latent_cond \
    --ni 2>&1 | tee training.log

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t plantvillage_train
```

### Monitor Training

```bash
# Check training log
tail -f /workspace/ddim/training.log

# Check wandb dashboard (if enabled)
# Project: ddim-plantvillage-full
# Run name: class-conditional-latent-256
```

### Training Outputs

- Checkpoints: `/workspace/ddim/exp/logs/plantvillage_100steps/ckpt_*.pth`
- Best model: `/workspace/ddim/exp/logs/plantvillage_100steps/ckpt_best.pth`
- Logs: `/workspace/ddim/training.log`
- TensorBoard: `/workspace/ddim/exp/tensorboard/plantvillage_100steps/`

---

## Phase 2: Generate Synthetic Dataset (~1-2 hours)

**Run this AFTER DDIM training completes**

```bash
cd /workspace/ddim

python3 generate_synthetic.py \
    --ckpt exp/logs/plantvillage_100steps/ckpt_best.pth \
    --output /workspace/PlantVillage_Synthetic \
    --batch_size 32 \
    --num_steps 100 \
    --cfg_scale 3.0
```

This will generate:
- **15,360 synthetic images** (1024 per class × 15 classes)
- Output directory: `/workspace/PlantVillage_Synthetic/`

---

## Phase 3: Ablation Studies

### Scenario 1: Train on Synthetic Only, Test on Real

**Training:**
```bash
cd /workspace/ddim

python3 classifier/train_classifier.py \
    --data_root /workspace/PlantVillage_Synthetic \
    --scenario synth_only \
    --output_dir results/scenario1_synth_only \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1
```

**Evaluation (on real test set):**
```bash
python3 classifier/evaluate_classifier.py \
    --checkpoint results/scenario1_synth_only/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir results/scenario1_synth_only/eval_on_real
```

---

### Scenario 2: Train on Real Only, Test on Real

**Training:**
```bash
cd /workspace/ddim

python3 classifier/train_classifier.py \
    --data_root /workspace/PlantVillage \
    --scenario real_only \
    --output_dir results/scenario2_real_only \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1
```

**Evaluation:**
```bash
python3 classifier/evaluate_classifier.py \
    --checkpoint results/scenario2_real_only/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir results/scenario2_real_only/eval
```

---

### Scenario 3: Train on Balanced Real (Traditional Augmentation), Test on Real

**Description:**
- All classes balanced to 1024 images
- Classes with >1024: downsample
- Classes with <1024: augment (flips, color jitter, random crops)

**Training:**
```bash
cd /workspace/ddim

python3 classifier/train_classifier.py \
    --data_root /workspace/PlantVillage \
    --scenario real_balanced \
    --output_dir results/scenario3_real_balanced \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1 \
    --target_count 1024
```

**Evaluation:**
```bash
python3 classifier/evaluate_classifier.py \
    --checkpoint results/scenario3_real_balanced/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir results/scenario3_real_balanced/eval
```

---

### Scenario 4: Train on Hybrid (Real + Synthetic), Test on Real

**Description:**
- All classes balanced to 1024 images
- Classes with ≥1024 real: use only real (downsample to 1024)
- Classes with <1024 real: use all real + synthetic to reach 1024

**Training:**
```bash
cd /workspace/ddim

python3 classifier/train_classifier.py \
    --data_root /workspace/PlantVillage \
    --synth_root /workspace/PlantVillage_Synthetic \
    --scenario hybrid \
    --output_dir results/scenario4_hybrid \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1 \
    --target_count 1024
```

**Evaluation:**
```bash
python3 classifier/evaluate_classifier.py \
    --checkpoint results/scenario4_hybrid/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir results/scenario4_hybrid/eval
```

---

## Phase 4: Compare All Results

**Run this AFTER all 4 scenarios complete**

```bash
cd /workspace/ddim

python3 classifier/compare_results.py \
    --scenario1_dir results/scenario1_synth_only/eval_on_real \
    --scenario2_dir results/scenario2_real_only/eval \
    --scenario3_dir results/scenario3_real_balanced/eval \
    --scenario4_dir results/scenario4_hybrid/eval \
    --output_dir results/comparison
```

**Outputs:**
- Comparison table: `results/comparison/comparison_table.csv`
- Accuracy bar chart: `results/comparison/accuracy_comparison.png`
- Per-class heatmap: `results/comparison/perclass_heatmap.png`

---

## Quick Test Commands

### Test Dataset Loading
```bash
cd /workspace/ddim

# Test different scenarios
python3 classifier/data_utils.py real_only
python3 classifier/data_utils.py real_balanced
python3 classifier/data_utils.py synth_only
python3 classifier/data_utils.py hybrid
```

### Test Model Creation
```bash
python3 classifier/models.py
```

---

## Expected Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | DDIM Training | 8-12 hours |
| 2 | Synthetic Generation | 1-2 hours |
| 3.1 | Scenario 1 Training | 2-3 hours |
| 3.2 | Scenario 2 Training | 2-3 hours |
| 3.3 | Scenario 3 Training | 2-3 hours |
| 3.4 | Scenario 4 Training | 2-3 hours |
| 4 | Results Comparison | 5 minutes |
| **Total** | | **20-25 hours** |

---

## Running Multiple Scenarios in Parallel

You can run all 4 classifier training scenarios in parallel using separate tmux sessions:

```bash
cd /workspace/ddim

# Scenario 1
tmux new-session -d -s scenario1 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage_Synthetic --scenario synth_only --output_dir results/scenario1_synth_only --batch_size 128 --epochs 100 --lr 0.1 2>&1 | tee results/scenario1_synth_only/train.log'

# Scenario 2
tmux new-session -d -s scenario2 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --scenario real_only --output_dir results/scenario2_real_only --batch_size 128 --epochs 100 --lr 0.1 2>&1 | tee results/scenario2_real_only/train.log'

# Scenario 3
tmux new-session -d -s scenario3 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --scenario real_balanced --output_dir results/scenario3_real_balanced --batch_size 128 --epochs 100 --lr 0.1 2>&1 | tee results/scenario3_real_balanced/train.log'

# Scenario 4
tmux new-session -d -s scenario4 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --synth_root /workspace/PlantVillage_Synthetic --scenario hybrid --output_dir results/scenario4_hybrid --batch_size 128 --epochs 100 --lr 0.1 2>&1 | tee results/scenario4_hybrid/train.log'

# Monitor all sessions
tmux ls

# Attach to a specific session
tmux attach -t scenario1
```

**Note:** Running all 4 in parallel requires sufficient GPU memory. If you have memory issues, run them sequentially instead.

---

## Checking Progress

### Check if training is running
```bash
nvidia-smi  # Check GPU usage
tmux ls     # List all tmux sessions
```

### Monitor specific scenario
```bash
# Attach to tmux session
tmux attach -t scenario1

# Or check log file
tail -f /workspace/ddim/results/scenario1_synth_only/train.log
```

### Check results
```bash
# List all result directories
ls -lh /workspace/ddim/results/

# Check if best model exists
ls -lh /workspace/ddim/results/scenario*/best_model.pth
```

---

## Files Created

### Configuration
- `/workspace/ddim/configs/plantvillage_latent.yml` (modified)

### Utils
- `/workspace/ddim/utils/inception_score.py`

### Classifier Scripts
- `/workspace/ddim/classifier/models.py`
- `/workspace/ddim/classifier/data_utils.py`
- `/workspace/ddim/classifier/train_classifier.py`
- `/workspace/ddim/classifier/evaluate_classifier.py`
- `/workspace/ddim/classifier/compare_results.py`

### Modified Files
- `/workspace/ddim/runners/latent_diffusion_cond.py` (IS integration)
- `/workspace/ddim/generate_synthetic.py` (1024 images/class)

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 64` or `--batch_size 32`

### Slow Data Loading
- Reduce num_workers: `--num_workers 2`

### Training Interrupted
- Resume from checkpoint (modify train_classifier.py to support resume)

### Synthetic Generation Fails
- Check if DDIM training completed successfully
- Verify checkpoint exists: `ls exp/logs/plantvillage_100steps/ckpt_best.pth`

---

## Contact / Questions

If you encounter any issues, check:
1. GPU memory usage: `nvidia-smi`
2. Disk space: `df -h`
3. Log files: `tail -f training.log`

Good luck with your ablation study! 🚀
