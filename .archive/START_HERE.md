# DDIM PlantVillage Ablation Study - Ready to Run! 🚀

## ✅ Setup Complete

All code has been prepared for your ablation study. Here's what's ready:

### 1. DDIM Training Setup
- ✅ Config modified: 100 inference steps (was 50)
- ✅ Inception Score (IS) implemented and integrated
- ✅ FID + IS will be calculated every 20k training steps
- ✅ Dependencies installed: torch-fidelity, scikit-learn, seaborn

### 2. Synthetic Data Generation
- ✅ Modified to generate 1024 images per class
- ✅ Will produce 15,360 total images (15 classes × 1024)

### 3. Classifier Training Scripts
- ✅ ResNet-18 model wrapper
- ✅ Data loading with 4 scenarios:
  - Scenario 1: Train on synthetic only
  - Scenario 2: Train on real only (unbalanced)
  - Scenario 3: Train on balanced real (traditional augmentation)
  - Scenario 4: Train on hybrid (real + synthetic)
- ✅ Evaluation scripts with metrics and confusion matrices
- ✅ Comparison scripts for all scenarios

---

## 🚀 Quick Start

### Step 1: Start DDIM Training (8-12 hours)

```bash
cd /workspace/ddim
tmux new-session -s plantvillage_train
python3 main.py --config plantvillage_latent.yml --exp exp --doc plantvillage_100steps --latent_cond --ni 2>&1 | tee training.log
```

Press `Ctrl+B` then `D` to detach from tmux.

### Step 2: Monitor Training

```bash
# Reattach to see progress
tmux attach -t plantvillage_train

# Or check log
tail -f /workspace/ddim/training.log

# Check GPU usage
nvidia-smi
```

### Step 3: Generate Synthetic Data (after training)

```bash
cd /workspace/ddim
python3 generate_synthetic.py \
    --ckpt exp/logs/plantvillage_100steps/ckpt_best.pth \
    --output /workspace/PlantVillage_Synthetic \
    --batch_size 32 \
    --num_steps 100 \
    --cfg_scale 3.0
```

### Step 4: Run Ablation Studies

See `ABLATION_STUDY_COMMANDS.md` for detailed commands for all 4 scenarios.

**Quick version - Run all 4 in parallel:**

```bash
cd /workspace/ddim

# Scenario 1: Synth only
tmux new-session -d -s s1 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage_Synthetic --scenario synth_only --output_dir results/scenario1_synth_only --batch_size 128 --epochs 100 --lr 0.1'

# Scenario 2: Real only
tmux new-session -d -s s2 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --scenario real_only --output_dir results/scenario2_real_only --batch_size 128 --epochs 100 --lr 0.1'

# Scenario 3: Real balanced
tmux new-session -d -s s3 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --scenario real_balanced --output_dir results/scenario3_real_balanced --batch_size 128 --epochs 100 --lr 0.1'

# Scenario 4: Hybrid
tmux new-session -d -s s4 'python3 classifier/train_classifier.py --data_root /workspace/PlantVillage --synth_root /workspace/PlantVillage_Synthetic --scenario hybrid --output_dir results/scenario4_hybrid --batch_size 128 --epochs 100 --lr 0.1'
```

### Step 5: Evaluate All Scenarios

```bash
# After each scenario finishes training, run evaluation
python3 classifier/evaluate_classifier.py --checkpoint results/scenario1_synth_only/best_model.pth --data_root /workspace/PlantVillage --output_dir results/scenario1_synth_only/eval_on_real

python3 classifier/evaluate_classifier.py --checkpoint results/scenario2_real_only/best_model.pth --data_root /workspace/PlantVillage --output_dir results/scenario2_real_only/eval

python3 classifier/evaluate_classifier.py --checkpoint results/scenario3_real_balanced/best_model.pth --data_root /workspace/PlantVillage --output_dir results/scenario3_real_balanced/eval

python3 classifier/evaluate_classifier.py --checkpoint results/scenario4_hybrid/best_model.pth --data_root /workspace/PlantVillage --output_dir results/scenario4_hybrid/eval
```

### Step 6: Compare Results

```bash
python3 classifier/compare_results.py
```

Results will be in: `/workspace/ddim/results/comparison/`

---

## 📊 What You'll Get

### DDIM Training Outputs
- Best checkpoint with FID and IS metrics
- Training logs and tensorboard data
- Sample images generated during training
- WandB tracking (if enabled)

### Ablation Study Results

For each scenario:
- ✅ Trained ResNet-18 model
- ✅ Test accuracy on real data
- ✅ Per-class precision/recall/F1
- ✅ Confusion matrix visualization
- ✅ Training history (loss, accuracy curves)

### Final Comparison
- ✅ Comparison table (CSV)
- ✅ Accuracy bar chart
- ✅ Per-class accuracy heatmap

---

## 📂 Directory Structure

```
/workspace/ddim/
├── configs/
│   └── plantvillage_latent.yml (modified: 100 inference steps)
├── utils/
│   └── inception_score.py (NEW)
├── classifier/ (NEW)
│   ├── models.py
│   ├── data_utils.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   └── compare_results.py
├── results/ (will be created)
│   ├── scenario1_synth_only/
│   ├── scenario2_real_only/
│   ├── scenario3_real_balanced/
│   ├── scenario4_hybrid/
│   └── comparison/
├── exp/logs/plantvillage_100steps/ (will be created)
│   ├── ckpt_*.pth
│   └── ckpt_best.pth
└── ABLATION_STUDY_COMMANDS.md (detailed commands)

/workspace/PlantVillage/ (original real data)
/workspace/PlantVillage_Synthetic/ (will be created after generation)
```

---

## ⏱️ Timeline

| Phase | Duration |
|-------|----------|
| DDIM Training | 8-12 hours |
| Synthetic Generation | 1-2 hours |
| 4 Classifier Trainings | 8-12 hours total |
| Evaluation & Comparison | 30 minutes |
| **TOTAL** | **~20-25 hours** |

---

## 💡 Tips

1. **Run DDIM training overnight** - it takes the longest
2. **Monitor GPU memory** with `nvidia-smi` if running multiple scenarios
3. **Check tmux sessions** with `tmux ls` to see what's running
4. **Review logs** if something fails - they're saved in each directory
5. **Be patient** - quality results take time!

---

## 📖 Documentation

- **ABLATION_STUDY_COMMANDS.md** - Complete command reference
- **Plan file**: `/root/.claude/plans/jolly-puzzling-pine.md` - Original implementation plan

---

## 🎯 Next Steps

1. Start DDIM training now (see Quick Start above)
2. While training, you can test the classifier scripts:
   ```bash
   python3 classifier/models.py
   python3 classifier/data_utils.py real_only
   ```
3. Wait for training to complete (~8-12 hours)
4. Generate synthetic data
5. Run all 4 ablation scenarios
6. Compare and analyze results

---

## ✨ Everything is ready! Start with Step 1 above.

Good luck with your research! 🚀🌱
