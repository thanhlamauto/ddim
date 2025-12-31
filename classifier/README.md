# PlantVillage Classifier - Ablation Study

Train ResNet-18 classifier on 4 different scenarios to compare synthetic vs real data.

## 📊 Scenarios

### Scenario 1: Synthetic Only
- **Train/Val:** Synthetic data (70/15 split)
- **Test:** Real data test set (15% - same as all scenarios)
- **Purpose:** Evaluate quality of synthetic data

### Scenario 2: Real Only (Unbalanced)
- **Train/Val/Test:** Real data (70/15/15 split)
- **Purpose:** Baseline performance

### Scenario 3: Real Balanced
- **Train/Val:** Real data balanced to 1024 images/class
  - Majority classes: downsample
  - Minority classes: oversample with augmentation
- **Test:** Real data test set (15% - same as all scenarios)
- **Purpose:** Improve minority class performance

### Scenario 4: Hybrid (Real + Synthetic)
- **Train/Val:** Real + Synthetic balanced to 1024 images/class
  - Majority classes (≥1024 real): downsample real only
  - Minority classes (<1024 real): all real + synthetic to fill
- **Test:** Real data test set (15% - same as all scenarios)
- **Purpose:** Augment minority classes with synthetic data

---

## ⚙️ Settings

```
Epochs: 50
Batch size: 128
Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
LR scheduler: Cosine annealing
Split: 70% train, 15% val, 15% test
Seed: 42 (for reproducibility)
```

**Important:** All scenarios use the **same real test set** (15% of real data) for fair comparison.

---

## 🚀 Usage

### Train All Scenarios (Sequential)
```bash
cd /workspace/ddim/classifier
./train_all_scenarios.sh
```

### Train Individual Scenarios

**Scenario 1: Synthetic Only**
```bash
cd /workspace/ddim/classifier
./train_scenario1.sh
```

**Scenario 2: Real Only**
```bash
python train_classifier.py \
    --scenario real_only \
    --data_root /workspace/PlantVillage \
    --output_dir ../results/scenario2_real_only \
    --epochs 50

python evaluate_classifier.py \
    --checkpoint ../results/scenario2_real_only/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir ../results/scenario2_real_only/eval
```

**Scenario 3: Real Balanced**
```bash
python train_classifier.py \
    --scenario real_balanced \
    --data_root /workspace/PlantVillage \
    --output_dir ../results/scenario3_real_balanced \
    --target_count 1024 \
    --epochs 50

python evaluate_classifier.py \
    --checkpoint ../results/scenario3_real_balanced/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir ../results/scenario3_real_balanced/eval
```

**Scenario 4: Hybrid**
```bash
python train_classifier.py \
    --scenario hybrid \
    --data_root /workspace/PlantVillage \
    --synth_root /workspace/PlantVillage_Synthetic \
    --output_dir ../results/scenario4_hybrid \
    --target_count 1024 \
    --epochs 50

python evaluate_classifier.py \
    --checkpoint ../results/scenario4_hybrid/best_model.pth \
    --data_root /workspace/PlantVillage \
    --output_dir ../results/scenario4_hybrid/eval
```

---

## 📁 Output Structure

```
results/
├── scenario1_synth_only/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── history.json
│   └── eval/
│       ├── test_results.json
│       ├── confusion_matrix.png
│       ├── predictions.npy
│       └── targets.npy
├── scenario2_real_only/
│   └── eval/
├── scenario3_real_balanced/
│   └── eval/
├── scenario4_hybrid/
│   └── eval/
└── comparison/
    ├── comparison_table.csv
    ├── accuracy_comparison.png
    └── perclass_heatmap.png
```

---

## 📊 Compare Results

After training all scenarios:

```bash
python compare_results.py \
    --scenario1_dir ../results/scenario1_synth_only/eval \
    --scenario2_dir ../results/scenario2_real_only/eval \
    --scenario3_dir ../results/scenario3_real_balanced/eval \
    --scenario4_dir ../results/scenario4_hybrid/eval \
    --output_dir ../results/comparison
```

This generates:
- Comparison table (CSV)
- Accuracy comparison plot
- Per-class recall heatmap

---

## 🔍 Key Features

✅ **Fair Comparison:** All scenarios test on same real test set
✅ **Consistent Split:** 70/15/15 across all scenarios
✅ **Reproducible:** Fixed seed (42)
✅ **Confusion Matrix:** Saved for each scenario
✅ **Detailed Metrics:** Precision, recall, F1-score per class
✅ **Visualization:** Comparison plots and heatmaps

---

## 📝 Notes

- **Scenario 1** trains on synthetic but tests on real - this evaluates synthetic data quality
- All other scenarios train and test on real data
- Test set is **never** used for training in any scenario
- Same random seed ensures same train/val/test split across all scenarios
