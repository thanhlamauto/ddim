#!/bin/bash
# Train classifier for all 4 scenarios with consistent settings
# All scenarios use 70/15/15 split and test on the same real test set

REAL_DATA="/workspace/PlantVillage"
SYNTH_DATA="/workspace/PlantVillage_Synthetic"
RESULTS_DIR="/workspace/ddim/results"
EPOCHS=50
BATCH_SIZE=128
SEED=42

# Split ratios (70/15/15)
TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15

echo "=== Train Classifier - All 4 Scenarios ==="
echo "Settings:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Split: ${TRAIN_RATIO}/${VAL_RATIO}/${TEST_RATIO} (train/val/test)"
echo "  Seed: $SEED"
echo ""

# ------------------------------------------------------------------------------
# Scenario 1: Synthetic Only
# Train/val on synthetic, test on real
# ------------------------------------------------------------------------------
echo "=== Scenario 1: Synthetic Only ==="
echo "Train/Val: Synthetic data"
echo "Test: Real data (same as other scenarios)"
echo ""

python train_classifier.py \
    --scenario synth_only \
    --data_root "$REAL_DATA" \
    --synth_root "$SYNTH_DATA" \
    --output_dir "$RESULTS_DIR/scenario1_synth_only" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

# Evaluate on real test set
echo ""
echo "Evaluating Scenario 1 on real test set..."
python evaluate_classifier.py \
    --checkpoint "$RESULTS_DIR/scenario1_synth_only/best_model.pth" \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario1_synth_only/eval" \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

echo ""
echo "✓ Scenario 1 complete!"
echo ""

# ------------------------------------------------------------------------------
# Scenario 2: Real Only (Unbalanced)
# Train/val/test all on real data
# ------------------------------------------------------------------------------
echo "=== Scenario 2: Real Only (Unbalanced) ==="
echo "Train/Val/Test: Real data (unbalanced)"
echo ""

python train_classifier.py \
    --scenario real_only \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario2_real_only" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

# Evaluate
echo ""
echo "Evaluating Scenario 2..."
python evaluate_classifier.py \
    --checkpoint "$RESULTS_DIR/scenario2_real_only/best_model.pth" \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario2_real_only/eval" \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

echo ""
echo "✓ Scenario 2 complete!"
echo ""

# ------------------------------------------------------------------------------
# Scenario 3: Real Balanced
# Train/val on balanced real (downsample majority, oversample minority)
# Test on same real test set
# ------------------------------------------------------------------------------
echo "=== Scenario 3: Real Balanced ==="
echo "Train/Val: Real data (balanced to 1024/class)"
echo "Test: Real data (same test set)"
echo ""

python train_classifier.py \
    --scenario real_balanced \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario3_real_balanced" \
    --target_count 1024 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

# Evaluate
echo ""
echo "Evaluating Scenario 3..."
python evaluate_classifier.py \
    --checkpoint "$RESULTS_DIR/scenario3_real_balanced/best_model.pth" \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario3_real_balanced/eval" \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

echo ""
echo "✓ Scenario 3 complete!"
echo ""

# ------------------------------------------------------------------------------
# Scenario 4: Hybrid (Real + Synthetic)
# Minority classes: real + synthetic to reach 1024
# Majority classes: downsample real to 1024
# Test on same real test set
# ------------------------------------------------------------------------------
echo "=== Scenario 4: Hybrid (Real + Synthetic) ==="
echo "Train/Val: Real + Synthetic (balanced to 1024/class)"
echo "  - Majority classes: downsample real"
echo "  - Minority classes: real + synthetic"
echo "Test: Real data (same test set)"
echo ""

python train_classifier.py \
    --scenario hybrid \
    --data_root "$REAL_DATA" \
    --synth_root "$SYNTH_DATA" \
    --output_dir "$RESULTS_DIR/scenario4_hybrid" \
    --target_count 1024 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

# Evaluate
echo ""
echo "Evaluating Scenario 4..."
python evaluate_classifier.py \
    --checkpoint "$RESULTS_DIR/scenario4_hybrid/best_model.pth" \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario4_hybrid/eval" \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

echo ""
echo "✓ Scenario 4 complete!"
echo ""

# ------------------------------------------------------------------------------
# Compare all results
# ------------------------------------------------------------------------------
echo "=== Comparing All Scenarios ==="
python compare_results.py \
    --scenario1_dir "$RESULTS_DIR/scenario1_synth_only/eval" \
    --scenario2_dir "$RESULTS_DIR/scenario2_real_only/eval" \
    --scenario3_dir "$RESULTS_DIR/scenario3_real_balanced/eval" \
    --scenario4_dir "$RESULTS_DIR/scenario4_hybrid/eval" \
    --output_dir "$RESULTS_DIR/comparison"

echo ""
echo "=== All Scenarios Complete! ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary:"
echo "  - Scenario 1 (Synth Only): $RESULTS_DIR/scenario1_synth_only/eval/"
echo "  - Scenario 2 (Real Only): $RESULTS_DIR/scenario2_real_only/eval/"
echo "  - Scenario 3 (Real Balanced): $RESULTS_DIR/scenario3_real_balanced/eval/"
echo "  - Scenario 4 (Hybrid): $RESULTS_DIR/scenario4_hybrid/eval/"
echo "  - Comparison: $RESULTS_DIR/comparison/"
