#!/bin/bash
# Scenario 1: Synthetic Only
# Train/val on synthetic, test on real (same test set as other scenarios)

REAL_DATA="/workspace/PlantVillage"
SYNTH_DATA="/workspace/PlantVillage_Synthetic"
RESULTS_DIR="/workspace/ddim/results"

echo "=== Scenario 1: Synthetic Only ==="
echo "Train/Val: Synthetic data (70/15 split)"
echo "Test: Real data test set (15%)"
echo ""

# Train on synthetic data
python train_classifier.py \
    --scenario synth_only \
    --data_root "$REAL_DATA" \
    --synth_root "$SYNTH_DATA" \
    --output_dir "$RESULTS_DIR/scenario1_synth_only" \
    --epochs 50 \
    --batch_size 128 \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# Evaluate on real test set
echo ""
echo "Evaluating on real test set..."
python evaluate_classifier.py \
    --checkpoint "$RESULTS_DIR/scenario1_synth_only/best_model.pth" \
    --data_root "$REAL_DATA" \
    --output_dir "$RESULTS_DIR/scenario1_synth_only/eval" \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

echo ""
echo "✓ Scenario 1 complete!"
echo "Results: $RESULTS_DIR/scenario1_synth_only/eval/"
echo "  - test_results.json"
echo "  - confusion_matrix.png"
