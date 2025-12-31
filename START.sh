#!/bin/bash
# Start training in tmux - Auto-check config

SESSION="plantvillage_training"
cd /workspace/ddim

echo "=== Resume Training in Tmux ==="

# Verify config
ITERS=$(grep "n_iters:" configs/plantvillage_latent.yml | grep -o '[0-9]*')
if [ "$ITERS" != "160000" ]; then
    echo "Updating config: $ITERS -> 160000 steps..."
    sed -i 's/n_iters: [0-9]*/n_iters: 160000/' configs/plantvillage_latent.yml
    echo "✓ Config updated"
fi
echo "✓ Config: n_iters = 160000"
echo ""

# Kill old session if exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Killing old session: $SESSION"
    tmux kill-session -t $SESSION
fi

# Create session
echo "Creating tmux session: $SESSION"
echo ""
tmux new-session -d -s $SESSION -n training

# Run training
tmux send-keys -t $SESSION:training "cd /workspace/ddim" C-m
tmux send-keys -t $SESSION:training "echo '=== Resume Training ==='" C-m
tmux send-keys -t $SESSION:training "echo 'Target: 100k -> 160k steps'" C-m
tmux send-keys -t $SESSION:training "echo '======================'" C-m
tmux send-keys -t $SESSION:training "python main.py --config plantvillage_latent.yml --doc plantvillage_100steps --exp exp --latent_cond --resume_training --ni" C-m

echo "✓ Training started!"
echo ""
echo "Commands:"
echo "  • Attach:  tmux attach -t $SESSION"
echo "  • Detach:  Ctrl+B, D"
echo "  • Logs:    tail -f exp/logs/plantvillage_100steps/stdout.txt"
echo "  • Kill:    tmux kill-session -t $SESSION"
echo ""

sleep 2
tmux attach -t $SESSION
