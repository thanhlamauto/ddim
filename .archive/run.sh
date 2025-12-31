#!/bin/bash
# Quick start: Resume training in tmux
# Usage: ./run.sh

SESSION="plantvillage_training"

echo "=== Resume Training in Tmux ==="

# Update config using unsafe_load (required for argparse.Namespace)
python3 << 'PYEOF'
import yaml

config_path = '/workspace/ddim/exp/logs/plantvillage_100steps/config.yml'

# Load with unsafe_load to handle argparse.Namespace
with open(config_path, 'r') as f:
    config = yaml.unsafe_load(f)

old_iters = config.training.n_iters
config.training.n_iters = 160000

# Save back
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f'✓ Config updated: {old_iters} -> 160000 steps')
PYEOF

if [ $? -ne 0 ]; then
    echo "✗ Failed to update config"
    exit 1
fi

# Check if session exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo ""
    echo "⚠ Session '$SESSION' already exists!"
    echo "Attaching to existing session..."
    tmux attach -t $SESSION
    exit 0
fi

echo ""
echo "Creating tmux session: $SESSION"
echo ""

# Create new session
tmux new-session -d -s $SESSION -n training

# Send commands
tmux send-keys -t $SESSION:training "cd /workspace/ddim" C-m
tmux send-keys -t $SESSION:training "echo '=== Resume Training ==='" C-m
tmux send-keys -t $SESSION:training "echo 'Target: 100k -> 160k steps'" C-m
tmux send-keys -t $SESSION:training "echo '======================'" C-m
tmux send-keys -t $SESSION:training "echo ''" C-m
tmux send-keys -t $SESSION:training "python main.py --config plantvillage_latent.yml --doc plantvillage_100steps --exp exp --latent_cond --resume_training --ni" C-m

echo "✓ Training started in tmux!"
echo ""
echo "Commands:"
echo "  Detach:     Ctrl+B, D"
echo "  Attach:     tmux attach -t $SESSION"
echo "  Kill:       tmux kill-session -t $SESSION"
echo "  View log:   tail -f exp/logs/plantvillage_100steps/stdout.txt"
echo ""
echo "Attaching in 2 seconds..."
sleep 2

# Attach
tmux attach -t $SESSION
