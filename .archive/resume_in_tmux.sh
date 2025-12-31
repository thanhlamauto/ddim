#!/bin/bash

# Resume training in tmux session
# This allows training to continue even if you disconnect

SESSION_NAME="plantvillage_training"
LOG_DIR="/workspace/ddim/exp/logs/plantvillage_100steps"
CONFIG_PATH="${LOG_DIR}/config.yml"

echo "=== Resume Training in Tmux ==="
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed. Installing..."
    apt-get update && apt-get install -y tmux
fi

# Check if session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "⚠ Tmux session '${SESSION_NAME}' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill existing session: tmux kill-session -t ${SESSION_NAME}"
    echo ""
    read -p "Kill existing session and start new? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        tmux kill-session -t ${SESSION_NAME}
        echo "✓ Killed existing session"
    else
        echo "Attaching to existing session..."
        tmux attach -t ${SESSION_NAME}
        exit 0
    fi
fi

# Update config to 160k steps
echo "Updating config to 160k steps..."
python3 << 'EOF'
import yaml
config_path = "/workspace/ddim/exp/logs/plantvillage_100steps/config.yml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['training'].__dict__['n_iters'] = 160000
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print("✓ Config updated: n_iters -> 160000")
EOF

echo ""
echo "Creating tmux session: ${SESSION_NAME}"
echo ""

# Create tmux session and run training
tmux new-session -d -s ${SESSION_NAME} -n training

# Send commands to tmux session
tmux send-keys -t ${SESSION_NAME}:training "cd /workspace/ddim" C-m
tmux send-keys -t ${SESSION_NAME}:training "echo '=== Starting Training Resume ==='" C-m
tmux send-keys -t ${SESSION_NAME}:training "echo 'Session: ${SESSION_NAME}'" C-m
tmux send-keys -t ${SESSION_NAME}:training "echo 'Target: 100k -> 160k steps'" C-m
tmux send-keys -t ${SESSION_NAME}:training "echo '==========================='" C-m
tmux send-keys -t ${SESSION_NAME}:training "echo ''" C-m

# Run the training command
tmux send-keys -t ${SESSION_NAME}:training "python main.py --config plantvillage_latent.yml --doc plantvillage_100steps --exp exp --latent_cond --resume_training --ni" C-m

echo "✓ Training started in tmux session!"
echo ""
echo "Useful commands:"
echo "  - Attach to session:    tmux attach -t ${SESSION_NAME}"
echo "  - Detach from session:  Ctrl+B then D"
echo "  - Kill session:         tmux kill-session -t ${SESSION_NAME}"
echo "  - List sessions:        tmux ls"
echo ""
echo "Attaching to session in 3 seconds..."
sleep 3

# Attach to the session
tmux attach -t ${SESSION_NAME}
