#!/usr/bin/env python3
"""
Resume training in tmux session - allows training to run in background
"""

import os
import sys
import subprocess
import yaml
import time


def update_config(config_path, target_steps):
    """Update n_iters in config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    old_steps = config['training'].__dict__.get('n_iters', 'unknown')
    config['training'].__dict__['n_iters'] = target_steps

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Config updated: {old_steps} -> {target_steps} steps")


def check_tmux_installed():
    """Check if tmux is installed"""
    try:
        subprocess.run(['tmux', '-V'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ tmux not installed. Installing...")
        try:
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'tmux'], check=True)
            print("✓ tmux installed")
            return True
        except:
            print("✗ Failed to install tmux")
            return False


def session_exists(session_name):
    """Check if tmux session exists"""
    try:
        subprocess.run(['tmux', 'has-session', '-t', session_name],
                       capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def create_training_session(session_name, target_steps=160000):
    """Create tmux session and start training"""

    print("=" * 60)
    print("Resume Training in Tmux")
    print("=" * 60)
    print()

    # Check tmux
    if not check_tmux_installed():
        return 1

    # Check if session exists
    if session_exists(session_name):
        print(f"⚠ Session '{session_name}' already exists!")
        print()
        response = input("Kill and recreate? (y/N): ")
        if response.lower() == 'y':
            subprocess.run(['tmux', 'kill-session', '-t', session_name])
            print(f"✓ Killed session '{session_name}'")
        else:
            print(f"\nAttaching to existing session '{session_name}'...")
            subprocess.run(['tmux', 'attach', '-t', session_name])
            return 0

    # Update config
    config_path = '/workspace/ddim/exp/logs/plantvillage_100steps/config.yml'
    if os.path.exists(config_path):
        update_config(config_path, target_steps)
    else:
        print(f"⚠ Warning: Config not found at {config_path}")

    print()
    print(f"Creating tmux session: {session_name}")
    print(f"Target: {target_steps} steps")
    print()

    # Create session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, '-n', 'training'])

    # Setup commands
    commands = [
        'cd /workspace/ddim',
        'echo "=== Resume Training ==="',
        f'echo "Session: {session_name}"',
        f'echo "Target: 100k -> {target_steps} steps"',
        'echo "========================"',
        'echo ""',
        'python main.py --config plantvillage_latent.yml --doc plantvillage_100steps --exp exp --latent_cond --resume_training --ni'
    ]

    # Send commands to session
    for cmd in commands:
        subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:training', cmd, 'C-m'])
        time.sleep(0.1)

    print("✓ Training started in tmux!")
    print()
    print("Tmux Commands:")
    print(f"  • Attach:  tmux attach -t {session_name}")
    print(f"  • Detach:  Ctrl+B then D")
    print(f"  • Kill:    tmux kill-session -t {session_name}")
    print(f"  • List:    tmux ls")
    print()
    print("Attaching in 2 seconds...")
    time.sleep(2)

    # Attach to session
    subprocess.run(['tmux', 'attach', '-t', session_name])
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resume training in tmux')
    parser.add_argument('--session', type=str, default='plantvillage_training',
                        help='Tmux session name')
    parser.add_argument('--target_steps', type=int, default=160000,
                        help='Target training steps')
    parser.add_argument('--no_attach', action='store_true',
                        help='Create session but do not attach')

    args = parser.parse_args()

    sys.exit(create_training_session(args.session, args.target_steps))
