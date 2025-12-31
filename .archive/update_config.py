#!/usr/bin/env python3
"""Update config n_iters to 160000"""
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

print(f'✓ Updated: n_iters {old_iters} -> 160000')
print(f'✓ Config: {config_path}')
