#!/usr/bin/env python3
"""
Load CompVis Latent Diffusion pretrained weights for PlantDoc fine-tuning
Converts cin256-v2 (ImageNet 1000 classes) → PlantDoc (28 classes)
"""

import torch
import argparse
import yaml
import os
import sys
from collections import OrderedDict


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_ldm_checkpoint(ckpt_path):
    """Load CompVis LDM checkpoint"""
    print(f"Loading checkpoint from {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"✓ Found state_dict with {len(state_dict)} keys")
    else:
        state_dict = ckpt
        print(f"✓ Direct state dict with {len(state_dict)} keys")

    return state_dict


def extract_unet_weights(state_dict):
    """Extract UNet weights from LDM checkpoint

    CompVis LDM structure:
    - model.diffusion_model.* → UNet
    - first_stage_model.* → VAE (we use SD VAE instead)
    - cond_stage_model.* → Conditioning (we use class embedding)
    """
    unet_weights = OrderedDict()

    for key, value in state_dict.items():
        # Extract diffusion_model (UNet) weights
        if key.startswith('model.diffusion_model.'):
            # Remove 'model.diffusion_model.' prefix
            new_key = key.replace('model.diffusion_model.', '')
            unet_weights[new_key] = value

    print(f"✓ Extracted {len(unet_weights)} UNet layers")
    return unet_weights


def adapt_class_embedding(unet_weights, num_classes_pretrained=1000, num_classes_target=28):
    """Adapt class embedding layer from ImageNet (1000) to PlantDoc (28)

    Strategy: Random initialization for PlantDoc classes
    Alternative: Could average ImageNet embeddings, but random init works better
    """
    adapted = OrderedDict()
    skipped = []

    for key, value in unet_weights.items():
        # Check if this is class embedding layer
        if 'label_emb' in key or 'class_emb' in key or 'y_embedder' in key:
            if value.shape[0] == num_classes_pretrained:
                print(f"  ⚠ Skipping {key}: shape {value.shape} (class embedding mismatch)")
                skipped.append(key)
                continue

        adapted[key] = value

    print(f"✓ Adapted weights: {len(adapted)} layers")
    if skipped:
        print(f"  Skipped {len(skipped)} class embedding layers (will be randomly initialized)")

    return adapted


def load_into_model(model, pretrained_weights, strict=False):
    """Load pretrained weights into model"""
    model_dict = model.state_dict()

    # Track statistics
    loaded = []
    skipped_shape = []
    skipped_missing = []

    # Filter pretrained weights
    pretrained_dict = {}
    for k, v in pretrained_weights.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                loaded.append(k)
            else:
                skipped_shape.append(f"{k}: {v.shape} → {model_dict[k].shape}")
        else:
            skipped_missing.append(k)

    # Update model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=strict)

    # Print summary
    print("\n" + "=" * 80)
    print("LOADING SUMMARY")
    print("=" * 80)
    print(f"✓ Loaded:  {len(loaded)}/{len(model_dict)} layers ({len(loaded)/len(model_dict)*100:.1f}%)")

    if skipped_shape:
        print(f"\n⚠ Skipped {len(skipped_shape)} layers due to shape mismatch:")
        for s in skipped_shape[:5]:
            print(f"  - {s}")
        if len(skipped_shape) > 5:
            print(f"  ... and {len(skipped_shape)-5} more")

    if skipped_missing:
        print(f"\n⚠ Skipped {len(skipped_missing)} layers not in model:")
        for s in skipped_missing[:5]:
            print(f"  - {s}")
        if len(skipped_missing) > 5:
            print(f"  ... and {len(skipped_missing)-5} more")

    # Check what's NOT loaded (will be randomly initialized)
    not_loaded = set(model_dict.keys()) - set(loaded)
    if not_loaded:
        print(f"\n📝 {len(not_loaded)} layers will be randomly initialized:")
        for layer in list(not_loaded)[:5]:
            print(f"  - {layer}")
        if len(not_loaded) > 5:
            print(f"  ... and {len(not_loaded)-5} more")

    print("=" * 80)

    return model


def main():
    parser = argparse.ArgumentParser(description="Load CompVis LDM pretrained weights")
    parser.add_argument("--pretrained", type=str, required=True,
                       help="Path to cin256-v2.ckpt")
    parser.add_argument("--config", type=str, default="configs/plantdoc_latent_pretrained.yml",
                       help="Config file for PlantDoc model")
    parser.add_argument("--output", type=str, default="pretrained_init_plantdoc.pth",
                       help="Output path for adapted weights")
    parser.add_argument("--num-classes-pretrained", type=int, default=1000,
                       help="Number of classes in pretrained model (ImageNet)")
    parser.add_argument("--num-classes-target", type=int, default=28,
                       help="Number of classes in target model (PlantDoc)")

    args = parser.parse_args()

    print("=" * 80)
    print("CompVis LDM → PlantDoc Pretrained Weight Loader")
    print("=" * 80)
    print(f"Pretrained:  {args.pretrained}")
    print(f"Config:      {args.config}")
    print(f"Output:      {args.output}")
    print(f"Classes:     {args.num_classes_pretrained} → {args.num_classes_target}")
    print("=" * 80)

    # Load config
    print("\n[1/5] Loading config...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # Create model
    print("\n[2/5] Creating PlantDoc model...")
    sys.path.insert(0, os.path.dirname(__file__))
    from models.diffusion import Model

    orig_img_size = config.data.image_size
    config.data.image_size = config.data.latent_size
    model = Model(config)
    config.data.image_size = orig_img_size

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load pretrained checkpoint
    print("\n[3/5] Loading pretrained checkpoint...")
    ldm_state_dict = load_ldm_checkpoint(args.pretrained)

    # Extract UNet weights
    print("\n[4/5] Extracting and adapting UNet weights...")
    unet_weights = extract_unet_weights(ldm_state_dict)
    adapted_weights = adapt_class_embedding(
        unet_weights,
        args.num_classes_pretrained,
        args.num_classes_target
    )

    # Load into model
    print("\n[5/5] Loading weights into model...")
    model = load_into_model(model, adapted_weights, strict=False)

    # Save adapted weights
    print(f"\nSaving adapted weights to {args.output}...")
    torch.save(model.state_dict(), args.output)
    print(f"✅ Saved!")

    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

    # Test load
    test_model = Model(config)
    test_model.load_state_dict(torch.load(args.output))
    print("✓ Weights can be loaded successfully")

    print("\n" + "=" * 80)
    print("✅ DONE! Ready for fine-tuning on PlantDoc")
    print("=" * 80)
    print(f"\nTo use in training:")
    print(f"  python main.py --config {args.config} --doc plantdoc_pretrained --plantdoc --ni")
    print()


if __name__ == "__main__":
    main()
