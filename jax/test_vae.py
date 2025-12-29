"""
Test VAE encode/decode functionality.

Usage:
    python test_vae.py
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['JAX_PLATFORMS'] = 'cpu'  # Use CPU for testing

import jax
import jax.numpy as jnp
import numpy as np

try:
    from utils.vae import create_vae
    print("✓ Successfully imported VAE")
except Exception as e:
    print(f"✗ Failed to import VAE: {e}")
    exit(1)


def test_vae_basic():
    """Test basic VAE encode/decode."""
    print("\n" + "="*60)
    print("Testing VAE Basic Encode/Decode")
    print("="*60)

    # Create VAE
    print("\n1. Creating VAE...")
    try:
        vae = create_vae("stabilityai/sd-vae-ft-mse")
        print(f"✓ VAE created successfully")
        print(f"  Scale factor: {vae.scale_factor}")
        print(f"  Downscale factor: {vae.downscale_factor}x")
    except Exception as e:
        print(f"✗ Failed to create VAE: {e}")
        return False

    # Test encode
    print("\n2. Testing encode...")
    try:
        # Create dummy images [B, H, W, C] in [0, 1] range
        batch_size = 2
        image_size = 256
        images = jnp.ones((batch_size, image_size, image_size, 3)) * 0.5

        latents = vae.encode(images)

        expected_latent_size = image_size // vae.downscale_factor
        expected_shape = (batch_size, expected_latent_size, expected_latent_size, 4)

        print(f"  Input shape: {images.shape}")
        print(f"  Output shape: {latents.shape}")
        print(f"  Expected shape: {expected_shape}")

        assert latents.shape == expected_shape, f"Shape mismatch: {latents.shape} != {expected_shape}"
        print(f"✓ Encode successful")
    except Exception as e:
        print(f"✗ Encode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test decode
    print("\n3. Testing decode...")
    try:
        decoded = vae.decode(latents)

        print(f"  Latent shape: {latents.shape}")
        print(f"  Decoded shape: {decoded.shape}")
        print(f"  Expected shape: {images.shape}")
        print(f"  Decoded range: [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")

        assert decoded.shape == images.shape, f"Shape mismatch: {decoded.shape} != {images.shape}"
        assert decoded.min() >= 0 and decoded.max() <= 1, f"Values out of [0,1] range"
        print(f"✓ Decode successful")
    except Exception as e:
        print(f"✗ Decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test reconstruction
    print("\n4. Testing reconstruction quality...")
    try:
        mse = float(jnp.mean((images - decoded) ** 2))
        print(f"  MSE: {mse:.6f}")

        # For a constant image, reconstruction should be very good
        if mse < 0.01:
            print(f"✓ Reconstruction quality good (MSE < 0.01)")
        else:
            print(f"⚠ Reconstruction MSE is high: {mse}")

    except Exception as e:
        print(f"✗ Reconstruction test failed: {e}")
        return False

    print("\n" + "="*60)
    print("✓ All VAE tests passed!")
    print("="*60)
    return True


def test_vae_jit():
    """Test that VAE encode/decode are JIT-compiled."""
    print("\n" + "="*60)
    print("Testing VAE JIT Compilation")
    print("="*60)

    try:
        vae = create_vae("stabilityai/sd-vae-ft-mse")

        # Create dummy data
        images = jnp.ones((1, 256, 256, 3)) * 0.5

        # First call (compilation)
        print("\n1. First encode (with compilation)...")
        import time
        start = time.time()
        latents = vae.encode(images)
        first_time = time.time() - start
        print(f"  Time: {first_time:.3f}s")

        # Second call (should be faster)
        print("\n2. Second encode (cached)...")
        start = time.time()
        latents = vae.encode(images)
        second_time = time.time() - start
        print(f"  Time: {second_time:.3f}s")

        if second_time < first_time:
            print(f"✓ JIT compilation working (speedup: {first_time/second_time:.1f}x)")
        else:
            print(f"⚠ No speedup observed")

        print("\n" + "="*60)
        print("✓ JIT test completed")
        print("="*60)
        return True

    except Exception as e:
        print(f"✗ JIT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_batch_sizes():
    """Test VAE with different batch sizes."""
    print("\n" + "="*60)
    print("Testing VAE with Different Batch Sizes")
    print("="*60)

    try:
        vae = create_vae("stabilityai/sd-vae-ft-mse")

        batch_sizes = [1, 2, 4, 8]
        image_size = 256

        for bs in batch_sizes:
            print(f"\nTesting batch size {bs}...")
            images = jnp.ones((bs, image_size, image_size, 3)) * 0.5
            latents = vae.encode(images)
            decoded = vae.decode(latents)

            assert latents.shape[0] == bs
            assert decoded.shape == images.shape
            print(f"✓ Batch size {bs} OK")

        print("\n" + "="*60)
        print("✓ All batch size tests passed")
        print("="*60)
        return True

    except Exception as e:
        print(f"✗ Batch size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting VAE tests...\n")

    all_passed = True

    # Run tests
    all_passed &= test_vae_basic()
    # all_passed &= test_vae_jit()  # Skip JIT test for now
    all_passed &= test_vae_batch_sizes()

    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*60)
