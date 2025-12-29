"""
Fix import compatibility for Kaggle TPU environment.

Run this before importing any JAX/Flax code.
"""

import sys
import warnings

def fix_jax_imports():
    """Fix JAX experimental.maps import for older Flax versions."""
    try:
        import jax
        from jax import experimental

        # Check if maps exists
        if not hasattr(experimental, 'maps'):
            print("⚠ JAX experimental.maps not found, creating compatibility shim...")

            # Create a dummy maps module for backward compatibility
            from types import ModuleType
            maps = ModuleType('maps')

            # Add commonly used functions (as no-ops or aliases)
            from jax.experimental import mesh_utils
            maps.mesh = lambda *args, **kwargs: None
            maps.Mesh = lambda *args, **kwargs: None

            # Inject it into jax.experimental
            experimental.maps = maps
            sys.modules['jax.experimental.maps'] = maps

            print("✓ JAX compatibility shim installed")

    except ImportError as e:
        print(f"⚠ Could not fix JAX imports: {e}")


def check_versions():
    """Check and report package versions."""
    try:
        import jax
        import flax
        import optax

        print("=" * 60)
        print("Package Versions:")
        print("=" * 60)
        print(f"JAX:    {jax.__version__}")
        print(f"Flax:   {flax.__version__}")
        print(f"Optax:  {optax.__version__}")
        print("=" * 60)

        # Check JAX version
        jax_version = tuple(map(int, jax.__version__.split('.')[:2]))
        if jax_version >= (0, 4):
            print("✓ JAX version OK for TPU")
        else:
            print("⚠ JAX version may be too old")

        # Check Flax version
        flax_version = tuple(map(int, flax.__version__.split('.')[:2]))
        if (0, 7) <= flax_version < (0, 9):
            print("✓ Flax version OK")
        else:
            print("⚠ Flax version may have compatibility issues")

    except ImportError as e:
        print(f"⚠ Could not check versions: {e}")


def setup_kaggle_environment():
    """Complete setup for Kaggle TPU environment."""
    import os

    # Set environment variables if not already set
    env_vars = {
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
        'JAX_PLATFORMS': 'tpu',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    print("✓ Environment variables set")

    # Apply fixes
    fix_jax_imports()
    check_versions()

    # Suppress warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*os.fork.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*hugepages.*')

    print("✓ Kaggle environment setup complete")


if __name__ == "__main__":
    setup_kaggle_environment()
