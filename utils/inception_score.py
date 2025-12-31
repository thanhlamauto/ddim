"""
Inception Score (IS) calculation using torch-fidelity library.
"""

import os
import logging
from pathlib import Path


def calculate_inception_score(image_dir, batch_size=50, splits=10, device='cuda'):
    """
    Calculate Inception Score for generated images.

    Args:
        image_dir (str): Directory containing generated images
        batch_size (int): Batch size for processing images (default: 50)
        splits (int): Number of splits for IS calculation (default: 10)
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        tuple: (is_mean, is_std) - mean and std of Inception Score
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        logging.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        return 0.0, 0.0

    image_dir = str(Path(image_dir).resolve())

    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return 0.0, 0.0

    # Count images in directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    num_images = sum(1 for f in Path(image_dir).rglob('*')
                     if f.suffix.lower() in image_extensions)

    if num_images == 0:
        logging.error(f"No images found in directory: {image_dir}")
        return 0.0, 0.0

    logging.info(f"Calculating Inception Score for {num_images} images in {image_dir}")

    try:
        # Calculate IS using torch-fidelity
        # Note: torch-fidelity automatically handles image loading and preprocessing
        metrics = calculate_metrics(
            input1=image_dir,
            isc=True,  # Calculate Inception Score
            isc_splits=splits,
            batch_size=batch_size,
            cuda=(device == 'cuda'),
            verbose=False
        )

        is_mean = metrics['inception_score_mean']
        is_std = metrics['inception_score_std']

        logging.info(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

        return is_mean, is_std

    except Exception as e:
        logging.error(f"Error calculating Inception Score: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 0.0, 0.0


if __name__ == "__main__":
    # Test the function
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inception_score.py <image_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    logging.basicConfig(level=logging.INFO)

    is_mean, is_std = calculate_inception_score(image_dir, batch_size=batch_size)
    print(f"\nInception Score: {is_mean:.2f} ± {is_std:.2f}")
