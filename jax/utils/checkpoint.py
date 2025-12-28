"""Checkpoint management with Orbax."""

import os
from typing import Any, Optional
import jax
import orbax.checkpoint as ocp


class CheckpointManager:
    """Checkpoint manager using Orbax."""

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=1,
        )

        self.manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        )

    def save(self, step: int, state: Any, metrics: Optional[dict] = None):
        """Save checkpoint.

        Args:
            step: Current training step
            state: Training state to save
            metrics: Optional metrics to save with checkpoint
        """
        self.manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )
        print(f"Saved checkpoint at step {step}")

    def restore(self, step: Optional[int] = None) -> Any:
        """Restore checkpoint.

        Args:
            step: Step to restore. If None, restore latest.

        Returns:
            Restored state
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            return None

        state = self.manager.restore(step)
        print(f"Restored checkpoint from step {step}")
        return state

    def latest_step(self) -> Optional[int]:
        """Get latest checkpoint step."""
        return self.manager.latest_step()

    def all_steps(self):
        """Get all available checkpoint steps."""
        return self.manager.all_steps()
