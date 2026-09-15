from typing import Any, Dict, Optional
import os
import torch


class CheckpointManager:
    """
    Handles saving, loading, and versioning of model checkpoints and metadata.
    """
    def __init__(self, save_dir: str = "models/checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        filename: str = "checkpoint.pt",
    ) -> str:
        """
        Saves model weights, optimizer state, and training metadata.

        Returns:
            filepath: Full path to the saved checkpoint
        """
        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {},
            "config": config or {},
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Loads state into model (and optionally optimizer) from a checkpoint file.

        Returns:
            checkpoint_data: Dictionary of metadata (epoch, loss, metrics, config)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        map_location = device if device is not None else torch.device("cpu")
        checkpoint = torch.load(filepath, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
