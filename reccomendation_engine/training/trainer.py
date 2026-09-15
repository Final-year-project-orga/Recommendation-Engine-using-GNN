from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.nn as nn
from reccomendation_engine.training.losses import BPRLoss
from reccomendation_engine.training.negative_sampling import NegativeSampler
from reccomendation_engine.training.checkpoint import CheckpointManager


class Trainer:
    """
    Trainer for GNN-based recommendation models.
    Supports LightGCN, GraphSAGE, and GAT using BPR loss and implicit feedback.
    """
    def __init__(
        self,
        model: nn.Module,
        n_adj: torch.Tensor,
        sampler: NegativeSampler,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        device: Optional[torch.device] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.device = device if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        if not isinstance(n_adj, torch.Tensor):
            self.n_adj = torch.tensor(n_adj, dtype=torch.float32, device=self.device)
        else:
            self.n_adj = n_adj.to(device=self.device)

        self.sampler = sampler
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager

        # Loss function
        self.loss_fn = loss_fn if loss_fn is not None else BPRLoss(reg_weight=weight_decay)

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=0.0  # L2 is handled in BPRLoss on ego embeddings
            )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "bpr_loss": [],
            "reg_loss": [],
            "time_sec": [],
        }

    def train_epoch(self) -> Tuple[float, float, float]:
        """Runs a single training epoch over all interaction pairs."""
        self.model.train()

        # Sample (user, pos_item, neg_item) triplets for this epoch
        users, pos_items, neg_items = self.sampler.sample_epoch_triplets()
        num_samples = len(users)

        # Shuffle triplets
        perm = torch.randperm(num_samples)
        users = users[perm].to(self.device)
        pos_items = pos_items[perm].to(self.device)
        neg_items = neg_items[perm].to(self.device)

        total_loss_accum = 0.0
        bpr_loss_accum = 0.0
        reg_loss_accum = 0.0
        num_batches = 0

        # Mini-batch iteration
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_users = users[start_idx:end_idx]
            batch_pos_items = pos_items[start_idx:end_idx]
            batch_neg_items = neg_items[start_idx:end_idx]

            self.optimizer.zero_grad()

            # Forward pass to get all final embeddings
            user_final_emb, item_final_emb, _ = self.model(self.n_adj)

            u_emb = user_final_emb[batch_users]
            pos_i_emb = item_final_emb[batch_pos_items]
            neg_i_emb = item_final_emb[batch_neg_items]

            # Calculate predicted preference scores (dot product)
            pos_scores = torch.sum(u_emb * pos_i_emb, dim=-1)
            neg_scores = torch.sum(u_emb * neg_i_emb, dim=-1)

            # Get initial ego embeddings for L2 regularization
            user_ego = self.model.user_embedding(batch_users)
            pos_item_ego = self.model.item_embedding(batch_pos_items)
            neg_item_ego = self.model.item_embedding(batch_neg_items)

            # Compute BPR loss
            loss, bpr_loss, reg_loss = self.loss_fn(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                user_ego_emb=user_ego,
                pos_item_ego_emb=pos_item_ego,
                neg_item_ego_emb=neg_item_ego,
            )

            loss.backward()
            self.optimizer.step()

            total_loss_accum += loss.item()
            bpr_loss_accum += bpr_loss.item()
            reg_loss_accum += reg_loss.item()
            num_batches += 1

        avg_total = total_loss_accum / max(num_batches, 1)
        avg_bpr = bpr_loss_accum / max(num_batches, 1)
        avg_reg = reg_loss_accum / max(num_batches, 1)

        return avg_total, avg_bpr, avg_reg

    def fit(
        self,
        epochs: int = 50,
        print_every: int = 10,
        save_best: bool = True,
        model_name: str = "lightgcn_best.pt",
    ) -> Dict[str, List[float]]:
        """
        Runs the full training loop for specified number of epochs.
        """
        best_loss = float("inf")
        print(f"Starting training on device: {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            total_loss, bpr_loss, reg_loss = self.train_epoch()
            elapsed = time.time() - t0

            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(total_loss)
            self.history["bpr_loss"].append(bpr_loss)
            self.history["reg_loss"].append(reg_loss)
            self.history["time_sec"].append(elapsed)

            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Total Loss: {total_loss:.4f} | "
                    f"BPR Loss: {bpr_loss:.4f} | "
                    f"Reg Loss: {reg_loss:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )

            if save_best and self.checkpoint_manager is not None and total_loss < best_loss:
                best_loss = total_loss
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=total_loss,
                    filename=model_name,
                )

        print(f"Training completed. Best Total Loss: {best_loss:.4f}")
        return self.history
