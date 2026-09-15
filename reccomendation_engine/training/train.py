import argparse
import os
import torch

from reccomendation_engine.convertor import num_users, num_items, interactions
from reccomendation_engine.models.matrix_create import make_n_adj_mat
from reccomendation_engine.models.lightgcn import LightGCN
from reccomendation_engine.training.losses import BPRLoss
from reccomendation_engine.training.negative_sampling import NegativeSampler
from reccomendation_engine.training.checkpoint import CheckpointManager
from reccomendation_engine.training.trainer import Trainer


def run_training(
    epochs: int = 50,
    embedding_dim: int = 64,
    num_layers: int = 3,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    save_dir: str = "models/checkpoints",
    model_name: str = "lightgcn_best.pt",
):
    """
    Train the LightGCN model on the recommendation graph.
    """
    print("=" * 60)
    print("LightGCN Recommendation Engine Training")
    print("=" * 60)
    print(f"Users: {num_users} | Items: {num_items} | Interactions: {len(interactions)}")
    print(f"Embedding Dim: {embedding_dim} | GNN Layers: {num_layers} | LR: {lr}")

    # 1. Build normalized adjacency matrix
    print("\n[1/4] Constructing normalized adjacency matrix...")
    n_adj_np = make_n_adj_mat()
    n_adj = torch.tensor(n_adj_np, dtype=torch.float32)

    # 2. Instantiate model
    print("[2/4] Initializing LightGCN model...")
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )

    # 3. Setup negative sampler and loss
    print("[3/4] Setting up Negative Sampler & BPR Loss...")
    sampler = NegativeSampler(
        num_users=num_users,
        num_items=num_items,
        interactions=interactions,
    )
    loss_fn = BPRLoss(reg_weight=weight_decay)

    # 4. Setup CheckpointManager & Trainer
    checkpoint_mgr = CheckpointManager(save_dir=save_dir)
    trainer = Trainer(
        model=model,
        n_adj=n_adj,
        sampler=sampler,
        loss_fn=loss_fn,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        checkpoint_manager=checkpoint_mgr,
    )

    # 5. Train
    print("[4/4] Starting training loop...")
    history = trainer.fit(
        epochs=epochs,
        print_every=max(1, epochs // 5),
        save_best=True,
        model_name=model_name,
    )

    saved_path = os.path.join(save_dir, model_name)
    print(f"\nModel checkpoint saved at: {saved_path}")
    return model, trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGCN Recommendation Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of LightGCN layers")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 regularization weight")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        lr=args.lr,
        weight_decay=args.reg,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )
