from typing import Dict, Optional, Tuple
import os
import torch
import numpy as np

from reccomendation_engine.convertor import num_users, num_items
from reccomendation_engine.models.matrix_create import make_n_adj_mat
from reccomendation_engine.models.lightgcn import LightGCN
from reccomendation_engine.training.checkpoint import CheckpointManager


def generate_and_save_embeddings(
    model: Optional[LightGCN] = None,
    checkpoint_path: Optional[str] = "models/checkpoints/lightgcn_best.pt",
    save_dir: str = "reccomendation_engine/embeddings",
    embedding_dim: int = 64,
    num_layers: int = 3,
    save_numpy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts final GNN representations for all users and items and saves them to disk.
    Compatible with Person 3's SimilarityCalculator and FAISS indexing.

    Returns:
        user_embeddings: torch.Tensor of shape (num_users, embedding_dim)
        product_embeddings: torch.Tensor of shape (num_items, embedding_dim)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load or prepare graph normalized adjacency
    n_adj_np = make_n_adj_mat()
    n_adj = torch.tensor(n_adj_np, dtype=torch.float32)

    # 2. Prepare model
    if model is None:
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint_mgr = CheckpointManager()
            checkpoint_mgr.load_checkpoint(checkpoint_path, model)
        else:
            print("Warning: Checkpoint not found. Generating embeddings from initialized model.")

    model.eval()

    # 3. Forward pass to get final representations
    with torch.no_grad():
        user_embeddings, product_embeddings, _ = model(n_adj)

    print(f"Computed User Embeddings: {user_embeddings.shape}")
    print(f"Computed Product Embeddings: {product_embeddings.shape}")

    # 4. Save PyTorch tensors (.pt)
    user_pt_path = os.path.join(save_dir, "user_embeddings.pt")
    product_pt_path = os.path.join(save_dir, "product_embeddings.pt")

    torch.save(user_embeddings, user_pt_path)
    torch.save(product_embeddings, product_pt_path)
    print(f"Saved: {user_pt_path}")
    print(f"Saved: {product_pt_path}")

    # 5. Optionally save NumPy arrays (.npy) for Candidate Generation
    if save_numpy:
        user_npy_path = os.path.join(save_dir, "user_embeddings.npy")
        product_npy_path = os.path.join(save_dir, "product_embeddings.npy")
        np.save(user_npy_path, user_embeddings.cpu().numpy())
        np.save(product_npy_path, product_embeddings.cpu().numpy())
        print(f"Saved: {user_npy_path}")
        print(f"Saved: {product_npy_path}")

    # 6. Save metadata mapping (user/item ID list)
    metadata_path = os.path.join(save_dir, "embedding_metadata.pt")
    metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "user_ids": [f"U{i}" for i in range(num_users)],
        "product_ids": [f"P{i}" for i in range(num_items)],
    }
    torch.save(metadata, metadata_path)
    print(f"Saved: {metadata_path}")

    return user_embeddings, product_embeddings


def load_embeddings(
    embedding_dir: str = "reccomendation_engine/embeddings",
    as_numpy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience loader to fetch user and item embeddings and metadata.
    """
    user_pt = os.path.join(embedding_dir, "user_embeddings.pt")
    item_pt = os.path.join(embedding_dir, "product_embeddings.pt")
    meta_pt = os.path.join(embedding_dir, "embedding_metadata.pt")

    if not os.path.exists(user_pt) or not os.path.exists(item_pt):
        raise FileNotFoundError(f"Embeddings not found in {embedding_dir}. Run generate_embeddings.py first.")

    user_emb = torch.load(user_pt)
    item_emb = torch.load(item_pt)
    meta = torch.load(meta_pt) if os.path.exists(meta_pt) else {}

    if as_numpy:
        return user_emb.cpu().numpy(), item_emb.cpu().numpy(), meta
    return user_emb, item_emb, meta


if __name__ == "__main__":
    generate_and_save_embeddings()
