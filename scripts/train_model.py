"""
Script to train the GNN recommendation model (LightGCN).
Run: python scripts/train_model.py
"""
import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reccomendation_engine.training.train import run_training

if __name__ == "__main__":
    run_training(
        epochs=50,
        embedding_dim=64,
        num_layers=3,
        lr=0.01,
        weight_decay=1e-4,
        batch_size=64,
        save_dir="models/checkpoints",
        model_name="lightgcn_best.pt",
    )
