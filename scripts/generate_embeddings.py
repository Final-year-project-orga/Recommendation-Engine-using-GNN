"""
Script to generate and export user & product embeddings using trained LightGCN.
execution command: python scripts/generate_embeddings.py
"""
import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reccomendation_engine.embeddings.generate_embeddings import generate_and_save_embeddings

if __name__ == "__main__":
    generate_and_save_embeddings(
        checkpoint_path="models/checkpoints/lightgcn_best.pt",
        save_dir="reccomendation_engine/embeddings",
        embedding_dim=64,
        num_layers=3,
        save_numpy=True,
    )
