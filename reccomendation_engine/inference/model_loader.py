"""
Person 4 scope: Model/artifact loading.

Separates "how do we get embeddings off disk" from "what do we do with them"
(recommender.py). This means if Person 2 changes how/where embeddings are
saved, only this file needs to change -- recommender.py and inference.py
don't need to know or care.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from reccomendation_engine.embeddings import load_embeddings


@dataclass
class ModelArtifacts:
    """Container for everything inference needs, loaded once and reused."""
    user_embeddings: np.ndarray       # shape (num_users, embedding_dim)
    product_embeddings: np.ndarray    # shape (num_items, embedding_dim)
    user_ids: List[str]
    product_ids: List[str]
    user_id_to_idx: Dict[str, int]


class ModelLoader:
    """
    Loads Person 2's trained embeddings + metadata from disk and hands back
    a ready-to-use ModelArtifacts object.

    Usage:
        loader = ModelLoader()
        artifacts = loader.load()
    """

    def __init__(self, embedding_dir: str = "reccomendation_engine/embeddings"):
        self.embedding_dir = embedding_dir

    def load(self) -> ModelArtifacts:
        try:
            user_emb, product_emb, metadata = load_embeddings(
                self.embedding_dir, as_numpy=True
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not load embeddings from '{self.embedding_dir}'. "
                f"Has Person 2's generate_embeddings.py been run yet? "
                f"Original error: {e}"
            ) from e

        user_ids: List[str] = metadata.get("user_ids", [])
        product_ids: List[str] = metadata.get("product_ids", [])

        if not user_ids or not product_ids:
            raise RuntimeError(
                "Embedding metadata is missing 'user_ids' or 'product_ids'. "
                "Check embedding_metadata.pt was generated correctly."
            )

        if len(user_ids) != user_emb.shape[0]:
            raise RuntimeError(
                f"Mismatch: {len(user_ids)} user_ids but "
                f"{user_emb.shape[0]} user embedding rows."
            )
        if len(product_ids) != product_emb.shape[0]:
            raise RuntimeError(
                f"Mismatch: {len(product_ids)} product_ids but "
                f"{product_emb.shape[0]} product embedding rows."
            )

        return ModelArtifacts(
            user_embeddings=user_emb,
            product_embeddings=product_emb,
            user_ids=user_ids,
            product_ids=product_ids,
            user_id_to_idx={uid: i for i, uid in enumerate(user_ids)},
        )


if __name__ == "__main__":
    loader = ModelLoader()
    artifacts = loader.load()
    print(f"Loaded {len(artifacts.user_ids)} users, {len(artifacts.product_ids)} products")
    print(f"Embedding dim: {artifacts.user_embeddings.shape[1]}")