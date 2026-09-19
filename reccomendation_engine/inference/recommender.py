"""
Person 4 scope: Inference layer.

Wires together:
  - model_loader.py (Person 2's trained embeddings, loaded and validated)
  - Person 3's similarity utilities (reccomendation_engine.candidate_generation.similarity)

This is the real integration point the pipeline was designed for -- Person 2's
generate_embeddings.py explicitly notes its output is "Compatible with Person 3's
SimilarityCalculator and FAISS indexing." No mocking needed for this part.

NOTE: Person 3's `candidate_generator.py` and `faiss_index.py` (for scaling past
a handful of products) and the `filtering/` + `ranking/` stages don't exist yet.
This class covers candidate generation via brute-force cosine similarity only --
swap `_get_top_k_indices` for a FAISS query once that's ready, and call
filtering/ranking here once Person 3 ships those.
"""

from typing import List, Tuple

from reccomendation_engine.inference.model_loader import ModelLoader
from reccomendation_engine.candidate_generation.similarity import SimilarityCalculator


class Recommender:
    #This is called delegation: Recommender just passes the parameter through rather than hardcoding it twice."embedding_dir"
    def __init__(self, embedding_dir: str = "reccomendation_engine/embeddings"):
        # Loading is now the ModelLoader's job -- it handles missing files,
        # metadata validation, and shape checks. This class only does inference.
        loader = ModelLoader(embedding_dir)
        self.artifacts = loader.load()

    def get_recommendations(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns a list of (product_id, score) tuples, ranked highest first.
        Empty list means "no recommendations" (e.g. unknown user) -- the API
        layer turns that into a 404.
        """
        if user_id not in self.artifacts.user_id_to_idx:
            return []
        # This is a guard clause — handle the edge case first, and exit early, 
        # so the rest of the function can assume "the user definitely exists" without nested if/else blocks.

        user_idx = self.artifacts.user_id_to_idx[user_id]
        user_vector = self.artifacts.user_embeddings[user_idx]

        k = min(top_k, len(self.artifacts.product_ids))
        top_indices, scores = SimilarityCalculator.top_k_by_cosine(
            query_vector=user_vector,
            item_vectors=self.artifacts.product_embeddings,
            k=k,
        )

        return [
            (self.artifacts.product_ids[idx], float(score))
            for idx, score in zip(top_indices, scores)
        ]


if __name__ == "__main__":
    rec = Recommender()
    print(rec.get_recommendations("U0", top_k=5))