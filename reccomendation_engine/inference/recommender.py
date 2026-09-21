
from typing import List, Tuple

from reccomendation_engine.inference.model_loader import ModelLoader
from reccomendation_engine.candidate_generation.candidate_generator import get_candidates
from reccomendation_engine.ranking.ranker import Ranker


class Recommender:
    def __init__(self, embedding_dir: str = "reccomendation_engine/exported_embedding"):
        loader = ModelLoader(embedding_dir)
        self.artifacts = loader.load()
        self.ranker = Ranker()

    def get_recommendations(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if user_id not in self.artifacts.user_id_to_idx:
            return []

        user_idx = self.artifacts.user_id_to_idx[user_id]
        user_vector = self.artifacts.user_embeddings[user_idx]

        # Ask for more candidates than top_k, since inventory filtering
        # (out-of-stock items) may drop some before ranking narrows to top_k.
        candidates = get_candidates(user_vector, top_k=max(top_k * 2, top_k), filters=None)

        if not candidates:
            return []

        ranked = self.ranker.get_top_n(candidates, n=top_k)

        # Person 3's product dicts use "item_id" as the key (from Postgres),
        # not "product_id" -- confirm this matches what the API layer expects.
        return [(item["item_id"], item["final_score"]) for item in ranked]


if __name__ == "__main__":
    rec = Recommender()
    print(rec.get_recommendations("U0", top_k=5))