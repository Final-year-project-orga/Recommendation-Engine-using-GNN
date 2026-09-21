

from typing import List, Tuple

from reccomendation_engine.inference.recommender import Recommender

# Loaded once at import time -- embeddings don't change between requests,
# so there's no reason to reload them on every call.
_recommender = Recommender()


def get_recommendations(user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
   
    candidates = _recommender.get_recommendations(user_id, top_k=top_k)

    # candidates = apply_filters(candidates)
    # candidates = apply_ranking(candidates)

    return candidates


if __name__ == "__main__":
    print(get_recommendations("U0", top_k=5))