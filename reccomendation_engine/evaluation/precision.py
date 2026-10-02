
from typing import List, Set

from reccomendation_engine.evaluation.metrics import get_hits


def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = None) -> float:
   
   # Precision@K = (number of relevant items in the top K recommendations) / K

  
    if k is None:
        k = len(recommended_items)

    top_k_items = recommended_items[:k]
    if not top_k_items:
        return 0.0

    hits = get_hits(top_k_items, relevant_items)
    return sum(hits) / len(top_k_items)


if __name__ == "__main__":
    recs = ["P3", "P0", "P1", "P4", "P2"]
    relevant = {"P0", "P1", "P3"}

    print(f"Precision@5: {precision_at_k(recs, relevant, k=5)}")
    print(f"Precision@3: {precision_at_k(recs, relevant, k=3)}")