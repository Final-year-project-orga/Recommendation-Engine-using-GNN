"""
(Normalized Discounted Cumulative Gain).
"""

import math
from typing import List, Set

from reccomendation_engine.evaluation.metrics import get_hits


def dcg_at_k(hits: List[int]) -> float:
   
    return sum(
        hit / math.log2(position + 2)
        for position, hit in enumerate(hits)
    )


def ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = None) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    """
    if not relevant_items:
        return 0.0

    if k is None:
        k = len(recommended_items)

    top_k_items = recommended_items[:k]
    hits = get_hits(top_k_items, relevant_items)
    actual_dcg = dcg_at_k(hits)

    num_relevant_in_top_k = min(len(relevant_items), k)
    ideal_hits = [1] * num_relevant_in_top_k
    ideal_dcg = dcg_at_k(ideal_hits)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


if __name__ == "__main__":
    recs_good_order = ["P0", "P1", "P9", "P8", "P7"]
    recs_bad_order = ["P9", "P8", "P0", "P1", "P7"]
    relevant = {"P0", "P1"}

    print(f"NDCG (relevant items ranked first): {ndcg_at_k(recs_good_order, relevant, k=5):.4f}")
    print(f"NDCG (relevant items buried):        {ndcg_at_k(recs_bad_order, relevant, k=5):.4f}")