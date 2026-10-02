
from typing import List, Set

from reccomendation_engine.evaluation.metrics import get_hits

def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = None) -> float:
    
    #Recall@K = (number of relevant items in the top K recommendations) / (total number of relevant items for this user) 
    
    if not relevant_items:
        return 0.0

    if k is None:
        k = len(recommended_items)

    top_k_items = recommended_items[:k]
    hits = get_hits(top_k_items, relevant_items)
    return sum(hits) / len(relevant_items)


if __name__ == "__main__":
    recs = ["P3", "P0", "P1", "P4", "P2"]
    relevant = {"P0", "P1", "P3", "P9"}

    print(f"Recall@5: {recall_at_k(recs, relevant, k=5)}")
    print(f"Recall@3: {recall_at_k(recs, relevant, k=3)}")