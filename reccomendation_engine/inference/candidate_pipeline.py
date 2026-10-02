
from typing import List, Optional

from reccomendation_engine.candidate_generation.chroma_client import ChromaDBClient
from reccomendation_engine.database.inventory_service import get_available_items
from reccomendation_engine.candidate_generation.ranker import Ranker


def get_ranked_candidates(user_embedding, top_k: int = 10, filters: Optional[dict] = None) -> List[dict]:
    db = ChromaDBClient()

    results = db.query(user_embedding, top_k=max(top_k * 2, top_k), filters=filters)

    candidate_ids = results["ids"][0]
    distances = results["distances"][0]

    available_products = get_available_items(candidate_ids)

    filtered_candidates = []
    for item_id, distance in zip(candidate_ids, distances):
        if item_id in available_products:
            product = available_products[item_id].copy()
            product["distance"] = distance
            filtered_candidates.append(product)

    if not filtered_candidates:
        return []

    ranker = Ranker()
    return ranker.get_top_n(filtered_candidates, n=top_k)