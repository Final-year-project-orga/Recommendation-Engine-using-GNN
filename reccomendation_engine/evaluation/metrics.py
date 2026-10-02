
from typing import Dict, List, Set, Tuple

from reccomendation_engine.convertor import interactions, num_users


def build_ground_truth(
    interactions: List[Tuple[int, int]],
    user_id_prefix: str = "U",
    product_id_prefix: str = "P",
) -> Dict[str, Set[str]]:

   # Converts raw (user_idx, item_idx) interaction tuples into format matching the string IDs
    
    ground_truth: Dict[str, Set[str]] = {}
    for user_idx, item_idx in interactions:
        user_id = f"{user_id_prefix}{user_idx}"
        product_id = f"{product_id_prefix}{item_idx}"
        ground_truth.setdefault(user_id, set()).add(product_id)
    return ground_truth


def get_hits(recommended_items: List[str], relevant_items: Set[str]) -> List[int]:
   # For a ranked list of recommended product_ids, returns a same-length list
   
    return [1 if item in relevant_items else 0 for item in recommended_items]


def default_ground_truth() -> Dict[str, Set[str]]:
    
    return build_ground_truth(interactions)


if __name__ == "__main__":
    gt = default_ground_truth()
    print("Ground truth for U0:", gt.get("U0"))

    example_recs = ["P3", "P0", "P1", "P4", "P2"]
    hits = get_hits(example_recs, gt.get("U0", set()))
    print(f"Recommended: {example_recs}")
    print(f"Hits:        {hits}")