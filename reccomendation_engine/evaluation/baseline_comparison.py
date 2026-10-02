
from collections import Counter
from typing import List

from reccomendation_engine.convertor import interactions, num_users, num_items
from reccomendation_engine.inference.inference import get_recommendations
from reccomendation_engine.evaluation.metrics import default_ground_truth
from reccomendation_engine.evaluation.precision import precision_at_k
from reccomendation_engine.evaluation.recall import recall_at_k
from reccomendation_engine.evaluation.ndcg import ndcg_at_k


def get_popularity_baseline(top_k: int = 5) -> List[str]:
    """
    Returns the top_k most-interacted-with products across ALL users,
    as a single fixed list -- zero personalization, pure popularity.
    """
    item_counts = Counter(item_idx for _, item_idx in interactions)
    most_common = item_counts.most_common(top_k)
    return [f"P{item_idx}" for item_idx, _count in most_common]


def compare_to_baseline(top_k: int = 5) -> dict:
    ground_truth = default_ground_truth()
    baseline_recs = get_popularity_baseline(top_k=top_k)

    gnn_scores = {"precision": [], "recall": [], "ndcg": []}
    baseline_scores = {"precision": [], "recall": [], "ndcg": []}

    for i in range(num_users):
        user_id = f"U{i}"
        relevant = ground_truth.get(user_id, set())

        gnn_recs = [pid for pid, score in get_recommendations(user_id, top_k=top_k)]
        gnn_scores["precision"].append(precision_at_k(gnn_recs, relevant, k=top_k))
        gnn_scores["recall"].append(recall_at_k(gnn_recs, relevant, k=top_k))
        gnn_scores["ndcg"].append(ndcg_at_k(gnn_recs, relevant, k=top_k))

        baseline_scores["precision"].append(precision_at_k(baseline_recs, relevant, k=top_k))
        baseline_scores["recall"].append(recall_at_k(baseline_recs, relevant, k=top_k))
        baseline_scores["ndcg"].append(ndcg_at_k(baseline_recs, relevant, k=top_k))

    def avg(scores):
        return sum(scores) / len(scores)

    return {
        "gnn": {metric: avg(values) for metric, values in gnn_scores.items()},
        "baseline": {metric: avg(values) for metric, values in baseline_scores.items()},
        "k": top_k,
    }


def print_comparison(results: dict) -> None:
    k = results["k"]
    gnn = results["gnn"]
    baseline = results["baseline"]

    print(f"{'Metric':<14} {'GNN Model':<12} {'Popularity Baseline':<20} {'Improvement':<10}")
    print("-" * 58)
    for metric in ["precision", "recall", "ndcg"]:
        gnn_val = gnn[metric]
        base_val = baseline[metric]
        improvement = gnn_val - base_val
        sign = "+" if improvement >= 0 else ""
        print(f"{metric + '@' + str(k):<14} {gnn_val:<12.3f} {base_val:<20.3f} {sign}{improvement:.3f}")

    print()
    print("CAVEAT: both measured against training interactions, not a")
    print("held-out test set -- comparison shows RELATIVE gain over a naive")
    print("baseline, but absolute numbers for both sides are still optimistic")
    print("until Person 1 ships a proper train/test split.")


if __name__ == "__main__":
    results = compare_to_baseline(top_k=5)
    print_comparison(results)