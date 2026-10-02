
from reccomendation_engine.inference.inference import get_recommendations
from reccomendation_engine.evaluation.metrics import default_ground_truth
from reccomendation_engine.evaluation.precision import precision_at_k
from reccomendation_engine.evaluation.recall import recall_at_k
from reccomendation_engine.evaluation.ndcg import ndcg_at_k
from reccomendation_engine.convertor import num_users


def evaluate_all_users(top_k: int = 5, user_id_prefix: str = "U") -> dict:
    ground_truth = default_ground_truth()

    per_user_results = {}
    for i in range(num_users):
        user_id = f"{user_id_prefix}{i}"

        recs = get_recommendations(user_id, top_k=top_k)
        recommended_ids = [product_id for product_id, score in recs]
        relevant = ground_truth.get(user_id, set())

        per_user_results[user_id] = {
            "precision": precision_at_k(recommended_ids, relevant, k=top_k),
            "recall": recall_at_k(recommended_ids, relevant, k=top_k),
            "ndcg": ndcg_at_k(recommended_ids, relevant, k=top_k),
        }

    num_users_evaluated = len(per_user_results)
    averages = {
        "precision": sum(r["precision"] for r in per_user_results.values()) / num_users_evaluated,
        "recall": sum(r["recall"] for r in per_user_results.values()) / num_users_evaluated,
        "ndcg": sum(r["ndcg"] for r in per_user_results.values()) / num_users_evaluated,
    }

    return {"per_user": per_user_results, "average": averages, "k": top_k}


def print_report(results: dict) -> None:
    k = results["k"]
    print(f"{'User':<6} {'Precision@' + str(k):<14} {'Recall@' + str(k):<12} {'NDCG@' + str(k):<10}")
    print("-" * 44)
    for user_id, metrics in results["per_user"].items():
        print(f"{user_id:<6} {metrics['precision']:<14.3f} {metrics['recall']:<12.3f} {metrics['ndcg']:<10.3f}")
    print("-" * 44)
    avg = results["average"]
    print(f"{'AVG':<6} {avg['precision']:<14.3f} {avg['recall']:<12.3f} {avg['ndcg']:<10.3f}")
    print()
    print("CAVEAT: measured against training interactions, not a held-out")
    print("test set -- treat as a pipeline sanity check, not a real")
    print("performance claim, until Person 1 ships a proper train/test split.")


if __name__ == "__main__":
    results = evaluate_all_users(top_k=5)
    print_report(results)