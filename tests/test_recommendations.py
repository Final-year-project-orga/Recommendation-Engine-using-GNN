"""
Person 4 scope: Tests for the inference layer (ModelLoader + Recommender).

Run with: pytest tests/test_recommender.py -v
"""

import pytest

from reccomendation_engine.inference.model_loader import ModelLoader, ModelArtifacts
from reccomendation_engine.inference.recommender import Recommender


# ---------- ModelLoader tests ----------

def test_model_loader_loads_real_embeddings():
    loader = ModelLoader()
    artifacts = loader.load()

    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.user_embeddings.shape[0] == len(artifacts.user_ids)
    assert artifacts.product_embeddings.shape[0] == len(artifacts.product_ids)
    assert artifacts.user_embeddings.shape[1] == artifacts.product_embeddings.shape[1]


def test_model_loader_builds_correct_id_index():
    loader = ModelLoader()
    artifacts = loader.load()

    for idx, uid in enumerate(artifacts.user_ids):
        assert artifacts.user_id_to_idx[uid] == idx


def test_model_loader_raises_clear_error_for_missing_dir():
    loader = ModelLoader(embedding_dir="reccomendation_engine/embeddings_DOES_NOT_EXIST")

    with pytest.raises(RuntimeError, match="Has Person 2's generate_embeddings.py been run yet"):
        loader.load()


# ---------- Recommender tests ----------

@pytest.fixture(scope="module")
def recommender():
    return Recommender()


def test_recommender_returns_expected_shape(recommender):
    results = recommender.get_recommendations("U0", top_k=5)

    assert isinstance(results, list)
    assert len(results) == 5
    for product_id, score in results:
        assert isinstance(product_id, str)
        assert isinstance(score, float)


def test_recommender_results_are_sorted_descending(recommender):
    results = recommender.get_recommendations("U0", top_k=10)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_recommender_unknown_user_returns_empty_list(recommender):
    results = recommender.get_recommendations("USER_NOT_IN_MAPPING_XYZ", top_k=5)
    assert results == []


def test_recommender_top_k_never_exceeds_available_products(recommender):
    results = recommender.get_recommendations("U0", top_k=9999)
    assert len(results) == len(recommender.artifacts.product_ids)


def test_recommender_scores_are_valid_cosine_range(recommender):
    # Cosine similarity is always in [-1, 1]
    results = recommender.get_recommendations("U0", top_k=10)
    for _, score in results:
        assert -1.0 <= score <= 1.0