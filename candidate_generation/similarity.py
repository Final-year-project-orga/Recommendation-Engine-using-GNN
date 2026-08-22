from typing import Tuple
import numpy as np

class SimilarityCalculator:
    """
    Utility class for similarity computations
    between user and item embeddings.
    """

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray,vector_b: np.ndarray) -> float:
        numerator = np.dot(vector_a, vector_b)

        denominator = (np.linalg.norm(vector_a)* np.linalg.norm(vector_b))

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def dot_product(vector_a: np.ndarray,vector_b: np.ndarray) -> float:
        
        return float(np.dot(vector_a, vector_b))

    @staticmethod
    def euclidean_distance(vector_a: np.ndarray,vector_b: np.ndarray) -> float:

        return float(
            np.linalg.norm(vector_a - vector_b)
        )

    @staticmethod
    def batch_cosine_similarity(query_vector: np.ndarray,item_vectors: np.ndarray) -> np.ndarray:
       
        """ Compute cosine similarity between one query vector and many item vectors."""

        query_norm = np.linalg.norm(query_vector)

        item_norms = np.linalg.norm(item_vectors, axis=1)

        similarities = (np.dot(item_vectors, query_vector)/(item_norms * query_norm + 1e-10))

        return similarities

    @staticmethod
    def batch_dot_product(query_vector: np.ndarray,item_vectors: np.ndarray) -> np.ndarray:
        """ Compute dot products between one query vector and many item vectors."""

        return np.dot(item_vectors, query_vector)

    @staticmethod
    def batch_euclidean_distance(query_vector: np.ndarray,item_vectors: np.ndarray) -> np.ndarray:

        """ Compute Euclidean distance between query vector and item vectors."""

        return np.linalg.norm( item_vectors - query_vector,axis=1)

    @staticmethod
    def top_k_by_cosine(query_vector: np.ndarray,item_vectors: np.ndarray,k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """ Return top-k most similar items using cosine similarity."""

        scores = (SimilarityCalculator.batch_cosine_similarity(query_vector,item_vectors))

        top_indices = np.argsort(scores)[::-1][:k]

        return top_indices, scores[top_indices]

    @staticmethod
    def top_k_by_dot_product(query_vector: np.ndarray, item_vectors: np.ndarray,k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # Return top-k items using dot product.

        scores = (SimilarityCalculator.batch_dot_product(query_vector,item_vectors))

        top_indices = np.argsort(scores)[::-1][:k]

        return top_indices, scores[top_indices]

    @staticmethod
    def top_k_by_distance(query_vector: np.ndarray,item_vectors: np.ndarray,k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        
        # Return top-k nearest items using Euclidean distance.
    
        distances = (SimilarityCalculator.batch_euclidean_distance(query_vector,item_vectors))

        top_indices = np.argsort(distances)[:k]

        return top_indices, distances[top_indices]