from typing import Dict, List, Optional, Set, Tuple
import random
import torch
import numpy as np


class NegativeSampler:
    """
    Negative Sampler for Implicit Feedback in Graph Recommendation.
    For each user-item interaction (u, i), samples an item j that user u has NOT interacted with.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        interactions: List[Tuple[int, int]],
        seed: Optional[int] = 42,
    ):
        """
        Args:
            num_users: Total number of users
            num_items: Total number of items
            interactions: List of (user_id, item_id) tuples
            seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.num_items = num_items
        self.interactions = interactions

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build positive interaction sets for fast O(1) lookup
        self.user_pos_items: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        for u, i in interactions:
            if 0 <= u < num_users and 0 <= i < num_items:
                self.user_pos_items[u].add(i)

    def sample_negative_item_for_user(self, user: int) -> int:
        """Sample a single negative item not in the user's positive interactions."""
        pos_items = self.user_pos_items.get(user, set())
        if len(pos_items) >= self.num_items:
            # Degenerate edge case: user interacted with all items
            return random.randint(0, self.num_items - 1)

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in pos_items:
                return neg_item

    def sample_batch(
        self, batch_interactions: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples negative items for a given list of (user, positive_item) pairs.

        Returns:
            users: LongTensor of shape (B,)
            pos_items: LongTensor of shape (B,)
            neg_items: LongTensor of shape (B,)
        """
        users = []
        pos_items = []
        neg_items = []

        for u, i in batch_interactions:
            j = self.sample_negative_item_for_user(u)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )

    def sample_epoch_triplets(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples one negative item for every interaction in the training set.
        Generates full-epoch training triplets (u, i, j).
        """
        return self.sample_batch(self.interactions)
