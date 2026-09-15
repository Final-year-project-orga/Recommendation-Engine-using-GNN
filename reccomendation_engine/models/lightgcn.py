from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from reccomendation_engine.models.layers import LightGCNLayer


class LightGCN(nn.Module):
    """
    imp points:
    - Strips out non-linear activations and feature transformation matrices.
    - Linearly propagates user and item embeddings on the bipartite graph.
    - Combines layer-wise representations via weighted sum.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        alpha: Optional[Union[float, List[float]]] = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding tables
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        # Layer combination coefficients (\alpha_k)
        if alpha is None:
            # Default: uniform weighting 1 / (K + 1)
            self.alpha = [1.0 / (self.num_layers + 1)] * (self.num_layers + 1)
        elif isinstance(alpha, (int, float)):
            self.alpha = [float(alpha)] * (self.num_layers + 1)
        else:
            assert len(alpha) == self.num_layers + 1, "Length of alpha must equal num_layers + 1"
            self.alpha = alpha

        # Graph propagation layers
        self.layers = nn.ModuleList([LightGCNLayer(dropout=dropout) for _ in range(self.num_layers)])

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings with normal distribution as per LightGCN paper.
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def get_ego_embeddings(self) -> torch.Tensor:

        # Concatenates user and item initial embeddings (Layer 0).
        # Shape: (num_users + num_items, embedding_dim)

        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(
        self, n_adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Multi-layer propagation over the graph.

        Args:
            n_adj: Normalized adjacency matrix (dense or sparse) of shape (total_nodes, total_nodes)

        Returns:
            final_user_embeddings: (num_users, embedding_dim)
            final_item_embeddings: (num_items, embedding_dim)
            layer_embeddings: list of embeddings at each layer [E^(0), E^(1), ..., E^(K)]
        """
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = [ego_embeddings]

        current_embeddings = ego_embeddings
        for layer in self.layers:
            current_embeddings = layer(n_adj, current_embeddings)
            all_embeddings.append(current_embeddings)

        # Weighted combination across layers: E = \sum \alpha_k E^(k)
        final_embeddings = torch.zeros_like(ego_embeddings)
        for k, emb in enumerate(all_embeddings):
            final_embeddings = final_embeddings + self.alpha[k] * emb

        final_user_embeddings = final_embeddings[:self.num_users]
        final_item_embeddings = final_embeddings[self.num_users:]

        return final_user_embeddings, final_item_embeddings, all_embeddings

    def get_user_embeddings(self, n_adj: torch.Tensor) -> torch.Tensor:
        """Get final user embeddings after propagation."""
        users_emb, _, _ = self.forward(n_adj)
        return users_emb

    def get_item_embeddings(self, n_adj: torch.Tensor) -> torch.Tensor:
        """Get final item embeddings after propagation."""
        _, items_emb, _ = self.forward(n_adj)
        return items_emb

    def get_all_embeddings(self, n_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method returning (final_user_embeddings, final_item_embeddings)."""
        users_emb, items_emb, _ = self.forward(n_adj)
        return users_emb, items_emb

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        n_adj: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute predicted recommendation scores: score(u, i) = e_u \cdot e_i

        Args:
            users: 1D tensor of user IDs
            items: 1D tensor of item IDs
            n_adj: Normalized adjacency matrix

        Returns:
            scores: 1D tensor of dot product scores
        """
        users_emb, items_emb, _ = self.forward(n_adj)
        u_emb = users_emb[users]
        i_emb = items_emb[items]
        scores = torch.sum(u_emb * i_emb, dim=-1)
        return scores
