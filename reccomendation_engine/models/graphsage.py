from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    r"""
    GraphSAGE Layer for recommendation graph.
    Propagates and aggregates neighbor representations:
    h_N(v) = Mean({h_u, \forall u \in N(v)})
    h_v^(k+1) = \sigma(W \cdot [h_v^(k) || h_N(v)])
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, act: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act

        # Linear projection combining self-representation and neighbor representation
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj_or_norm: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj_or_norm: Row-normalized adjacency matrix (N x N) or normalized adj
            embeddings: Current node embeddings (N x in_dim)
        Returns:
            Updated node embeddings (N x out_dim)
        """
        if not isinstance(adj_or_norm, torch.Tensor):
            adj_or_norm = torch.tensor(adj_or_norm, dtype=embeddings.dtype, device=embeddings.device)
        else:
            adj_or_norm = adj_or_norm.to(device=embeddings.device, dtype=embeddings.dtype)

        # Aggregate neighbors
        if adj_or_norm.is_sparse:
            neighbor_feats = torch.sparse.mm(adj_or_norm, embeddings)
        else:
            neighbor_feats = torch.matmul(adj_or_norm, embeddings)

        # Concatenate self and neighbor representations
        combined = torch.cat([embeddings, neighbor_feats], dim=-1)

        if self.training and self.dropout > 0.0:
            combined = F.dropout(combined, p=self.dropout, training=True)

        out = self.linear(combined)
        if self.act:
            out = F.relu(out)

        # Normalize to unit sphere (common in GraphSAGE)
        out = F.normalize(out, p=2, dim=-1)
        return out


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for learning user and item representations.
    Supports initial ID embeddings or optional external node features.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Node ID embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            next_dim = hidden_dim if not is_last else embedding_dim
            self.layers.append(
                GraphSAGELayer(
                    in_dim=current_dim,
                    out_dim=next_dim,
                    dropout=dropout,
                    act=not is_last,
                )
            )
            current_dim = next_dim

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_ego_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(self, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.get_ego_embeddings()
        for layer in self.layers:
            x = layer(adj_norm, x)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        return user_emb, item_emb

    def predict(self, users: torch.Tensor, items: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(adj_norm)
        u_feat = user_emb[users]
        i_feat = item_emb[items]
        return torch.sum(u_feat * i_feat, dim=-1)
