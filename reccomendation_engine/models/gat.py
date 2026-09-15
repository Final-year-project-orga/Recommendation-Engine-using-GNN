from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer.
    Computes self-attention over graph neighbors.
    Supports adjacency matrices (dense or sparse).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
        alpha_leaky: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha_leaky = alpha_leaky
        self.concat = concat

        self.linear = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha_leaky)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, adj: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: Adjacency matrix (N x N) (binary or connection weights)
            embeddings: Node embeddings (N x in_dim)
        Returns:
            Output embeddings (N x (out_dim * num_heads)) or (N x out_dim)
        """
        num_nodes = embeddings.size(0)
        h = self.linear(embeddings).view(num_nodes, self.num_heads, self.out_dim)

        # Attention scores: a_src \cdot h_i + a_dst \cdot h_j
        alpha_src = (h * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # (N, heads)

        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32, device=embeddings.device)
        else:
            adj = adj.to(device=embeddings.device, dtype=torch.float32)

        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj

        # Mask non-edges: add self-loops so a node can attend to itself
        adj_with_self = adj_dense + torch.eye(num_nodes, device=adj_dense.device)
        mask = adj_with_self > 0  # (N, N)

        out_heads = []
        for head in range(self.num_heads):
            # Compute pairwise attention logits: e_ij = LeakyReLU(alpha_src[i] + alpha_dst[j])
            logits = self.leaky_relu(alpha_src[:, head].unsqueeze(1) + alpha_dst[:, head].unsqueeze(0))
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))

            attn = F.softmax(logits, dim=1)
            if self.training and self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout, training=True)

            h_head = h[:, head, :]  # (N, out_dim)
            out_head = torch.matmul(attn, h_head)  # (N, out_dim)
            out_heads.append(out_head)

        if self.concat:
            out = torch.cat(out_heads, dim=-1)
        else:
            out = torch.mean(torch.stack(out_heads, dim=0), dim=0)

        return out


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network for Recommendation.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if is_last:
                self.layers.append(
                    GATLayer(
                        in_dim=current_dim,
                        out_dim=embedding_dim,
                        num_heads=1,
                        dropout=dropout,
                        concat=False,
                    )
                )
            else:
                head_dim = embedding_dim // num_heads
                self.layers.append(
                    GATLayer(
                        in_dim=current_dim,
                        out_dim=head_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                current_dim = head_dim * num_heads

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_ego_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.get_ego_embeddings()
        for layer in self.layers:
            x = layer(adj, x)
            x = F.elu(x)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        return user_emb, item_emb

    def predict(self, users: torch.Tensor, items: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(adj)
        u_feat = user_emb[users]
        i_feat = item_emb[items]
        return torch.sum(u_feat * i_feat, dim=-1)
