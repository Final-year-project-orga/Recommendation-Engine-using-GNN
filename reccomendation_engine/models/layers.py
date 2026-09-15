import torch
import torch.nn as nn


class LightGCNLayer(nn.Module):
    r"""
    Single message-passing layer for LightGCN.
    Computes: E^(k+1) = \tilde{A} * E^(k)
    where \tilde{A} is the symmetric normalized adjacency matrix: D^{-1/2} A D^{-1/2}
    Supports both dense and sparse PyTorch tensors.
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, n_adj: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_adj: Normalized adjacency matrix (N x N), dense Tensor or sparse_coo_tensor
            embeddings: Current node embeddings (N x embedding_dim)
        Returns:
            Propagated node embeddings (N x embedding_dim)
        """
        if not isinstance(n_adj, torch.Tensor):
            n_adj = torch.tensor(n_adj, dtype=embeddings.dtype, device=embeddings.device)
        else:
            n_adj = n_adj.to(device=embeddings.device, dtype=embeddings.dtype)

        if n_adj.is_sparse:
            if self.training and self.dropout > 0.0:
                indices = n_adj._indices()
                values = n_adj._values()
                mask = torch.rand(values.shape, device=values.device) >= self.dropout
                indices = indices[:, mask]
                values = values[mask] / (1.0 - self.dropout)
                n_adj_dropped = torch.sparse_coo_tensor(indices, values, n_adj.shape, device=values.device)
                propagated = torch.sparse.mm(n_adj_dropped, embeddings)
            else:
                propagated = torch.sparse.mm(n_adj, embeddings)
        else:
            if self.training and self.dropout > 0.0:
                embeddings = torch.nn.functional.dropout(embeddings, p=self.dropout, training=True)
            propagated = torch.matmul(n_adj, embeddings)

        return propagated