import torch
import torch.nn as nn

class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, n_adj, embeddings):
        n_adj_tensor = torch.tensor(n_adj, dtype=torch.float32)
        propagated_embeddings = torch.matmul(n_adj_tensor, embeddings)

        return propagated_embeddings