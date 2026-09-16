import torch
import torch.nn as nn

from reccomendation_engine.models.layers import LightGCNLayer
from reccomendation_engine.models.matrix_create import n_adj

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = self.num_items + self.num_users
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embeddings = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        self.layers = nn.ModuleList(
            [LightGCNLayer() for _ in range(self.num_layers)]
        )

    def forward(self):
        E0_item = self.item_embeddings.weight
        E0_user = self.user_embeddings.weight        
        E0_combined = torch.cat((E0_user, E0_item), dim=0)

        embeddings_list = []
        embeddings_list.append(E0_combined)

        current_embeddings = E0_combined
        for layer in self.layers:
            current_embeddings = layer(current_embeddings, n_adj)
            embeddings_list.append(current_embeddings)


        embedding_sum = torch.zeros(self.total_nodes, self.embedding_dim)
        
        for i in embeddings_list:
            embedding_sum = embedding_sum + i

        E_final = embedding_sum/(self.num_layers+1)

        user_embeddings = E_final[:self.num_users]
        item_embeddings = E_final[self.num_users:]

        return user_embeddings, item_embeddings



#---------For Testing------------
# model = LightGCN(6, 10, 4, 3)
# u_emb, i_emb = model.forward()

# print("User Embeddings:")
# print(u_emb)
# print(u_emb.shape)

# print()

# print("Item Embedding:")
# print(i_emb)
# print(i_emb.shape)
