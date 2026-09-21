from reccomendation_engine.convertor import num_items, num_users, total_nodes
from reccomendation_engine.models.lightgcn import LightGCN

from reccomendation_engine.training.negative_sampling import generate_negative_samples
from reccomendation_engine.training.losses import BPR_loss

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Important variables
epochs = 300
learning_rate = 0.01
embedding_dim = 4
num_layers = 3


# Function for separate user, positive and negative embeddings
def seperate_embedding(batched_triplets, user_embeddings, item_embeddings):

    user_ids = batched_triplets[0]
    positive_item_ids = batched_triplets[1]
    negative_item_ids = batched_triplets[2]

    user_embeddings_tensor = user_embeddings[user_ids]
    pos_item_embeddings_tensor = item_embeddings[positive_item_ids]
    neg_item_embeddings_tensor = item_embeddings[negative_item_ids]

    return user_embeddings_tensor, pos_item_embeddings_tensor, neg_item_embeddings_tensor


# Dataset class
class CustomDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]


generated_triplets = generate_negative_samples()
dataset = CustomDataset(generated_triplets)

dataloader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True
)


# LightGCN instance
model = LightGCN(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=embedding_dim,
    num_layers=num_layers
)


# Adam Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# Training
for epoch in range(epochs):
    total_loss_per_epoch = 0

    for batched_triplets in dataloader:

        optimizer.zero_grad()

        user_embeddings, item_embeddings = model()

        sep_user_emb, sep_pos_item_emb, sep_neg_item_emb = seperate_embedding(
            batched_triplets,
            user_embeddings,
            item_embeddings
        )

        loss = BPR_loss(
            sep_user_emb,
            sep_pos_item_emb,
            sep_neg_item_emb
        )

        loss.backward()
        optimizer.step()

        total_loss_per_epoch += loss.item()

    mean_loss_per_epoch = total_loss_per_epoch / len(dataloader)

    print(f"epoch: {epoch + 1} --- loss: {mean_loss_per_epoch:.6f}")


# Exporting the final embeddings

model.eval()

with torch.no_grad():
    user_embeddings, item_embeddings = model()


# Save the embeddings
torch.save(user_embeddings, "reccomendation_engine/exported_embedding/user_embeddings.pt")
torch.save(item_embeddings, "reccomendation_engine/exported_embedding/item_embeddings.pt")

print("\nTraining complete.")
print("User embeddings shape:", user_embeddings.shape)
print("Item embeddings shape:", item_embeddings.shape)
print("User embeddings saved to: user_embeddings.pt")
print("Item embeddings saved to: item_embeddings.pt")