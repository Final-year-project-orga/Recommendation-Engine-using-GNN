from chroma_client import ChromaDBClient
import numpy as np
import json
import torch

db = ChromaDBClient()

item_ids = np.load("../data/demo_item_ids.npy",allow_pickle=True).tolist()

item_embeddings=torch.load("../exported_embedding/item_embeddings.pt", map_location="cpu")

item_embeddings = (item_embeddings.cpu().detach().numpy())

with open("../data/products.json","r") as f:
    products = json.load(f)

metadata = []

for product in products:
    metadata.append(
        {
            "category":
                product["category"],

            "brand":
                product["brand"]
        }
    )


db.add_item_embeddings(item_ids,item_embeddings,metadata)

print("Embeddings uploaded successfully!")
