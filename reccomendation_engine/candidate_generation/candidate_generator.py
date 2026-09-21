import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from chroma_client import ChromaDBClient
from database.inventory_service import get_available_items
from ranker import Ranker


db = ChromaDBClient()

user_embeddings=torch.load("../exported_embedding/user_embeddings.pt",map_location="cpu")

user_embeddings = (user_embeddings.cpu().detach().numpy())

user_embedding = user_embeddings[0]

results = db.query(user_embedding, top_k=10, filters = None)


candidates= results['ids'][0]

available_products= get_available_items(candidates)


candidate_ids = results["ids"][0]
distances = results["distances"][0]

filtered_candidates = []

for item_id, distance in zip(candidate_ids,distances):

    if item_id in available_products:
        product = available_products[item_id].copy()

        product["distance"] = distance

        filtered_candidates.append(product)

r = Ranker()

print("Final recommendations are: ", r.get_top_n(filtered_candidates,5))