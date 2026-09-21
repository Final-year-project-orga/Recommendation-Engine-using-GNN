import chromadb
import numpy as np

from chromadb.config import Settings
from dotenv import load_dotenv
import os

load_dotenv()


class ChromaDBClient:

    def __init__(self):

        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )

        self.collection = self.client.get_or_create_collection(
            name="demo_products"
        )

    def add_item_embeddings(self,item_ids,item_embeddings,metadata):

        self.collection.add(
            ids=item_ids,
            embeddings=item_embeddings.tolist(),
            metadatas= metadata
        )

    def query(self,user_embedding,top_k, filters):

        return self.collection.query(
            query_embeddings=[
                user_embedding.tolist()
            ],
            n_results=top_k,
            where= filters
        )