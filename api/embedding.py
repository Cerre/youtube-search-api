import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model_name="text-embedding-3-large", index_name="video-data-medium"):
        self.embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)

    def generate_embedding(self, text):
        response = self.embedding_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding[:3072]  # Truncate to 3072 dimensions

    def add_to_index(self, id, embedding, metadata):
        self.index.upsert(vectors=[(id, embedding, metadata)])

    def search(self, query_embedding, top_k=10):
        return self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)