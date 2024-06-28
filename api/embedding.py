import os
from openai import OpenAI
import chromadb

class EmbeddingGenerator:
    def __init__(self, model_name="text-embedding-3-large", collection_name="video_data_medium", embedding_size=3072):
        self.model_name = model_name
        self.db_client = chromadb.PersistentClient(path="./chroma")
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        self.collection = self.db_client.get_or_create_collection(collection_name)
        self._embedding_client = None

    @property
    def embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._embedding_client

    def generate_embedding(self, text):
        response = self.embedding_client.embeddings.create(model=self.model_name, input=[text])
        embedding = response.data[0].embedding[:self.embedding_size]
        return embedding