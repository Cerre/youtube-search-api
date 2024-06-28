class ChromaDBSearch:
    def __init__(self, collection):
        self.collection = collection

    def find_nearest(self, query_embedding, n_results=10):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results