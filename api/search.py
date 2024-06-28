class PineconeSearch:
    def __init__(self, index):
        self.index = index

    def find_nearest(self, query_embedding, n_results=10):
        results = self.index.query(vector=query_embedding, top_k=n_results, include_metadata=True)
        matches = results.get('matches', [])
        return [
            {
                "id": match['id'],
                "score": match['score'],
                "metadata": match['metadata'],
                "text": match['metadata'].get('text', '')
            }
            for match in matches
        ]