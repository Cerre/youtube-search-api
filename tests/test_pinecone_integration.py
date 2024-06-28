import pytest
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from api.embedding import EmbeddingGenerator

# Load environment variables
load_dotenv()

@pytest.fixture(scope="module")
def pinecone_client():
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

@pytest.fixture(scope="module")
def embedding_generator():
    return EmbeddingGenerator()

def test_pinecone_connection(pinecone_client):
    indexes = pinecone_client.list_indexes()
    assert len(indexes) > 0, "No indexes found in Pinecone"

def test_index_exists(pinecone_client):
    index_name = "video-data-medium"
    indexes = pinecone_client.list_indexes()
    index_names = [index['name'] for index in indexes]
    assert index_name in index_names, f"Index '{index_name}' not found. Available indexes: {index_names}"

def test_index_stats(pinecone_client):
    index = pinecone_client.Index("video-data-medium")
    stats = index.describe_index_stats()
    assert stats['total_vector_count'] > 0, "Index is empty"
    assert stats['dimension'] == 3072, "Incorrect embedding dimension"

def test_query_structure(embedding_generator):
    query_text = "competitive player achievements"
    query_embedding = embedding_generator.generate_embedding(query_text)
    results = embedding_generator.search(query_embedding, top_k=1)
    
    assert 'matches' in results, "Query result doesn't contain 'matches'"
    assert len(results['matches']) > 0, "No matches found"
    
    match = results['matches'][0]
    assert 'id' in match, "Match doesn't contain 'id'"
    assert 'score' in match, "Match doesn't contain 'score'"
    assert 'metadata' in match, "Match doesn't contain 'metadata'"

def test_metadata_fields(embedding_generator):
    query_text = "competitive player achievements"
    query_embedding = embedding_generator.generate_embedding(query_text)
    results = embedding_generator.search(query_embedding, top_k=1)
    
    metadata = results['matches'][0]['metadata']
    expected_fields = ['author', 'channel_id', 'channel_url', 'publish_date', 
                       'start_time', 'text', 'thumbnail_url', 'title', 'video_id']
    
    for field in expected_fields:
        assert field in metadata, f"Metadata is missing the '{field}' field"

def test_search_relevance(embedding_generator):
    query_text = "The worst we've seen in 1v1 keeping the ball up history"
    query_embedding = embedding_generator.generate_embedding(query_text)
    results = embedding_generator.search(query_embedding, top_k=5)
    
    for match in results['matches']:
        assert 'ball' in match['metadata']['text'].lower(), \
               "Search results don't seem relevant to the query"

if __name__ == "__main__":
    pytest.main()