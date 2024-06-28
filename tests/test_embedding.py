import pytest
from unittest.mock import MagicMock, patch
from api.embedding import EmbeddingGenerator

@pytest.fixture
def mock_openai():
    with patch('api.embedding.OpenAI') as mock:
        mock_instance = MagicMock()
        mock_instance.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 3072)]
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_pinecone():
    with patch('api.embedding.Pinecone') as mock:
        mock_instance = MagicMock()
        mock_instance.Index.return_value = MagicMock()
        mock.return_value = mock_instance
        yield mock

def test_embedding_generator_initialization(mock_openai, mock_pinecone):
    generator = EmbeddingGenerator()
    assert generator.model_name == "text-embedding-3-large"
    assert mock_openai.called
    assert mock_pinecone.called

def test_generate_embedding(mock_openai, mock_pinecone):
    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding("test text")
    assert len(embedding) == 3072
    mock_openai.return_value.embeddings.create.assert_called_once_with(
        input=["test text"], 
        model="text-embedding-3-large"
    )

def test_add_to_index(mock_openai, mock_pinecone):
    generator = EmbeddingGenerator()
    generator.add_to_index("test_id", [0.1] * 3072, {"key": "value"})
    generator.index.upsert.assert_called_once_with(
        vectors=[("test_id", [0.1] * 3072, {"key": "value"})]
    )

def test_search(mock_openai, mock_pinecone):
    generator = EmbeddingGenerator()
    generator.search([0.1] * 3072, top_k=5)
    generator.index.query.assert_called_once_with(
        vector=[0.1] * 3072, 
        top_k=5, 
        include_metadata=True
    )