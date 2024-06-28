# File: tests/test_embedding.py

import pytest
from api.embedding import EmbeddingGenerator

def test_embedding_generator_initialization():
    generator = EmbeddingGenerator()
    assert generator.model_name == "text-embedding-3-large"
    assert generator.collection_name == "video_data_medium"
    assert generator.embedding_size == 3072

@pytest.mark.asyncio
async def test_generate_embedding(mocker):
    mock_openai = mocker.patch("api.embedding.OpenAI")
    mock_openai.return_value.embeddings.create.return_value.data = [
        type('obj', (object,), {'embedding': [0.1] * 3072})()
    ]
    
    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding("test text")
    
    assert len(embedding) == 3072
    mock_openai.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-large", 
        input=["test text"]
    )