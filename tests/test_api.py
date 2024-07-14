import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.main import app, verify_api_key, initialize_components
import os
import hashlib

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_api_key():
    test_api_key = "test_api_key"
    hashed_key = hashlib.sha256(test_api_key.encode()).hexdigest()
    with patch.dict(os.environ, {'API_KEY_HASH': hashed_key}):
        yield test_api_key

@pytest.fixture(autouse=True)
def mock_dependencies():
    mock_embedding_generator = MagicMock()
    mock_embedding_generator.generate_embedding.return_value = [0.1] * 3072

    mock_pinecone_search = MagicMock()
    mock_pinecone_search.find_nearest.return_value = []

    mock_llm_handler = MagicMock()
    mock_llm_handler.find_best_match.return_value = ('video1', '00:01:00', 'This is the best match')

    with patch('api.main.initialize_components') as mock_initialize_components:
        mock_initialize_components.return_value = (mock_embedding_generator, mock_pinecone_search, mock_llm_handler)
        yield

def test_search_endpoint_no_api_key():
    response = client.post("/search/", json={"text": "test query"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}

def test_search_endpoint_invalid_api_key():
    response = client.post(
        "/search/",
        json={"text": "test query"},
        headers={"X-API-Key": "invalid_api_key"}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Could not validate API key"}