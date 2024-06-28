import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.main import app, verify_api_key

client = TestClient(app)

@pytest.fixture
def mock_pinecone():
    with patch('api.embedding.Pinecone') as mock:
        mock_instance = MagicMock()
        mock_instance.Index.return_value = MagicMock()
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_openai():
    with patch('api.embedding.OpenAI') as mock:
        mock_instance = MagicMock()
        mock_instance.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 3072)]
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_llm_handler():
    with patch('api.main.LLMHandler') as mock:
        mock_instance = MagicMock()
        mock_instance.find_best_match.return_value = ('video1', '00:01:00', 'This is the best match')
        mock.return_value = mock_instance
        yield mock

@pytest.fixture(autouse=True)
def mock_api_key():
    with patch.dict('os.environ', {'API_KEY': 'test_api_key'}):
        yield

def test_search_endpoint_no_api_key():
    response = client.post("/search/", json={"text": "test query"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Could not validate API key"}

def test_search_endpoint_invalid_api_key():
    response = client.post(
        "/search/",
        json={"text": "test query"},
        headers={"X-API-Key": "invalid_api_key"}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Could not validate API key"}

def test_search_endpoint_authorized(mock_pinecone, mock_openai, mock_llm_handler):
    response = client.post(
        "/search/",
        json={"text": "test query"},
        headers={"X-API-Key": "test_api_key"}
    )
    
    assert response.status_code == 200
    assert "video_id" in response.json()
    assert "timestamp" in response.json()
    assert "url_with_timestamp" in response.json()
    assert "explanation" in response.json()
    breakpoint()

    assert response.json()["video_id"] == "video1"
    assert response.json()["timestamp"] == "00:01:00"
    assert "This is the best match" in response.json()["explanation"]