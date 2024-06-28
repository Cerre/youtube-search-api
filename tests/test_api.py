import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, PropertyMock
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from api.main import app, initialize_components
from api.embedding import EmbeddingGenerator
from api.llm import LLMHandler

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test_api_key")

@pytest.fixture(autouse=True)
def mock_openai(mocker):
    mock_openai = MagicMock()
    mock_embedding_client = PropertyMock(return_value=mock_openai)
    mocker.patch.object(EmbeddingGenerator, 'embedding_client', mock_embedding_client)
    mock_llm_client = PropertyMock(return_value=mock_openai)
    mocker.patch.object(LLMHandler, 'client', mock_llm_client)
    return mock_openai

@pytest.fixture(autouse=True)
def mock_chromadb(mocker):
    mock_chromadb = MagicMock()
    mocker.patch("api.embedding.chromadb.PersistentClient", return_value=mock_chromadb)
    mock_chromadb.get_or_create_collection.return_value = MagicMock()
    return mock_chromadb

@pytest.fixture(autouse=True)
def mock_components(mocker):
    mock_embedding_generator = MagicMock(spec=EmbeddingGenerator)
    mock_chroma_search = MagicMock()
    mock_llm_handler = MagicMock(spec=LLMHandler)
    mocker.patch("api.main.initialize_components", return_value=(mock_embedding_generator, mock_chroma_search, mock_llm_handler))
    return mock_embedding_generator, mock_chroma_search, mock_llm_handler

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

def test_search_endpoint_authorized(mock_components):
    mock_embedding_generator, mock_chroma_search, mock_llm_handler = mock_components
    
    mock_embedding_generator.generate_embedding.return_value = [0.1, 0.2, 0.3]
    mock_chroma_search.find_nearest.return_value = {
        "documents": [["Test document"]],
        "metadatas": [{"video_id": "test_video", "start_time": "00:01:00"}]
    }
    mock_llm_handler.find_best_match.return_value = ("test_video", "00:01:00", "Test explanation")
    
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