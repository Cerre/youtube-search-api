# File: tests/test_search.py

import pytest
from api.search import ChromaDBSearch

def test_find_nearest(mocker):
    mock_collection = mocker.Mock()
    mock_collection.query.return_value = {
        "documents": [["Test document"]],
        "metadatas": [{"video_id": "test_video", "start_time": "00:01:00"}],
        "distances": [[0.5]]
    }
    
    search = ChromaDBSearch(mock_collection)
    results = search.find_nearest([0.1, 0.2, 0.3])
    
    assert "documents" in results
    assert "metadatas" in results
    assert "distances" in results
    mock_collection.query.assert_called_once()