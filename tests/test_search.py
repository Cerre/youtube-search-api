import pytest
from unittest.mock import MagicMock
from api.search import PineconeSearch

@pytest.fixture
def mock_pinecone_index():
    return MagicMock()

def test_find_nearest(mock_pinecone_index):
    search = PineconeSearch(mock_pinecone_index)
    mock_pinecone_index.query.return_value = {
        'matches': [
            {'id': '1', 'score': 0.9, 'metadata': {'text': 'test1', 'key': 'value1'}},
            {'id': '2', 'score': 0.8, 'metadata': {'text': 'test2', 'key': 'value2'}}
        ]
    }
    
    result = search.find_nearest([0.1] * 3072, n_results=2)
    
    
    assert 'id' in result
    assert 'score' in result
    assert 'metadata' in result
    assert 'text' in result
    assert len(result) == 2
    assert result['ids'] == ['1', '2']
    assert result['documents'] == ['test1', 'test2']
    assert result['metadatas'] == [{'text': 'test1', 'key': 'value1'}, {'text': 'test2', 'key': 'value2'}]
    assert result['distances'] == pytest.approx([0.1, 0.2], rel=1e-9)  # 1 - score

    mock_pinecone_index.query.assert_called_once_with(vector=[0.1] * 3072, top_k=2, include_metadata=True)