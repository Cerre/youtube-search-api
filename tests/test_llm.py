import pytest
from unittest.mock import MagicMock, patch
from api.llm import LLMHandler

@pytest.fixture
def mock_openai():
    with patch('api.llm.OpenAI') as mock:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='Best match: 1\nBrief explanation: This is the best match.'))
        ]
        mock.return_value = mock_instance
        yield mock

def test_find_best_match(mock_openai):
    handler = LLMHandler("gpt-4")
    search_results = [
        {
            'id': '1',
            'score': 0.9,
            'metadata': {'video_id': 'video1', 'start_time': '00:01:00'},
            'text': 'This is document 1'
        },
        {
            'id': '2',
            'score': 0.8,
            'metadata': {'video_id': 'video2', 'start_time': '00:02:00'},
            'text': 'This is document 2'
        }
    ]
    
    video_id, timestamp, explanation = handler.find_best_match("test query", search_results)
    
    assert video_id == 'video1'
    assert timestamp == '00:01:00'
    assert explanation == "This is the best match."

    mock_openai.return_value.chat.completions.create.assert_called_once()

def test_find_best_match_no_clear_match(mock_openai):
    handler = LLMHandler("gpt-4")
    mock_openai.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content='Best match: 3\nBrief explanation: No clear best match.'))
    ]
    
    search_results = [
        {
            'id': '1',
            'score': 0.9,
            'metadata': {'video_id': 'video1', 'start_time': '00:01:00'},
            'text': 'This is document 1'
        },
        {
            'id': '2',
            'score': 0.8,
            'metadata': {'video_id': 'video2', 'start_time': '00:02:00'},
            'text': 'This is document 2'
        }
    ]
    
    video_id, timestamp, explanation = handler.find_best_match("test query", search_results)
    
    assert video_id is None
    assert timestamp is None
    assert explanation == "No clear best match found."