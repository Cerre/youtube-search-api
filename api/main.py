from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from .embedding import EmbeddingGenerator
from .llm import LLMHandler
from .search import PineconeSearch

load_dotenv()

# Global variables
embedding_generator = None
chroma_search = None
llm_handler = None
youtube_url_watch = "https://www.youtube.com/watch?v"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key setup
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class Query(BaseModel):
    text: str

def create_app():
    return FastAPI(
        title="YouTube Search API",
        description="API for finding best matches in YouTube transcripts using ChromaDB",
        version="1.0.0",
        openapi_tags=[{"name": "search", "description": "Search operations"}],
    )

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key}")
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

def initialize_components():
    global embedding_generator, pinecone_search, llm_handler
    if embedding_generator is None:
        model = "text-embedding-3-large"
        embedding_generator = EmbeddingGenerator(model)
        pinecone_search = PineconeSearch(embedding_generator.index)
        llm_handler = LLMHandler(model)
    return embedding_generator, pinecone_search, llm_handler

def format_timestamp(timestamp: str) -> str:
    parts = timestamp.split(":")
    formatted_parts = []
    if len(parts) == 3:
        if int(parts[0]) > 0:
            formatted_parts.append(parts[0] + "h")
        formatted_parts.append(parts[1] + "m")
    elif len(parts) == 2:
        formatted_parts.append(parts[0] + "m")
    else:
        raise ValueError("Timestamp format is incorrect. Expected HH:MM:SS or MM:SS.")
    formatted_parts.append(parts[-1] + "s")
    return "".join(formatted_parts)

def process_search_result(video_id, timestamp, output_text):
    if video_id and timestamp:
        timestamp_formatted = format_timestamp(timestamp)
        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={timestamp_formatted}",
            "explanation": output_text
        }
    else:
        return {
            "message": "No specific match found",
            "explanation": output_text
        }

app = create_app()

_embedding_generator, _pinecone_search, _llm_handler = initialize_components()
@app.post("/search/", tags=["search"])
async def search(query: Query, api_key: str = Depends(verify_api_key)):
    try:
        query_embedding = _embedding_generator.generate_embedding(query.text)
        search_results = _pinecone_search.find_nearest(query_embedding)
        video_id, timestamp, output_text = _llm_handler.find_best_match(query.text, search_results)
        return process_search_result(video_id, timestamp, output_text)
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def run_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()