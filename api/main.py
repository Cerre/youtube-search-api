from fastapi import FastAPI, HTTPException, Depends
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
pinecone_search = None
llm_handler = None
youtube_url_watch = "https://www.youtube.com/watch?v"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_HASH = os.getenv("API_KEY_HASH")

api_key_header = APIKeyHeader(name="X-API-Key")

class Query(BaseModel):
    text: str

app = FastAPI(
    title="YouTube Search API",
    description="API for finding best matches in YouTube transcripts using Pinecone",
    version="1.0.0",
    openapi_tags=[{"name": "search", "description": "Search operations"}],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    import hashlib
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    if not api_key or hashed_key != API_KEY_HASH:
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

@app.post("/search/", tags=["search"], dependencies=[Depends(verify_api_key)])
async def search(query: Query):
    try:
        embedding_generator, pinecone_search, llm_handler = initialize_components()
        query_embedding = embedding_generator.generate_embedding(query.text)
        search_results = pinecone_search.find_nearest(query_embedding)
        video_id, timestamp, output_text = llm_handler.find_best_match(query.text, search_results)
        return process_search_result(video_id, timestamp, output_text)
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")