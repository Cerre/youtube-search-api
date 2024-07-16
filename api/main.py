from fastapi import FastAPI, HTTPException, Depends, Request
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
async def add_security_headers_and_log_requests(request: Request, call_next):
    # Log request details
    start_time = time.time()
    logging.info(f"Request received: {request.method} {request.url}")
    logging.info(f"Client IP: {request.client.host}")

    # Process the request
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Log response details
    process_time = time.time() - start_time
    logging.info(f"Response status: {response.status_code}")
    logging.info(f"Process time: {process_time:.2f} seconds")

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

import time
import asyncio

@app.post("/search/", tags=["search"], dependencies=[Depends(verify_api_key)])
async def search(query: Query):
    try:
        start_time = time.time()
        embedding_generator, pinecone_search, llm_handler = initialize_components()
        
        logger.info(f"Starting embedding generation for query: {query.text}")
        query_embedding = await asyncio.to_thread(embedding_generator.generate_embedding, query.text)
        logger.info(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting Pinecone search")
        search_results = await asyncio.to_thread(pinecone_search.find_nearest, query_embedding)
        logger.info(f"Pinecone search completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting LLM processing")
        video_id, timestamp, output_text = await asyncio.to_thread(llm_handler.find_best_match, query.text, search_results)
        logger.info(f"LLM processing completed in {time.time() - start_time:.2f} seconds")
        
        result = process_search_result(video_id, timestamp, output_text)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")