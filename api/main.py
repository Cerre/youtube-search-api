from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
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

def format_timestamp(timestamp: str) -> int:
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError("Timestamp format is incorrect. Expected HH:MM:SS or MM:SS.")

def format_timestamp(timestamp: str) -> int:
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError("Timestamp format is incorrect. Expected HH:MM:SS or MM:SS.")

def format_url_timestamp(timestamp: str) -> str:
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
        seconds = format_timestamp(timestamp)
        url_timestamp = format_url_timestamp(timestamp)
        return {
            "video_id": video_id,
            "timestamp": seconds,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={url_timestamp}",
            "explanation": output_text
        }
    else:
        return {
            "message": "No specific match found",
            "explanation": output_text
        }

import time
import asyncio

@app.get("/version-check")
async def version_check():
    return {
        "version": "2.0",  # Update this with each deployment
        "timestamp": int(time.time()),
        "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>YouTube Search API</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
                h1 { color: #333; }
                .button { display: inline-block; padding: 10px 20px; margin: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Welcome to the YouTube Search API</h1>
            <p>This API allows you to search for YouTube videos based on text queries.</p>
            <p>For detailed API documentation and interactive testing, please visit:</p>
            <a href="/docs" class="button">Swagger UI</a>
            <a href="/redoc" class="button">ReDoc</a>
            <h2>Quick Start</h2>
            <p>To use the API, send a POST request to the <code>/search/</code> endpoint with your query.</p>
            <p>Example using curl:</p>
            <pre><code>curl -X POST "https://your-api-url.com/search/" 
     -H "Content-Type: application/json" 
     -H "X-API-Key: your_api_key" 
     -d '{"text": "Your search query here"}'
            </code></pre>
            <h2>Need Help?</h2>
            <p>If you need assistance or have any questions, please refer to the documentation or contact our support team.</p>
        </body>
    </html>
    """