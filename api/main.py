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
        pinecone_index_name = "johnniboi-text-embedding-3-large"
        embedding_generator = EmbeddingGenerator(model, pinecone_index_name)
        pinecone_search = PineconeSearch(embedding_generator.index)
        llm_handler = LLMHandler(model)
    return embedding_generator, pinecone_search, llm_handler


def process_search_result(video_id, timestamp, explanation, text):
    if video_id and timestamp:
        seconds = int(float(timestamp))
        # url_timestamp = format_timestamp(timestamp)
        return {
            "video_id": video_id,
            "timestamp": seconds,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={seconds}s",
            "text": text,
            "explanation": explanation
        }
    else:
        return {
            "message": "No specific match found",
            "explanation": explanation
        }

import time
import asyncio

def process_multiple_search_results(matched_results):
    processed_results = []
    for result, explanation in matched_results:
        video_id = result['metadata'].get('id')
        timestamp = result['metadata'].get('start_time', '0')
        text = result['metadata'].get('text', '')
        seconds = int(float(timestamp))
        processed_results.append({
            "video_id": video_id,
            "timestamp": seconds,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={seconds}s",
            "text": text,
            "score": result['score'],
            "explanation": explanation
        })
    return {"results": processed_results}


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

@app.post("/search/", tags=["search"], dependencies=[Depends(verify_api_key)])
async def search(query: Query):
    try:
        start_time = time.time()
        embedding_generator, pinecone_search, llm_handler = initialize_components()
        
        logger.info(f"Starting embedding generation for query: {query.text}")
        query_embedding = await asyncio.to_thread(embedding_generator.generate_embedding, query.text)
        logger.info(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting Pinecone search")
        search_results = await asyncio.to_thread(pinecone_search.find_nearest, query_embedding, n_results=5)
        logger.info(f"Pinecone search completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting LLM processing")
        video_id, timestamp, explanation, text = await asyncio.to_thread(llm_handler.find_best_match, query.text, search_results)
        logger.info(f"LLM processing completed in {time.time() - start_time:.2f} seconds")
        
        result = process_search_result(video_id, timestamp, explanation, text)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/search_multiple/", tags=["search"], dependencies=[Depends(verify_api_key)])
async def search_multiple(query: Query):
    try:
        start_time = time.time()
        embedding_generator, pinecone_search, llm_handler = initialize_components()
        
        logger.info(f"Starting embedding generation for query: {query.text}")
        query_embedding = await asyncio.to_thread(embedding_generator.generate_embedding, query.text)
        logger.info(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting Pinecone search")
        search_results = await asyncio.to_thread(pinecone_search.find_nearest, query_embedding, n_results=10)  # Fetch more results for LLM to choose from
        logger.info(f"Pinecone search completed in {time.time() - start_time:.2f} seconds")
        
        logger.info("Starting LLM processing")
        best_matches = await asyncio.to_thread(llm_handler.find_best_matches, query.text, search_results)
        logger.info(f"LLM processing completed in {time.time() - start_time:.2f} seconds")
        
        results = process_multiple_search_results(best_matches)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return results
    except Exception as e:
        logger.error(f"An error occurred during multiple search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")