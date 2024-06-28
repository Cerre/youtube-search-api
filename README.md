# YouTube Search API

This API allows searching through YouTube video transcripts using vector embeddings.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `API_KEY`: Your custom API key for access control
   - `OPENAI_API_KEY`: Your OpenAI API key for generating embeddings
   - Add any other necessary keys (e.g., for your vector database)
4. Run the API: `uvicorn api.main:app --reload`

## Usage

Send a POST request to `/search/` with a JSON body:

```json
{
  "text": "Your search query here"
}
