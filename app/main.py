from fastapi import FastAPI, Query
from schemas import SearchResponse
from search import search_songs
import logging

app = FastAPI()
logging.basicConfig(level=logging.info)

@app.get("/search", response_model=SearchResponse)
def search(user_query: str = Query(..., min_length=3)):
    logging.info(f"🔍 Search query received: {user_query}")
    results = search_songs(user_query)
    logging.info(f"✅ Top result: {results[0]['track_name'] if results else 'No match'}")
    return {"results": results}
