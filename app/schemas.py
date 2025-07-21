from pydantic import BaseModel
from typing import List

class SongResult(BaseModel):
    track_name: str
    artist_name: str
    genre: str
    valence: str
    energy: str
    lyrics_snippet: str

class SearchResponse(BaseModel):
    results: List[SongResult]