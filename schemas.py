from pydantic import BaseModel, HttpUrl
from typing import List

class RAGRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    documents: HttpUrl
    questions: List[str]

class RAGResponse(BaseModel):
    """Defines the structure of the API response."""
    answers: List[str]

class Chunk(BaseModel):
    """
    Defines the data structure for a processed chunk of text.
    This is what gets stored in the cache.
    """
    text: str
    page_number: int
    embedding: List[float]