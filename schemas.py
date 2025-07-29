# /rag_project/schemas.py

from pydantic import BaseModel, HttpUrl
from typing import List

class RAGRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RAGResponse(BaseModel):
    answers: List[str]

class Chunk(BaseModel):
    text: str
    page_number: int
    embedding: List[float]