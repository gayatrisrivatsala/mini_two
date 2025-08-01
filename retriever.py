# /retriever.py

import httpx
import numpy as np
from typing import List
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from schemas import Chunk
from config import settings
from llm_handler import generate_hypothetical_answer
import logging

logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
RERANKER_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

async def embed_text(text: str) -> np.ndarray:
    """Embeds a single string."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            EMBEDDING_API_URL, headers=HUGGINGFACE_HEADERS,
            json={"inputs": [text], "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return np.array(response.json())

async def rerank_results(question: str, chunks: List[Chunk]) -> List[Chunk]:
    """Re-ranks chunks using the BGE Re-ranker."""
    pairs = [[question, chunk.text] for chunk in chunks]
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            RERANKER_API_URL, headers=HUGGINGFACE_HEADERS,
            json={"inputs": pairs, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        scores = response.json()
    scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0]['score'], reverse=True)
    return [chunk for score, chunk in scored_chunks]

async def find_relevant_chunks(question: str, all_chunks: List[Chunk], semaphore: asyncio.Semaphore, top_k: int = 5) -> List[Chunk]:
    """Finds relevant chunks using rate-limited HyDE followed by re-ranking."""
    if not all_chunks:
        return []

    logger.info("Generating hypothetical answer for HyDE...")
    hypothetical_answer = await generate_hypothetical_answer(question, semaphore)
    
    search_embedding = await embed_text(hypothetical_answer)
    chunk_embeddings = np.array([chunk.embedding for chunk in all_chunks])
    
    similarities = cosine_similarity(search_embedding, chunk_embeddings)[0]
    
    candidate_k = min(len(all_chunks), top_k * 10)
    candidate_indices = np.argsort(similarities)[::-1][:candidate_k]
    candidate_chunks = [all_chunks[i] for i in candidate_indices]
    
    if not candidate_chunks:
        return []

    logger.info(f"Re-ranking {len(candidate_chunks)} candidates...")
    try:
        reranked_chunks = await rerank_results(question, candidate_chunks)
    except Exception as e:
        logger.error(f"Re-ranking API failed: {e}. Falling back to similarity search.")
        return candidate_chunks[:top_k]

    return reranked_chunks[:top_k]