import httpx
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# Corrected Imports: Removed the leading dots.
from schemas import Chunk
from config import settings
import logging

logger = logging.getLogger(__name__)

# Correct, direct model endpoints for embedding and re-ranking
EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
RERANKER_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

async def embed_question(question: str) -> np.ndarray:
    """Embeds a single question via Hugging Face API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            EMBEDDING_API_URL, headers=HUGGINGFACE_HEADERS,
            json={"inputs": [question], "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return np.array(response.json())

async def rerank_results(question: str, chunks: List[Chunk]) -> List[Chunk]:
    """Re-ranks a list of chunks using the BGE Re-ranker API for maximum accuracy."""
    pairs = [[question, chunk.text] for chunk in chunks]
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            RERANKER_API_URL, headers=HUGGINGFACE_HEADERS,
            json={"inputs": pairs, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        scores = response.json()

    # Pair scores with the original chunks and sort by the new relevance score
    scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0]['score'], reverse=True)
    
    # Return just the chunks, now in the correct order of relevance
    return [chunk for score, chunk in scored_chunks]

async def find_relevant_chunks(question: str, all_chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
    """
    Finds relevant chunks using a two-stage retrieve-and-rerank strategy.
    """
    if not all_chunks:
        return []

    # Stage 1: Fast Retrieval (initial candidate search)
    question_embedding = await embed_question(question)
    chunk_embeddings = np.array([chunk.embedding for chunk in all_chunks])
    
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    
    # Retrieve a larger pool of candidates for the re-ranker to process
    candidate_k = min(len(all_chunks), top_k * 5)
    candidate_indices = np.argsort(similarities)[::-1][:candidate_k]
    candidate_chunks = [all_chunks[i] for i in candidate_indices]

    # Stage 2: High-Accuracy Re-ranking
    try:
        logger.info(f"Re-ranking {len(candidate_chunks)} candidates for top {top_k} results...")
        reranked_chunks = await rerank_results(question, candidate_chunks)
        return reranked_chunks[:top_k]
    except Exception as e:
        logger.error(f"Re-ranking API failed: {e}. Falling back to basic similarity search.")
        return candidate_chunks[:top_k]