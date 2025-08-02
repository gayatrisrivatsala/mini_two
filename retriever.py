# /retriever.py - OPTIMIZED VERSION

import httpx
import numpy as np
from typing import List, Dict, Tuple
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter

from schemas import Chunk
from config import settings
from llm_handler import generate_hypothetical_answer
import logging

logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
RERANKER_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-reranker-large"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

class AdvancedRetriever:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms and entities from query."""
        # Extract potential key terms
        key_terms = []
        
        # Numbers (often important in insurance)
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        key_terms.extend(numbers)
        
        # Monetary amounts
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        key_terms.extend(money)
        
        # Percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        key_terms.extend(percentages)
        
        # Important insurance terms
        insurance_terms = [
            'premium', 'deductible', 'coverage', 'policy', 'claim', 'benefit',
            'exclusion', 'liability', 'copay', 'coinsurance', 'maximum', 'minimum',
            'annual', 'monthly', 'yearly', 'limit', 'amount', 'fee', 'cost'
        ]
        
        text_lower = text.lower()
        for term in insurance_terms:
            if term in text_lower:
                key_terms.append(term)
        
        return list(set(key_terms))
    
    def keyword_filter(self, question: str, chunks: List[Chunk]) -> List[Chunk]:
        """Filter chunks based on keyword matching."""
        key_terms = self.extract_key_terms(question)
        
        if not key_terms:
            return chunks
        
        scored_chunks = []
        question_lower = question.lower()
        
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            score = 0
            
            # Exact keyword matches
            for term in key_terms:
                if term.lower() in chunk_lower:
                    score += 2
            
            # Question word overlap
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            overlap = len(question_words.intersection(chunk_words))
            score += overlap * 0.1
            
            scored_chunks.append((score, chunk))
        
        # Sort by score and return chunks with score > 0
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        filtered_chunks = [chunk for score, chunk in scored_chunks if score > 0]
        
        return filtered_chunks if filtered_chunks else chunks
    
    def setup_tfidf(self, chunks: List[Chunk]):
        """Setup TF-IDF vectorizer for keyword-based retrieval."""
        try:
            texts = [chunk.text for chunk in chunks]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("TF-IDF vectorizer setup completed")
        except Exception as e:
            logger.warning(f"TF-IDF setup failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

async def embed_text(text: str) -> np.ndarray:
    """Embeds a single string with retry logic."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    EMBEDDING_API_URL, 
                    headers=HUGGINGFACE_HEADERS,
                    json={"inputs": [text], "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                return np.array(response.json())
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to embed text after {max_retries} attempts: {e}")
                raise
            await asyncio.sleep(2 ** attempt)

async def rerank_results(question: str, chunks: List[Chunk], max_chunks: int = 50) -> List[Chunk]:
    """Enhanced re-ranking with better error handling."""
    if not chunks:
        return chunks
    
    # Limit the number of chunks to rerank for API efficiency
    chunks_to_rerank = chunks[:max_chunks]
    pairs = [[question, chunk.text] for chunk in chunks_to_rerank]
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    RERANKER_API_URL, 
                    headers=HUGGINGFACE_HEADERS,
                    json={"inputs": pairs, "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                scores = response.json()
                
            scored_chunks = list(zip(scores, chunks_to_rerank))
            scored_chunks.sort(key=lambda x: x[0]['score'], reverse=True)
            
            reranked = [chunk for score, chunk in scored_chunks]
            
            # Add any remaining chunks that weren't reranked
            if len(chunks) > max_chunks:
                reranked.extend(chunks[max_chunks:])
            
            return reranked
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Re-ranking failed after {max_retries} attempts: {e}")
                return chunks
            logger.warning(f"Re-ranking attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(2 ** attempt)
    
    return chunks

def hybrid_search(question: str, chunks: List[Chunk], retriever: AdvancedRetriever, 
                 semantic_weight: float = 0.7) -> List[Tuple[float, Chunk]]:
    """Combines semantic and keyword-based search."""
    results = []
    
    # Semantic search scores (from embeddings)
    if chunks and len(chunks[0].embedding) > 0:
        try:
            question_embedding = embed_text(question)  # This would need to be async in real use
            chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
            semantic_scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
        except:
            semantic_scores = np.zeros(len(chunks))
    else:
        semantic_scores = np.zeros(len(chunks))
    
    # TF-IDF scores
    tfidf_scores = np.zeros(len(chunks))
    if retriever.tfidf_vectorizer and retriever.tfidf_matrix is not None:
        try:
            question_tfidf = retriever.tfidf_vectorizer.transform([question])
            tfidf_scores = cosine_similarity(question_tfidf, retriever.tfidf_matrix)[0]
        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e}")
    
    # Combine scores
    for i, chunk in enumerate(chunks):
        combined_score = (semantic_weight * semantic_scores[i] + 
                         (1 - semantic_weight) * tfidf_scores[i])
        results.append((combined_score, chunk))
    
    return sorted(results, key=lambda x: x[0], reverse=True)

async def find_relevant_chunks(question: str, all_chunks: List[Chunk], 
                             semaphore: asyncio.Semaphore, top_k: int = 5) -> List[Chunk]:
    """Enhanced retrieval with multiple strategies."""
    if not all_chunks:
        return []

    logger.info(f"Finding relevant chunks for question (from {len(all_chunks)} total chunks)")
    
    # Initialize retriever
    retriever = AdvancedRetriever()
    retriever.setup_tfidf(all_chunks)
    
    # Step 1: Keyword filtering to reduce search space
    keyword_filtered = retriever.keyword_filter(question, all_chunks)
    logger.info(f"Keyword filtering reduced chunks to {len(keyword_filtered)}")
    
    # If keyword filtering was too aggressive, use more chunks
    if len(keyword_filtered) < max(10, top_k * 2):
        keyword_filtered = all_chunks[:min(200, len(all_chunks))]
    
    # Step 2: Generate hypothetical answer for HyDE
    try:
        hypothetical_answer = await generate_hypothetical_answer(question, semaphore)
        logger.info("Generated hypothetical answer for HyDE")
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, using original question")
        hypothetical_answer = question
    
    # Step 3: Semantic search with hypothetical answer
    try:
        search_embedding = await embed_text(hypothetical_answer)
        chunk_embeddings = np.array([chunk.embedding for chunk in keyword_filtered])
        
        similarities = cosine_similarity(search_embedding, chunk_embeddings)[0]
        
        # Get more candidates for reranking
        candidate_k = min(len(keyword_filtered), top_k * 8)
        candidate_indices = np.argsort(similarities)[::-1][:candidate_k]
        candidate_chunks = [keyword_filtered[i] for i in candidate_indices]
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        candidate_chunks = keyword_filtered[:top_k * 4]
    
    if not candidate_chunks:
        return []

    # Step 4: Re-ranking for final selection
    logger.info(f"Re-ranking {len(candidate_chunks)} candidates")
    try:
        reranked_chunks = await rerank_results(question, candidate_chunks)
        final_chunks = reranked_chunks[:top_k]
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}, using semantic similarity results")
        final_chunks = candidate_chunks[:top_k]
    
    logger.info(f"Selected {len(final_chunks)} final chunks")
    
    # Log the selected chunks for debugging
    for i, chunk in enumerate(final_chunks):
        logger.debug(f"Chunk {i+1} (page {chunk.page_number}): {chunk.text[:100]}...")
    
    return final_chunks