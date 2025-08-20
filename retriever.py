# /retriever.py - ENHANCED VERSION (Maintaining your successful approach)

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

class EnhancedRetriever:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms with enhanced insurance domain knowledge."""
        key_terms = []
        
        # Extract numbers (often critical in insurance)
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        key_terms.extend(numbers)
        
        # Extract monetary amounts
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        key_terms.extend(money)
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        key_terms.extend(percentages)
        
        # Extract time periods
        time_periods = re.findall(r'\d+\s*(?:day|month|year|week)s?', text.lower())
        key_terms.extend(time_periods)
        
        # Enhanced insurance terms with more comprehensive coverage
        insurance_terms = [
            'premium', 'deductible', 'coverage', 'policy', 'claim', 'benefit',
            'exclusion', 'liability', 'copay', 'coinsurance', 'maximum', 'minimum',
            'annual', 'monthly', 'yearly', 'limit', 'amount', 'fee', 'cost',
            'waiting', 'period', 'pre-existing', 'maternity', 'hospitalization',
            'treatment', 'diagnosis', 'surgery', 'emergency', 'outpatient',
            'inpatient', 'daycare', 'consultation', 'prescription', 'medicine',
            'hospital', 'doctor', 'specialist', 'nurse', 'ambulance', 'discharge',
            'admission', 'room', 'icu', 'intensive', 'care', 'unit', 'bed',
            'cashless', 'reimbursement', 'network', 'provider', 'ayush',
            'alternative', 'therapy', 'rehabilitation', 'physiotherapy',
            'dental', 'optical', 'vision', 'hearing', 'mental', 'health',
            'wellness', 'preventive', 'checkup', 'screening', 'vaccination',
            'immunization', 'chronic', 'acute', 'terminal', 'critical',
            'illness', 'disease', 'condition', 'disorder', 'syndrome',
            'infection', 'virus', 'bacteria', 'cancer', 'tumor', 'malignant',
            'benign', 'chemotherapy', 'radiation', 'dialysis', 'transplant',
            'organ', 'donor', 'recipient', 'surgery', 'operation', 'procedure',
            'diagnostic', 'test', 'scan', 'xray', 'mri', 'ct', 'ultrasound',
            'biopsy', 'endoscopy', 'colonoscopy', 'mammography', 'ecg', 'ekg'
        ]
        
        text_lower = text.lower()
        for term in insurance_terms:
            if term in text_lower:
                key_terms.append(term)
        
        return list(set(key_terms))
    
    def keyword_filter(self, question: str, chunks: List[Chunk]) -> List[Chunk]:
        """Enhanced but less aggressive keyword filtering."""
        key_terms = self.extract_key_terms(question)
        
        if not key_terms:
            return chunks
        
        scored_chunks = []
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            score = 0
            
            # Exact keyword matches (high weight)
            for term in key_terms:
                if term.lower() in chunk_lower:
                    score += 3  # Increased weight for exact matches
            
            # Question word overlap (generous scoring)
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            overlap = len(question_words.intersection(chunk_words))
            score += overlap * 0.3  # Increased from 0.1
            
            # Bonus for definition-like content
            if any(indicator in chunk_lower for indicator in ['means', 'defined as', 'refers to', 'definition']):
                if any(qword in question_lower for qword in ['what is', 'define', 'meaning']):
                    score += 2
            
            # Bonus for procedural content
            if any(indicator in chunk_lower for indicator in ['procedure', 'process', 'steps', 'how to']):
                if any(qword in question_lower for qword in ['how', 'process', 'procedure']):
                    score += 2
            
            scored_chunks.append((score, chunk))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Less aggressive filtering - keep more potentially relevant chunks
        filtered_chunks = [chunk for score, chunk in scored_chunks if score > 0]
        
        # Ensure we have enough chunks - if filtering is too aggressive, keep more
        min_chunks = max(len(chunks) // 3, 15)  # At least 1/3 of chunks or 15, whichever is higher
        if len(filtered_chunks) < min_chunks:
            filtered_chunks = [chunk for score, chunk in scored_chunks[:min_chunks]]
        
        return filtered_chunks if filtered_chunks else chunks
    
    def setup_tfidf(self, chunks: List[Chunk]):
        """Setup TF-IDF vectorizer for hybrid search."""
        try:
            texts = [chunk.text for chunk in chunks]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.85,  # Slightly higher to keep more terms
                lowercase=True
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("TF-IDF vectorizer setup completed")
        except Exception as e:
            logger.warning(f"TF-IDF setup failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

async def embed_text(text: str) -> np.ndarray:
    """Embeds a single string with enhanced retry logic."""
    max_retries = 4  # Increased retries
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
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
            wait_time = min(2 ** attempt, 10)  # Cap wait time at 10 seconds
            logger.warning(f"Embedding attempt {attempt + 1} failed, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)

async def rerank_results(question: str, chunks: List[Chunk], max_chunks: int = 50) -> List[Chunk]:
    """Enhanced re-ranking with better error handling and fallback."""
    if not chunks or len(chunks) <= 3:
        return chunks
    
    # Limit chunks for API efficiency but be generous
    chunks_to_rerank = chunks[:max_chunks]
    pairs = [[question, chunk.text] for chunk in chunks_to_rerank]
    
    max_retries = 4
    
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
                
            # Combine scores with chunks and sort
            scored_chunks = list(zip(scores, chunks_to_rerank))
            scored_chunks.sort(key=lambda x: x[0]['score'], reverse=True)
            
            reranked = [chunk for score, chunk in scored_chunks]
            
            # Add any remaining chunks that weren't reranked
            if len(chunks) > max_chunks:
                reranked.extend(chunks[max_chunks:])
            
            logger.info(f"Successfully reranked {len(chunks_to_rerank)} chunks")
            return reranked
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Re-ranking failed after {max_retries} attempts: {e}")
                return chunks
            wait_time = min(3 ** attempt, 15)  # Exponential backoff with cap
            logger.warning(f"Re-ranking attempt {attempt + 1} failed, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    return chunks

def hybrid_search(question: str, chunks: List[Chunk], retriever: EnhancedRetriever, 
                 semantic_weight: float = 0.75) -> List[Tuple[float, Chunk]]:
    """Enhanced hybrid search combining semantic and keyword-based approaches."""
    if not chunks:
        return []
    
    results = []
    
    # Semantic search scores (primary)
    semantic_scores = np.zeros(len(chunks))
    if chunks and len(chunks[0].embedding) > 0:
        try:
            # This needs to be called from an async context in practice
            # For now, we'll use the existing embeddings
            chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
            if chunk_embeddings.size > 0:
                # Use a simple scoring based on existing embeddings
                # In practice, you'd embed the question here
                semantic_scores = np.random.random(len(chunks))  # Placeholder
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            semantic_scores = np.zeros(len(chunks))
    
    # TF-IDF scores (secondary)
    tfidf_scores = np.zeros(len(chunks))
    if retriever.tfidf_vectorizer and retriever.tfidf_matrix is not None:
        try:
            question_tfidf = retriever.tfidf_vectorizer.transform([question])
            tfidf_scores = cosine_similarity(question_tfidf, retriever.tfidf_matrix)[0]
        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e}")
    
    # Combine scores with slight preference for semantic
    for i, chunk in enumerate(chunks):
        combined_score = (semantic_weight * semantic_scores[i] + 
                         (1 - semantic_weight) * tfidf_scores[i])
        results.append((combined_score, chunk))
    
    return sorted(results, key=lambda x: x[0], reverse=True)

async def find_relevant_chunks(question: str, all_chunks: List[Chunk], 
                             semaphore: asyncio.Semaphore, top_k: int = 12) -> List[Chunk]:
    """
    Enhanced retrieval maintaining your successful approach while improving performance.
    Increased top_k from default for better context coverage.
    """
    if not all_chunks:
        return []

    logger.info(f"Finding relevant chunks for question (from {len(all_chunks)} total chunks)")
    
    # Initialize retriever
    retriever = EnhancedRetriever()
    retriever.setup_tfidf(all_chunks)
    
    # Step 1: Enhanced but less aggressive keyword filtering
    keyword_filtered = retriever.keyword_filter(question, all_chunks)
    logger.info(f"Keyword filtering selected {len(keyword_filtered)} chunks")
    
    # Ensure we have enough chunks for good context
    if len(keyword_filtered) < max(20, top_k * 2):
        # If filtering was too aggressive, use more chunks
        keyword_filtered = all_chunks[:min(len(all_chunks), max(50, len(keyword_filtered) * 2))]
        logger.info(f"Expanded to {len(keyword_filtered)} chunks for better coverage")
    
    # Step 2: Generate hypothetical answer for improved retrieval
    try:
        hypothetical_answer = await generate_hypothetical_answer(question, semaphore)
        logger.info("Generated hypothetical answer for enhanced retrieval")
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, using original question")
        hypothetical_answer = question
    
    # Step 3: Semantic search with hypothetical answer
    try:
        search_embedding = await embed_text(hypothetical_answer)
        chunk_embeddings = np.array([chunk.embedding for chunk in keyword_filtered])
        
        similarities = cosine_similarity(search_embedding, chunk_embeddings)[0]
        
        # Get more candidates for reranking - be generous
        candidate_k = min(len(keyword_filtered), top_k * 6)  # More candidates
        candidate_indices = np.argsort(similarities)[::-1][:candidate_k]
        candidate_chunks = [keyword_filtered[i] for i in candidate_indices]
        
        logger.info(f"Semantic search selected {len(candidate_chunks)} candidates")
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Fallback to keyword-filtered chunks
        candidate_chunks = keyword_filtered[:top_k * 4]
        logger.info(f"Using fallback selection: {len(candidate_chunks)} chunks")
    
    if not candidate_chunks:
        logger.warning("No candidate chunks found, using top chunks from all")
        return all_chunks[:top_k]

    # Step 4: Re-ranking for final selection
    logger.info(f"Re-ranking {len(candidate_chunks)} candidates")
    try:
        reranked_chunks = await rerank_results(question, candidate_chunks, max_chunks=60)
        final_chunks = reranked_chunks[:top_k]
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}, using semantic similarity order")
        final_chunks = candidate_chunks[:top_k]
    
    logger.info(f"Selected {len(final_chunks)} final chunks for answer generation")
    
    # Log some debug info about selected chunks
    if logger.isEnabledFor(logging.DEBUG):
        for i, chunk in enumerate(final_chunks[:3]):  # Log first 3 chunks
            logger.debug(f"Chunk {i+1} (page {chunk.page_number}): {chunk.text[:100]}...")
    
    return final_chunks