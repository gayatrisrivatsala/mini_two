# /main.py - OPTIMIZED VERSION

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import Optional, List
import time

from config import settings
from schemas import RAGRequest, RAGResponse
from cache import processed_document_cache
from pdf_processor import process_pdf_to_chunks
from retriever import find_relevant_chunks
from llm_handler import ask_mistral, generate_multi_perspective_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized concurrency settings
API_CONCURRENCY = 3  # Reduced for better rate limit management
MISTRAL_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY)

app = FastAPI(
    title="High-Accuracy RAG API - Optimized",
    description="Enhanced RAG system with advanced retrieval and processing",
    version="2.0.0"
)

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Enhanced API key verification."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization[7:]  # Remove "Bearer " prefix
    if token != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

async def process_single_question(question: str, all_chunks: List, question_index: int) -> tuple:
    """Process a single question with enhanced error handling."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing question {question_index + 1}: '{question[:50]}...'")
        
        # Enhanced retrieval with more chunks for better context
        relevant_chunks = await find_relevant_chunks(
            question, 
            all_chunks, 
            MISTRAL_SEMAPHORE, 
            top_k=8  # Increased from 7
        )
        
        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for question {question_index + 1}")
            return question_index, "Could not find any relevant context in the document for this question."
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for question {question_index + 1}")
        
        # Use enhanced answer generation
        answer = await generate_multi_perspective_answer(question, relevant_chunks, MISTRAL_SEMAPHORE)
        
        processing_time = time.time() - start_time
        logger.info(f"Question {question_index + 1} processed in {processing_time:.2f}s")
        
        return question_index, answer
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing question {question_index + 1} after {processing_time:.2f}s: {e}")
        return question_index, f"Error processing question: {str(e)}"

async def process_questions_batch(questions: List[str], all_chunks: List, batch_size: int = 2) -> List[str]:
    """Process questions in optimized batches to balance speed and rate limits."""
    answers = [""] * len(questions)
    
    # Process questions in small batches to respect rate limits
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(questions))))
        
        logger.info(f"Processing batch {i//batch_size + 1}: questions {i+1} to {min(i + batch_size, len(questions))}")
        
        # Process batch concurrently
        batch_tasks = [
            process_single_question(question, all_chunks, idx)
            for question, idx in zip(batch_questions, batch_indices)
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            
            question_index, answer = result
            answers[question_index] = answer
        
        # Small delay between batches to respect rate limits
        if i + batch_size < len(questions):
            await asyncio.sleep(1)
    
    return answers

@app.post("/mini-2/run", response_model=RAGResponse, tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def process_questions(request: RAGRequest):
    """Main endpoint with enhanced processing and error handling."""
    start_time = time.time()
    doc_url = str(request.documents)
    
    logger.info(f"Processing request with {len(request.questions)} questions for document: {doc_url}")
    
    # Document processing with enhanced caching
    if doc_url in processed_document_cache:
        logger.info(f"Cache HIT for document: {doc_url}")
        all_chunks = processed_document_cache[doc_url]
    else:
        logger.info(f"Cache MISS. Processing document: {doc_url}")
        try:
            processing_start = time.time()
            all_chunks = await process_pdf_to_chunks(doc_url)
            
            if not all_chunks:
                raise ValueError("Failed to extract any content from the document.")
            
            processed_document_cache[doc_url] = all_chunks
            processing_time = time.time() - processing_start
            logger.info(f"Document processed successfully in {processing_time:.2f}s - {len(all_chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Failed to process PDF from {doc_url}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process document. Error: {str(e)}"
            )
    
    # Validate we have content to work with
    if not all_chunks:
        raise HTTPException(
            status_code=500,
            detail="Document processing resulted in no usable content"
        )
    
    # Process questions with optimized batching
    try:
        answers = await process_questions_batch(request.questions, all_chunks, batch_size=2)
        
        # Validate all questions were answered
        for i, answer in enumerate(answers):
            if not answer or answer.strip() == "":
                answers[i] = "Unable to process this question due to an internal error."
        
        total_time = time.time() - start_time
        logger.info(f"All {len(request.questions)} questions processed successfully in {total_time:.2f}s")
        
        return RAGResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in question processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing questions: {str(e)}"
        )

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cache_size": len(processed_document_cache),
        "timestamp": time.time()
    }

@app.get("/cache/stats", tags=["Cache"], dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get cache statistics."""
    return {
        "cached_documents": len(processed_document_cache),
        "cache_keys": list(processed_document_cache.keys()),
        "total_chunks": sum(len(chunks) for chunks in processed_document_cache.values())
    }

@app.delete("/cache/clear", tags=["Cache"], dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear the document cache."""
    cleared_count = len(processed_document_cache)
    processed_document_cache.clear()
    logger.info(f"Cache cleared - {cleared_count} documents removed")
    return {"message": f"Cache cleared - {cleared_count} documents removed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )