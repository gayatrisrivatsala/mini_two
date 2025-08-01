# /main.py

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import Optional

from config import settings
from schemas import RAGRequest, RAGResponse
from cache import processed_document_cache
from pdf_processor import process_pdf_to_chunks
from retriever import find_relevant_chunks
from llm_handler import ask_mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The semaphore is kept as a defense-in-depth measure.
API_CONCURRENCY = 5
MISTRAL_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY)

app = FastAPI(title="High-Accuracy RAG API")

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Dependency to verify the API key."""
    if authorization != f"Bearer {settings.API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def process_questions(request: RAGRequest):
    """Main endpoint to process a document and answer questions."""
    doc_url = str(request.documents)
    
    if doc_url in processed_document_cache:
        logger.info(f"Cache HIT for document: {doc_url}")
        all_chunks = processed_document_cache[doc_url]
    else:
        logger.info(f"Cache MISS. Processing document: {doc_url}")
        try:
            all_chunks = await process_pdf_to_chunks(doc_url)
            if not all_chunks:
                raise ValueError("Failed to extract any content from the document.")
            processed_document_cache[doc_url] = all_chunks
        except Exception as e:
            logger.error(f"Failed to process PDF from {doc_url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document. Error: {str(e)}")

    answers = []
    
    # --- THE DEFINITIVE FIX: Process questions sequentially ---
    # Instead of creating a burst of concurrent requests with asyncio.gather,
    # we now process each question one by one. This is slower, but it
    # guarantees we will not violate the API's rate limit.
    for question in request.questions:
        logger.info(f"Answering question: '{question}'")
        
        # Each 'await' will fully complete before the next loop iteration begins.
        relevant_chunks = await find_relevant_chunks(question, all_chunks, MISTRAL_SEMAPHORE, top_k=7)
        
        if not relevant_chunks:
            answers.append("Could not find any relevant context in the document for this question.")
            continue
        
        answer = await ask_mistral(question, relevant_chunks, MISTRAL_SEMAPHORE)
        answers.append(answer)
        
    return RAGResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)