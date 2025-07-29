import logging
import asyncio # Import the asyncio library
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import Optional

from config import settings
from schemas import RAGRequest, RAGResponse
from cache import processed_document_cache
from pdf_processor import process_pdf_to_chunks
from retriever import find_relevant_chunks
from llm_handler import ask_mistral
from model_loader import shared_model # Ensure model is loaded at startup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized RAG API")

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {settings.API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/", tags=["General"])
async def root():
    return {"message": "RAG API is running."}

@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def process_questions(request: RAGRequest):
    doc_url = str(request.documents)
    
    if doc_url in processed_document_cache:
        logger.info(f"Cache HIT for document: {doc_url}")
        all_chunks = processed_document_cache[doc_url]
    else:
        logger.info(f"Cache MISS. Processing document: {doc_url}")
        try:
            all_chunks = await process_pdf_to_chunks(doc_url)
            processed_document_cache[doc_url] = all_chunks
        except Exception as e:
            logger.error(f"Failed to process PDF from {doc_url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document. Error: {str(e)}")

    answers = []
    for question in request.questions:
        logger.info(f"Answering question: '{question}'")
        relevant_chunks = find_relevant_chunks(question, all_chunks, top_k=7)
        if not relevant_chunks:
            answers.append("Could not find any relevant context in the document for this question.")
            continue
        
        answer = await ask_mistral(question, relevant_chunks)
        answers.append(answer)
        
        # --- KEY ADDITION ---
        # Add a 1-second delay to respect API rate limits before the next call.
        await asyncio.sleep(1.5) 
        
    return RAGResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)