import logging
import asyncio
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import Optional

# Corrected Imports: All imports are direct since files are in the same folder.
from config import settings
from schemas import RAGRequest, RAGResponse
from cache import processed_document_cache
from pdf_processor import process_pdf_to_chunks
from retriever import find_relevant_chunks
from llm_handler import ask_mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="High-Accuracy RAG API")

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Dependency to verify the custom API key in the request header."""
    if authorization != f"Bearer 0c945ddefc63d6c04f42e8d435e9e19f19275cfbaedc6e92afe2edf5afa45011":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/", tags=["General"])
async def root():
    return {"message": "RAG API is running."}

@app.post("/hackrx/run", response_model=RAGResponse, tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def process_questions(request: RAGRequest):
    """
    Main endpoint to process a document and answer questions about it.
    """
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
    for question in request.questions:
        logger.info(f"Answering question: '{question}'")
        
        relevant_chunks = await find_relevant_chunks(question, all_chunks, top_k=5)
        
        if not relevant_chunks:
            answers.append("Could not find any relevant context in the document for this question.")
            continue
        
        answer = await ask_mistral(question, relevant_chunks)
        answers.append(answer)
        
        # A small delay to respect API rate limits when asking multiple questions.
        await asyncio.sleep(1.0) 
        
    return RAGResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    # Make the project runnable with 'python main.py'
    uvicorn.run(app, host="0.0.0.0", port=8000)