# /pdf_processor.py

import httpx
import logging
from io import BytesIO
from typing import List
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from schemas import Chunk
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

async def download_pdf(url: str) -> bytes:
    """Downloads a PDF from a URL."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

async def generate_embeddings_via_api(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts by calling the Hugging Face API."""
    logger.info(f"Generating embeddings for {len(texts)} chunks via API...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        all_embeddings = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.post(
                EMBEDDING_API_URL,
                headers=HUGGINGFACE_HEADERS,
                json={"inputs": batch, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            all_embeddings.extend(response.json())
    return all_embeddings

async def process_pdf_to_chunks(url: str) -> List[Chunk]:
    """
    Processes a PDF into small, semantically-aware chunks that are safe for all models.
    """
    logger.info(f"Starting robust semantic chunking for: {url}")
    
    pdf_bytes = await download_pdf(url)
    
    full_text = ""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text(x_tolerance=2) + "\n\n"

    if not full_text.strip():
        raise ValueError("Could not extract any meaningful text from the document.")

    # Create small, semantic chunks guaranteed to be under model token limits.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,  # A safe size for the bge-reranker-large model
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    texts = text_splitter.split_text(full_text)
    
    logger.info(f"Created {len(texts)} semantic chunks.")
    
    # We no longer need complex page mapping; the context is self-contained.
    raw_chunks = [{"text": text, "page_number": 0} for text in texts]
        
    texts_to_embed = [chunk["text"] for chunk in raw_chunks]
    embeddings = await generate_embeddings_via_api(texts_to_embed)
    
    processed_chunks = [
        Chunk(
            text=chunk_data["text"],
            page_number=chunk_data["page_number"], # Page number is less critical with small chunks
            embedding=embeddings[i]
        )
        for i, chunk_data in enumerate(raw_chunks)
    ]
    
    logger.info(f"Successfully processed and embedded {len(processed_chunks)} chunks.")
    return processed_chunks