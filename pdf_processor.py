import httpx
import logging
from io import BytesIO
from typing import List
from unstructured.partition.pdf import partition_pdf

# Corrected Imports: Removed the leading dots.
from schemas import Chunk
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Correct, direct model endpoint for the embedding model
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
        response = await client.post(
            EMBEDDING_API_URL,
            headers=HUGGINGFACE_HEADERS,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return response.json()

async def process_pdf_to_chunks(url: str) -> List[Chunk]:
    """
    Processes a digital PDF using the 'unstructured' FAST strategy.
    """
    logger.info(f"Starting digital PDF processing with 'fast' strategy for: {url}")
    
    pdf_bytes = await download_pdf(url)
    
    # Use the 'fast' strategy optimized for digital (non-scanned) PDFs.
    elements = partition_pdf(
        file=BytesIO(pdf_bytes),
        strategy="fast"
    )
    
    raw_chunks = [
        {"text": el.text, "page_number": el.metadata.page_number or 0}
        for el in elements if hasattr(el, 'text') and len(el.text.split()) > 10
    ]

    if not raw_chunks:
        raise ValueError("Could not extract any meaningful text chunks from the document.")
        
    texts_to_embed = [chunk["text"] for chunk in raw_chunks]
    embeddings = await generate_embeddings_via_api(texts_to_embed)
    
    processed_chunks = [
        Chunk(
            text=chunk_data["text"],
            page_number=chunk_data["page_number"],
            embedding=embeddings[i]
        )
        for i, chunk_data in enumerate(raw_chunks)
    ]
    
    logger.info(f"Successfully processed and embedded {len(processed_chunks)} chunks.")
    return processed_chunks