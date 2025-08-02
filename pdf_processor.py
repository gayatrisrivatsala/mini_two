# /pdf_processor.py - OPTIMIZED VERSION

import httpx
import logging
from io import BytesIO
from typing import List, Dict, Tuple
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import defaultdict

from schemas import Chunk
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

async def download_pdf(url: str) -> bytes:
    """Downloads a PDF from a URL with better error handling."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            raise

def clean_and_enhance_text(text: str) -> str:
    """Enhanced text cleaning that preserves important information."""
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix camelCase splits
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Fix number-letter splits
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Fix letter-number splits
    
    # Preserve important punctuation and formatting
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove unwanted characters but keep essential ones
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\%\$\&\@\#\/\\]', ' ', text)
    
    # Final cleanup
    return ' '.join(text.split()).strip()

def extract_structured_content(pdf_page) -> Dict[str, str]:
    """Extracts structured content from PDF page."""
    content = {
        'main_text': '',
        'tables': '',
        'metadata': ''
    }
    
    # Extract main text
    main_text = pdf_page.extract_text(x_tolerance=2, y_tolerance=2)
    if main_text:
        content['main_text'] = clean_and_enhance_text(main_text)
    
    # Extract tables
    try:
        tables = pdf_page.extract_tables()
        if tables:
            table_text = []
            for table in tables:
                for row in table:
                    if row:
                        clean_row = [str(cell).strip() if cell else '' for cell in row]
                        table_text.append(' | '.join(clean_row))
            content['tables'] = '\n'.join(table_text)
    except Exception as e:
        logger.warning(f"Failed to extract tables: {e}")
    
    return content

async def generate_embeddings_via_api(texts: List[str]) -> List[List[float]]:
    """Optimized embedding generation with better batching."""
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    if not texts:
        return []
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        all_embeddings = []
        batch_size = 32  # Reduced batch size for better stability
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        EMBEDDING_API_URL,
                        headers=HUGGINGFACE_HEADERS,
                        json={"inputs": batch, "options": {"wait_for_model": True}},
                        timeout=120.0
                    )
                    response.raise_for_status()
                    batch_embeddings = response.json()
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)
    
    return all_embeddings

def create_smart_chunks(text: str, page_num: int) -> List[Dict[str, any]]:
    """Creates intelligent chunks with overlap and context preservation."""
    if not text.strip():
        return []
    
    # Use multiple text splitters for different content types
    splitters = [
        # Primary splitter for general content
        RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased from 450
            chunk_overlap=200,  # Increased overlap for context
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len
        ),
        # Secondary splitter for dense content
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n", ". ", " ", ""],
            length_function=len
        )
    ]
    
    all_chunks = []
    
    # Try primary splitter first
    try:
        primary_chunks = splitters[0].split_text(text)
        
        # If chunks are too small, use secondary splitter
        if primary_chunks and len(primary_chunks) > 1 and all(len(chunk) < 300 for chunk in primary_chunks[:3]):
            secondary_chunks = splitters[1].split_text(text)
            chunks_to_use = secondary_chunks if secondary_chunks else primary_chunks
        else:
            chunks_to_use = primary_chunks
            
    except Exception as e:
        logger.warning(f"Primary chunking failed: {e}, falling back to secondary")
        chunks_to_use = splitters[1].split_text(text)
    
    # Create chunk objects with enhanced metadata
    for i, chunk_text in enumerate(chunks_to_use):
        if len(chunk_text.strip()) < 50:  # Skip very small chunks
            continue
            
        chunk_data = {
            "text": chunk_text.strip(),
            "page_number": page_num,
            "chunk_index": i,
            "chunk_type": "main_content"
        }
        all_chunks.append(chunk_data)
    
    return all_chunks

async def process_pdf_to_chunks(url: str) -> List[Chunk]:
    """
    Enhanced PDF processing with better structure preservation and chunking.
    """
    logger.info(f"Starting enhanced PDF processing for: {url}")
    
    try:
        pdf_bytes = await download_pdf(url)
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise ValueError(f"Could not download PDF from {url}")
    
    page_contents = []
    
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            logger.info(f"Processing {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    content = extract_structured_content(page)
                    
                    # Combine all content for this page
                    page_text_parts = []
                    
                    if content['main_text']:
                        page_text_parts.append(content['main_text'])
                    
                    if content['tables']:
                        page_text_parts.append(f"[TABLE DATA]: {content['tables']}")
                    
                    if page_text_parts:
                        full_page_text = '\n\n'.join(page_text_parts)
                        page_contents.append((page_num, full_page_text))
                        
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        raise ValueError("Could not process PDF file")
    
    if not page_contents:
        raise ValueError("Could not extract any meaningful text from the document")
    
    # Create chunks from all pages
    all_raw_chunks = []
    
    for page_num, page_text in page_contents:
        page_chunks = create_smart_chunks(page_text, page_num)
        all_raw_chunks.extend(page_chunks)
    
    if not all_raw_chunks:
        raise ValueError("No valid chunks could be created from the document")
    
    logger.info(f"Created {len(all_raw_chunks)} chunks from {len(page_contents)} pages")
    
    # Generate embeddings
    texts_to_embed = [chunk["text"] for chunk in all_raw_chunks]
    
    try:
        embeddings = await generate_embeddings_via_api(texts_to_embed)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise ValueError("Could not generate embeddings for document chunks")
    
    # Create final Chunk objects
    processed_chunks = []
    for i, chunk_data in enumerate(all_raw_chunks):
        if i < len(embeddings):
            chunk = Chunk(
                text=chunk_data["text"],
                page_number=chunk_data["page_number"],
                embedding=embeddings[i]
            )
            processed_chunks.append(chunk)
    
    logger.info(f"Successfully processed {len(processed_chunks)} chunks with embeddings")
    return processed_chunks