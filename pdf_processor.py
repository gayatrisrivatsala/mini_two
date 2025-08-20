# /pdf_processor.py - ENHANCED VERSION (Based on your successful approach)

import httpx
import logging
from io import BytesIO
from typing import List, Dict, Tuple
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import defaultdict
import asyncio

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
    """Enhanced text cleaning that preserves important insurance information."""
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common PDF extraction issues while preserving insurance terms
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix camelCase splits
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Fix number-letter splits
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Fix letter-number splits
    
    # Preserve important insurance document structure
    text = re.sub(r'Section\s*([A-Z])\s*\.', r'Section \1.', text)
    text = re.sub(r'Def\s*\.?\s*(\d+)\s*\.', r'Def. \1.', text)
    
    # Preserve important punctuation and formatting
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove unwanted characters but keep essential ones for insurance docs
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\%\$\&\@\#\/\\]', ' ', text)
    
    # Final cleanup
    return ' '.join(text.split()).strip()

def extract_structured_content(pdf_page) -> Dict[str, str]:
    """Extracts structured content from PDF page with enhanced table handling."""
    content = {
        'main_text': '',
        'tables': '',
        'metadata': ''
    }
    
    # Extract main text with better tolerance settings
    main_text = pdf_page.extract_text(x_tolerance=3, y_tolerance=3)
    if main_text:
        content['main_text'] = clean_and_enhance_text(main_text)
    
    # Enhanced table extraction for insurance documents
    try:
        tables = pdf_page.extract_tables()
        if tables:
            table_text = []
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 1:
                    continue
                
                # Add table header for context
                table_text.append(f"[TABLE {table_idx + 1}]:")
                
                for row_idx, row in enumerate(table):
                    if row:
                        # Clean and join cells, handling None values
                        clean_row = []
                        for cell in row:
                            if cell is not None:
                                cell_text = str(cell).strip()
                                if cell_text:
                                    clean_row.append(cell_text)
                        
                        if clean_row:
                            # Use different separators for header vs data rows
                            if row_idx == 0:
                                table_text.append(' | '.join(clean_row))
                                table_text.append('-' * 50)  # Separator line
                            else:
                                table_text.append(' | '.join(clean_row))
                
                table_text.append('')  # Empty line between tables
            
            content['tables'] = '\n'.join(table_text)
    except Exception as e:
        logger.warning(f"Failed to extract tables: {e}")
    
    return content

async def generate_embeddings_via_api(texts: List[str]) -> List[List[float]]:
    """Optimized embedding generation with better error handling and retry logic."""
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    if not texts:
        return []
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        all_embeddings = []
        batch_size = 24  # Slightly smaller for stability with complex insurance text
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            max_retries = 4  # More retries for critical processing
            
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
                    
                    # Log progress for large documents
                    if len(texts) > 100:
                        logger.info(f"Embedding progress: {min(i + batch_size, len(texts))}/{len(texts)}")
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)
    
    return all_embeddings

def create_smart_chunks(text: str, page_num: int) -> List[Dict[str, any]]:
    """Enhanced chunking optimized for insurance documents with better context preservation."""
    if not text.strip():
        return []
    
    # Enhanced text splitters with better parameters for insurance content
    splitters = [
        # Primary splitter for comprehensive context
        RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better insurance context
            chunk_overlap=300,  # Higher overlap for policy continuity
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len
        ),
        # Secondary splitter for dense content
        RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Still substantial for context
            chunk_overlap=200,  # Maintain good overlap
            separators=["\n", ". ", " ", ""],
            length_function=len
        )
    ]
    
    all_chunks = []
    
    # Try primary splitter first
    try:
        primary_chunks = splitters[0].split_text(text)
        
        # Check if chunks are too small (indicating dense formatting)
        if primary_chunks and len(primary_chunks) > 1:
            avg_chunk_size = sum(len(chunk) for chunk in primary_chunks[:5]) / min(5, len(primary_chunks))
            if avg_chunk_size < 400:
                logger.info(f"Using secondary splitter for page {page_num} due to dense formatting")
                secondary_chunks = splitters[1].split_text(text)
                chunks_to_use = secondary_chunks if secondary_chunks else primary_chunks
            else:
                chunks_to_use = primary_chunks
        else:
            chunks_to_use = primary_chunks
            
    except Exception as e:
        logger.warning(f"Primary chunking failed: {e}, falling back to secondary")
        chunks_to_use = splitters[1].split_text(text)
    
    # Create chunk objects with enhanced metadata
    for i, chunk_text in enumerate(chunks_to_use):
        if len(chunk_text.strip()) < 80:  # Slightly higher minimum for meaningful content
            continue
            
        # Detect chunk type for potential future use
        chunk_type = "main_content"
        if "def." in chunk_text.lower() and any(str(num) in chunk_text for num in range(1, 20)):
            chunk_type = "definition"
        elif "table" in chunk_text.lower() or "|" in chunk_text:
            chunk_type = "table_data"
        elif "section" in chunk_text.lower() and any(letter in chunk_text.lower() for letter in "abcdefgh"):
            chunk_type = "section_header"
        
        chunk_data = {
            "text": chunk_text.strip(),
            "page_number": page_num,
            "chunk_index": i,
            "chunk_type": chunk_type
        }
        all_chunks.append(chunk_data)
    
    return all_chunks

async def process_pdf_to_chunks(url: str) -> List[Chunk]:
    """
    Enhanced PDF processing that maintains your successful approach while improving
    handling of complex insurance documents.
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
                    
                    # Combine all content for this page with better organization
                    page_text_parts = []
                    
                    if content['main_text']:
                        page_text_parts.append(content['main_text'])
                    
                    if content['tables']:
                        # Add clear separator for table content
                        page_text_parts.append(f"\n[TABLE DATA FROM PAGE {page_num}]:\n{content['tables']}")
                    
                    if page_text_parts:
                        full_page_text = '\n\n'.join(page_text_parts)
                        page_contents.append((page_num, full_page_text))
                        
                        # Log progress for large documents
                        if len(pdf.pages) > 20 and page_num % 10 == 0:
                            logger.info(f"Processed {page_num}/{len(pdf.pages)} pages")
                        
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
    
    # Generate embeddings with progress tracking
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