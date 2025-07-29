# # /rag_project/pdf_processor.py

# import fitz  # PyMuPDF
# import httpx
# from io import BytesIO
# from typing import List
# import logging
# import re
# from sentence_transformers import SentenceTransformer
# from schemas import Chunk

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# async def download_pdf(url: str) -> bytes:
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         response = await client.get(url)
#         response.raise_for_status()
#         return response.content

# def extract_and_chunk_pdf(pdf_bytes: bytes) -> List[dict]:
#     """Extracts text blocks from a PDF and treats them as semantic chunks."""
#     chunks_with_pages = []
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     for page_num, page in enumerate(doc):
#         # The "blocks" option preserves paragraph structure
#         blocks = page.get_text("blocks")
#         for b in blocks:
#             # b[4] contains the text of the block
#             text = b[4].strip()
#             text = re.sub(r'\s+', ' ', text) # Clean up whitespace
#             # Filter out small, likely irrelevant blocks of text
#             if len(text) > 30:
#                 chunks_with_pages.append({
#                     "text": text,
#                     "page_number": page_num + 1
#                 })
#     doc.close()
#     return chunks_with_pages

# def generate_embeddings(texts: List[str]) -> List[List[float]]:
#     logger.info(f"Generating embeddings for {len(texts)} chunks...")
#     embeddings = model.encode(texts, convert_to_numpy=True)
#     return embeddings.tolist()

# async def process_pdf_to_chunks(url: str) -> List[Chunk]:
#     logger.info(f"Starting PDF processing for: {url}")
#     pdf_bytes = await download_pdf(url)
    
#     raw_chunks = extract_and_chunk_pdf(pdf_bytes)
#     if not raw_chunks:
#         raise ValueError("Could not extract any text chunks from the document.")
        
#     texts_to_embed = [chunk["text"] for chunk in raw_chunks]
#     embeddings = generate_embeddings(texts_to_embed)
    
#     processed_chunks = [
#         Chunk(
#             text=chunk_data["text"],
#             page_number=chunk_data["page_number"],
#             embedding=embeddings[i]
#         )
#         for i, chunk_data in enumerate(raw_chunks)
#     ]
    
#     logger.info(f"Successfully processed PDF into {len(processed_chunks)} chunks.")
#     return processed_chunks




# /rag_project/pdf_processor.py
import fitz  # PyMuPDF
import httpx
import logging
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from schemas import Chunk
import nltk

# Download required NLTK data with proper error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# Import after ensuring data is downloaded
try:
    from nltk.tokenize import sent_tokenize
except:
    # Fallback to simple sentence splitting if NLTK fails
    def sent_tokenize(text):
        """Simple sentence tokenizer as fallback"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    # Remove common page artifacts
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_and_chunk_pdf(pdf_bytes: bytes) -> List[Dict]:
    """Enhanced PDF extraction with better chunking."""
    chunks_with_pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Try multiple extraction methods for better results
        # Method 1: Get text blocks (preserves some structure)
        blocks = page.get_text("blocks")
        
        # Combine blocks intelligently
        page_chunks = []
        current_chunk = ""
        
        for block in blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                text = clean_text(block[4])
                
                # Skip very short blocks
                if len(text) < 20:
                    continue
                
                # Check if this block should be combined with previous
                if current_chunk and len(current_chunk) + len(text) < 500:
                    current_chunk += " " + text
                else:
                    if current_chunk and len(current_chunk) > 50:
                        page_chunks.append(current_chunk)
                    current_chunk = text
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) > 50:
            page_chunks.append(current_chunk)
        
        # If blocks method didn't work well, try full page text
        if not page_chunks:
            full_text = page.get_text()
            if full_text.strip():
                # Split by paragraphs (double newlines or indentation)
                paragraphs = re.split(r'\n\s*\n', full_text)
                for para in paragraphs:
                    cleaned = clean_text(para)
                    if len(cleaned) > 50:
                        page_chunks.append(cleaned)
        
        # Create overlapping chunks for better context
        for i, chunk_text in enumerate(page_chunks):
            chunks_with_pages.append({
                "text": chunk_text,
                "page_number": page_num + 1
            })
            
            # If chunk is large, split it with overlap
            if len(chunk_text) > 1000:
                sentences = sent_tokenize(chunk_text)
                sub_chunks = []
                current = ""
                
                for sent in sentences:
                    if len(current) + len(sent) < 500:
                        current += " " + sent if current else sent
                    else:
                        if current:
                            sub_chunks.append(current.strip())
                        # Keep last sentence for overlap
                        current = sent
                
                if current:
                    sub_chunks.append(current.strip())
                
                # Replace large chunk with smaller overlapping ones
                if len(sub_chunks) > 1:
                    chunks_with_pages.pop()  # Remove the large chunk
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks_with_pages.append({
                            "text": sub_chunk,
                            "page_number": page_num + 1
                        })
    
    doc.close()
    return chunks_with_pages

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings with batching for efficiency."""
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    return all_embeddings

async def process_pdf_to_chunks(url: str) -> List[Chunk]:
    """Main processing function with enhanced error handling."""
    logger.info(f"Starting PDF processing for: {url}")
    
    try:
        pdf_bytes = await download_pdf(url)
        
        raw_chunks = extract_and_chunk_pdf(pdf_bytes)
        if not raw_chunks:
            raise ValueError("Could not extract any text chunks from the document.")
        
        # Filter out low-quality chunks
        quality_chunks = []
        for chunk in raw_chunks:
            text = chunk["text"]
            # Basic quality checks
            if len(text.split()) < 10:  # Too few words
                continue
            if len(set(text.split())) < 5:  # Too repetitive
                continue
            if text.count('\x00') > 0:  # Null characters (corrupted)
                continue
            quality_chunks.append(chunk)
        
        if not quality_chunks:
            # Fallback to original chunks if quality filtering was too aggressive
            quality_chunks = raw_chunks
        
        texts_to_embed = [chunk["text"] for chunk in quality_chunks]
        embeddings = generate_embeddings(texts_to_embed)
        
        processed_chunks = [
            Chunk(
                text=chunk_data["text"],
                page_number=chunk_data["page_number"],
                embedding=embeddings[i]
            )
            for i, chunk_data in enumerate(quality_chunks)
        ]
        
        logger.info(f"Successfully processed PDF into {len(processed_chunks)} chunks.")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

# Alternative implementation without NLTK dependency
def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitter without NLTK."""
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# If NLTK continues to fail, replace sent_tokenize with simple_sentence_split