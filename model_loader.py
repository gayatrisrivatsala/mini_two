# /rag_project/model_loader.py

from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing and loading the SentenceTransformer model...")
# This line loads the model into memory. It will only be run once when the application starts.
shared_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
logger.info("SentenceTransformer model loaded successfully and is ready.")