# /config.py - OPTIMIZED VERSION

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Enhanced settings with performance optimization parameters.
    """
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    # API Keys
    API_KEY: str
    MISTRAL_API_KEY: str
    HUGGINGFACE_API_KEY: str

    # Performance Settings
    MAX_CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNKS_PER_DOCUMENT: int = 500
    
    # Retrieval Settings
    DEFAULT_TOP_K: int = 8
    MAX_CANDIDATE_CHUNKS: int = 50
    RERANK_MAX_CHUNKS: int = 30
    
    # API Rate Limiting
    MISTRAL_MAX_CONCURRENT: int = 3
    HUGGINGFACE_MAX_CONCURRENT: int = 5
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_DELAY: float = 1.0
    
    # Embedding Settings
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    
    # Text Processing
    ENABLE_ADVANCED_CLEANING: bool = True
    PRESERVE_TABLES: bool = True
    EXTRACT_NUMBERS_AGGRESSIVELY: bool = True
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    MAX_CACHE_SIZE: int = 100  # Maximum number of documents to cache
    CACHE_TTL_HOURS: int = 24
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_DEBUG_LOGGING: bool = False
    
    # Advanced Features
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_MULTI_PERSPECTIVE: bool = True
    ENABLE_ANSWER_VERIFICATION: bool = True
    
    # Question Processing
    QUESTION_BATCH_SIZE: int = 2
    ENABLE_PARALLEL_PROCESSING: bool = True
    
    # Model Selection
    SMALL_MODEL: str = "mistral-small-latest"
    LARGE_MODEL: str = "mistral-large-latest"
    USE_LARGE_MODEL_FOR_COMPLEX: bool = True

# Create settings instance
settings = Settings()

# Validate critical settings
def validate_settings():
    """Validate that all critical settings are properly configured."""
    required_keys = ["API_KEY", "MISTRAL_API_KEY", "HUGGINGFACE_API_KEY"]
    
    for key in required_keys:
        if not getattr(settings, key, None):
            raise ValueError(f"Missing required environment variable: {key}")
    
    # Validate numeric settings
    if settings.MAX_CHUNK_SIZE < settings.MIN_CHUNK_SIZE:
        raise ValueError("MAX_CHUNK_SIZE must be greater than MIN_CHUNK_SIZE")
    
    if settings.CHUNK_OVERLAP >= settings.MAX_CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be less than MAX_CHUNK_SIZE")
    
    print("✓ All settings validated successfully")

# Performance monitoring
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    
    # Memory management
    MAX_MEMORY_MB = 2048
    GARBAGE_COLLECT_FREQUENCY = 10  # Every N requests
    
    # Request timeout settings
    PDF_DOWNLOAD_TIMEOUT = 180
    EMBEDDING_TIMEOUT = 120
    LLM_TIMEOUT = 120
    RERANK_TIMEOUT = 180
    
    # Batch processing
    OPTIMAL_BATCH_SIZE = 2
    MAX_CONCURRENT_QUESTIONS = 4
    
    # Quality thresholds
    MIN_SIMILARITY_THRESHOLD = 0.1
    MIN_RERANK_SCORE = 0.0
    MIN_ANSWER_LENGTH = 10
    
    # Feature flags for A/B testing
    ENABLE_HYDE = True
    ENABLE_KEYWORD_FILTERING = True
    ENABLE_TFIDF_HYBRID = True
    ENABLE_CONTEXT_EXPANSION = True

# Initialize performance config
perf_config = PerformanceConfig()

# Validate settings on import
if __name__ != "__main__":
    try:
        validate_settings()
    except Exception as e:
        print(f"Warning: Settings validation failed: {e}")