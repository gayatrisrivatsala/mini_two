# Enhanced config.py for Insurance Documents

from pydantic_settings import BaseSettings
from typing import Optional, Dict
import os

class InsuranceSettings(BaseSettings):
    """Enhanced settings optimized for insurance document processing."""
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    # API Keys
    API_KEY: str
    MISTRAL_API_KEY: str
    HUGGINGFACE_API_KEY: str

    # Insurance Document Processing Settings
    MAX_CHUNK_SIZE: int = 1500  # Larger chunks for insurance context
    CHUNK_OVERLAP: int = 300    # Higher overlap for policy continuity
    MIN_CHUNK_SIZE: int = 100   # Larger minimum for meaningful content
    MAX_CHUNKS_PER_DOCUMENT: int = 800  # More chunks for comprehensive documents
    
    # Insurance-Specific Retrieval Settings
    DEFAULT_TOP_K: int = 10     # More chunks for complex insurance questions
    MAX_CANDIDATE_CHUNKS: int = 60  # More candidates for better coverage
    RERANK_MAX_CHUNKS: int = 40     # More chunks to rerank
    
    # Enhanced for complex insurance documents
    DEFINITION_CHUNK_PRIORITY: float = 2.0
    TABLE_CHUNK_PRIORITY: float = 1.8
    BENEFIT_CHUNK_PRIORITY: float = 1.5
    EXCLUSION_CHUNK_PRIORITY: float = 1.5
    
    # Multi-query expansion settings
    ENABLE_INSURANCE_QUERY_EXPANSION: bool = True
    MAX_EXPANDED_QUERIES: int = 3
    
    # API Rate Limiting (adjusted for insurance complexity)
    MISTRAL_MAX_CONCURRENT: int = 2  # Conservative for complex queries
    HUGGINGFACE_MAX_CONCURRENT: int = 4
    API_RETRY_ATTEMPTS: int = 4  # More retries for critical insurance data
    API_RETRY_DELAY: float = 1.5
    
    # Embedding Settings
    EMBEDDING_BATCH_SIZE: int = 24  # Slightly smaller for complex insurance text
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    
    # Insurance Text Processing
    ENABLE_ADVANCED_CLEANING: bool = True
    PRESERVE_TABLES: bool = True
    PRESERVE_DEFINITIONS: bool = True
    PRESERVE_SECTION_MARKERS: bool = True
    EXTRACT_FINANCIAL_DATA: bool = True
    EXTRACT_PERCENTAGES: bool = True
    EXTRACT_TIME_PERIODS: bool = True
    
    # Cache Settings (insurance docs are often reused)
    ENABLE_CACHE: bool = True
    MAX_CACHE_SIZE: int = 50
    CACHE_TTL_HOURS: int = 48  # Longer cache for insurance docs
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_DEBUG_LOGGING: bool = True  # More logging for insurance complexity
    
    # Insurance-Specific Features
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_MULTI_PERSPECTIVE: bool = True
    ENABLE_ANSWER_VERIFICATION: bool = True
    ENABLE_DEFINITION_EXTRACTION: bool = True
    ENABLE_TABLE_STRUCTURE_PRESERVATION: bool = True
    
    # Question Processing
    QUESTION_BATCH_SIZE: int = 1  # Process insurance questions individually
    ENABLE_PARALLEL_PROCESSING: bool = False  # Sequential for better accuracy
    
    # Model Selection for Insurance
    SMALL_MODEL: str = "mistral-small-latest"
    LARGE_MODEL: str = "mistral-large-latest"
    USE_LARGE_MODEL_FOR_COMPLEX: bool = True
    USE_LARGE_MODEL_FOR_FINANCIAL: bool = True
    USE_LARGE_MODEL_FOR_DEFINITIONS: bool = True

class InsurancePerformanceConfig:
    """Enhanced performance configuration for insurance document processing."""
    
    # Memory management for larger documents
    MAX_MEMORY_MB = 3072  # Increased for insurance documents
    GARBAGE_COLLECT_FREQUENCY = 5  # More frequent for large docs
    
    # Request timeout settings (higher for insurance complexity)
    PDF_DOWNLOAD_TIMEOUT = 240
    EMBEDDING_TIMEOUT = 180
    LLM_TIMEOUT = 180
    RERANK_TIMEOUT = 240
    
    # Batch processing
    OPTIMAL_BATCH_SIZE = 1  # Individual processing for insurance
    MAX_CONCURRENT_QUESTIONS = 2  # Conservative for accuracy
    
    # Quality thresholds (stricter for insurance)
    MIN_SIMILARITY_THRESHOLD = 0.15
    MIN_RERANK_SCORE = 0.05
    MIN_ANSWER_LENGTH = 20
    
    # Insurance-specific feature flags
    ENABLE_HYDE = True
    ENABLE_KEYWORD_FILTERING = True
    ENABLE_TFIDF_HYBRID = True
    ENABLE_CONTEXT_EXPANSION = True
    ENABLE_DEFINITION_PRIORITY = True
    ENABLE_TABLE_PRIORITY = True
    ENABLE_FINANCIAL_EXTRACTION = True
    
    # Insurance document quality metrics
    MIN_DEFINITION_CHUNKS = 5  # Ensure we extract enough definitions
    MIN_TABLE_CHUNKS = 3       # Ensure we extract table data
    MAX_CONTEXT_WINDOW = 8000  # Larger context for complex insurance queries

# Create enhanced settings instance
insurance_settings = InsuranceSettings()
insurance_perf_config = InsurancePerformanceConfig()

# Maintain backward compatibility
settings = insurance_settings
perf_config = insurance_perf_config

def validate_insurance_settings():
    """Validate settings specifically for insurance document processing."""
    required_keys = ["API_KEY", "MISTRAL_API_KEY", "HUGGINGFACE_API_KEY"]
    
    for key in required_keys:
        if not getattr(insurance_settings, key, None):
            raise ValueError(f"Missing required environment variable: {key}")
    
    # Validate insurance-specific settings
    if insurance_settings.MAX_CHUNK_SIZE < 1000:
        print("Warning: MAX_CHUNK_SIZE is quite small for insurance documents")
    
    if insurance_settings.CHUNK_OVERLAP < 200:
        print("Warning: CHUNK_OVERLAP might be too small for insurance context preservation")
    
    if not insurance_settings.PRESERVE_TABLES:
        print("Warning: Table preservation is disabled - this may affect insurance data extraction")
    
    print("✓ Insurance-optimized settings validated successfully")

# Insurance document type detection
class InsuranceDocumentDetector:
    """Detect and classify insurance document types for optimal processing."""
    
    @staticmethod
    def detect_document_type(text_sample: str) -> Dict[str, bool]:
        """Detect what type of insurance document this is."""
        text_lower = text_sample.lower()
        
        return {
            'health_insurance': any(term in text_lower for term in [
                'health', 'medical', 'hospital', 'treatment', 'illness'
            ]),
            'life_insurance': any(term in text_lower for term in [
                'life insurance', 'death benefit', 'life cover', 'mortality'
            ]),
            'general_insurance': any(term in text_lower for term in [
                'general insurance', 'property', 'motor', 'travel'
            ]),
            'has_definitions': 'def.' in text_lower and any(num in text_lower for num in ['1.', '2.', '3.']),
            'has_sections': any(f'section {letter}' in text_lower for letter in 'abcdefgh'),
            'has_tables': any(term in text_lower for term in [
                'schedule', 'table', 'benefits', 'coverage table'
            ]),
            'has_exclusions': any(term in text_lower for term in [
                'exclusion', 'not covered', 'exceptions', 'limitations'
            ]),
            'has_waiting_periods': 'waiting period' in text_lower,
            'has_financial_data': any(term in text_lower for term in [
                'premium', 'sum insured', 'deductible', 'copay'
            ])
        }
    
    @staticmethod
    def get_processing_recommendations(doc_type: Dict[str, bool]) -> Dict[str, any]:
        """Get processing recommendations based on document type."""
        recommendations = {
            'chunk_size': 1500,
            'chunk_overlap': 300,
            'priority_extractors': [],
            'special_handling': []
        }
        
        if doc_type['has_definitions']:
            recommendations['priority_extractors'].append('definitions')
            recommendations['chunk_size'] = max(recommendations['chunk_size'], 1200)
        
        if doc_type['has_tables']:
            recommendations['priority_extractors'].append('tables')
            recommendations['special_handling'].append('preserve_table_structure')
        
        if doc_type['has_financial_data']:
            recommendations['priority_extractors'].append('financial_data')
            recommendations['special_handling'].append('extract_monetary_amounts')
        
        if doc_type['health_insurance']:
            recommendations['priority_extractors'].extend(['medical_terms', 'exclusions'])
        
        return recommendations

# Initialize enhanced settings
if __name__ != "__main__":
    try:
        validate_insurance_settings()
    except Exception as e:
        print(f"Warning: Insurance settings validation failed: {e}")