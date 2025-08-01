# /config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Loads all environment variables from the .env file.
    """
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    # FastAPI application key to secure your endpoints
    API_KEY: str

    # A single API key for the Mistral LLM for all operations
    MISTRAL_API_KEY: str

    # API key for Hugging Face Inference APIs
    HUGGINGFACE_API_KEY: str

# Create a single, globally accessible settings instance
settings = Settings()