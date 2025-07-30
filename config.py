from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Loads all environment variables from the .env file.
    Pydantic ensures that all required variables are present on startup.
    """
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    # FastAPI application key to secure your endpoints
    API_KEY: str

    # API key for the Mistral LLM for answer generation
    MISTRAL_API_KEY: str

    # API key for Hugging Face Inference APIs for embedding and re-ranking
    HUGGINGFACE_API_KEY: str

# Create a single, globally accessible settings instance
settings = Settings()