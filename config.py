
### 3. `config.py`


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    API_KEY: str = "hackrx_secure_key_2024_production"
    MISTRAL_API_KEY: str

settings = Settings()