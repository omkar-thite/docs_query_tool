from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore'
    )                            
    
    github_api_token: SecretStr
    hf_token: SecretStr
    database_url: SecretStr

settings = Settings()