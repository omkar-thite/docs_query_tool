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
    async_database_url: SecretStr
    
    db_host: str
    app_user: str
    app_password: SecretStr
    app_db: str
    db_port: str


    emb_model_name: str = 'msmarco-bert-base-dot-v5'


settings = Settings()
