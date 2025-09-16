from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Jira settings
    jira_url: str = ""
    jira_email: str = ""
    jira_project: str = ""
    jira_api_token: str = ""
    
    # GitHub settings
    github_token: str = ""
    github_repo_owner: str = ""
    github_repo: str = ""
    github_branch_prefix: str = "feature"
    
    # LLM settings
    openai_api_key: str
    huggingface_api_key: str
    anthropic_api_key: str
    google_api_key: str
    model_name: str = "gpt-4-turbo"
    
    # RAG settings
    vector_db_path: str = "./chroma_db"
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()