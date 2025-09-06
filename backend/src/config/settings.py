from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Jira settings
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    
    # GitHub settings
    github_token: str = ""
    github_repo: str = ""
    github_branch_prefix: str = "feature/auto-bdd"
    
    # LLM settings
    openai_api_key: str
    huggingface_api_key: str
    model_name: str = "gpt-4"
    ollama_model_name: str = "gpt-oss:20b"
    
    # RAG settings
    vector_db_path: str = "./chroma_db"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()