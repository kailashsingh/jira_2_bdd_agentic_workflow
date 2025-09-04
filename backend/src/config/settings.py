from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Jira settings
    jira_url: str
    jira_email: str
    jira_api_token: str
    
    # GitHub settings
    github_token: str
    github_repo: str
    github_branch_prefix: str = "feature/auto-bdd"
    
    # LLM settings
    openai_api_key: str
    model_name: str = "gpt-4"
    
    # RAG settings
    vector_db_path: str = "./chroma_db"
    
    class Config:
        env_file = ".env"

settings = Settings()