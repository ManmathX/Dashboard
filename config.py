"""
Configuration management for LLM Evaluation Framework.
Loads environment variables and provides centralized settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    
    # Database Configuration
    # Database Configuration - REMOVED
    # mongodb_uri: str = "mongodb://localhost:27017"
    # mongodb_database: str = "llm_evaluation"
    
    # Judge LLM Configuration
    judge_model_provider: str = "groq"  # openai, anthropic, groq, gemini, or perplexity
    judge_model_name: str = "llama-3.3-70b-versatile"
    judge_temperature: float = 0.1  # Low temperature for consistent evaluation
    judge_max_tokens: int = 4000
    
    # Target LLM Default Configuration
    default_target_provider: str = "openai"
    default_target_model: str = "gpt-3.5-turbo"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Security
    cors_origins: str = "http://localhost:3000,http://localhost:8000,null"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def validate_api_keys(self) -> None:
        """Validate that required API keys are present."""
        if self.judge_model_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI as judge")
        if self.judge_model_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic as judge")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "groq":
            return self.groq_api_key
        elif provider == "gemini":
            return self.gemini_api_key
        elif provider == "perplexity":
            return self.perplexity_api_key
        return None


# Global settings instance
settings = Settings()
