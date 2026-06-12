"""Movie Recommendation Microservice - App Configuration"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # --- Data & Model Paths ---
    data_dir: str = "data/ml-100k"
    model_dir: str = "models"
    db_path: str = "data/movies.db"

    # --- Pipeline ---
    top_n: int = 10
    cache_ttl_seconds: int = 86400  # 24 hours

    # --- SVD Default Hyperparameters ---
    svd_n_factors: int = 100
    svd_n_epochs: int = 20
    svd_lr: float = 0.005
    svd_reg: float = 0.02

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
