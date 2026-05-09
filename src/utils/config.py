"""Configuration loader using python-dotenv."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class Config:
    llm_provider: str = "openai"
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    whisper_model_size: str = "tiny"
    confidence_threshold: float = 0.4
    low_confidence_halt_ratio: float = 0.5
    db_path: str = "data/calls.db"
    db_encryption_key: str = ""
    max_retries_per_node: int = 3
    llm_timeout_seconds: int = 60
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "call-center-intelligence"


def load_config() -> Config:
    return Config(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny"),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.4")),
        low_confidence_halt_ratio=float(os.getenv("LOW_CONFIDENCE_HALT_RATIO", "0.5")),
        db_path=os.getenv("DB_PATH", "/tmp/calls.db" if os.getenv("STREAMLIT_SERVER_PORT") else "data/calls.db"),
        db_encryption_key=os.getenv("DB_ENCRYPTION_KEY", ""),
        max_retries_per_node=int(os.getenv("MAX_RETRIES_PER_NODE", "3")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "call-center-intelligence"),
    )
