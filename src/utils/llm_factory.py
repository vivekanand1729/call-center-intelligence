"""Multi-provider LLM factory: OpenAI, Gemini, Groq."""
from __future__ import annotations

import os
from typing import Any


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Any:
    from src.utils.config import load_config
    cfg = load_config()

    provider = provider or cfg.llm_provider
    timeout = timeout or cfg.llm_timeout_seconds

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=cfg.openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=cfg.google_api_key or os.getenv("GOOGLE_API_KEY"),
            timeout=timeout,
        )
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model or "llama-3.3-70b-versatile",
                groq_api_key=cfg.groq_api_key or os.getenv("GROQ_API_KEY"),
                timeout=timeout,
            )
        except ImportError as e:
            raise ImportError("langchain-groq is not installed. Run: pip install langchain-groq") from e
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: openai, gemini, groq")


# Allow Optional import without circular
from typing import Optional
