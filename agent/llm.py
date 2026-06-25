"""LLM factory honoring the config backend switch.

Returns a LangChain chat model. ``LLM_BACKEND=ollama`` (default) gives the local
Qwen/Llama model; ``LLM_BACKEND=claude`` gives Claude Sonnet for the final demo.
Swapping is a one-line config/env change -- callers never branch on backend.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings


@lru_cache(maxsize=2)
def get_chat_model():
    backend = settings.backend
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=settings.ollama.temperature,
        )
    if backend == "claude":
        from langchain_anthropic import ChatAnthropic

        if not settings.claude.api_key:
            raise RuntimeError(
                "LLM_BACKEND=claude but ANTHROPIC_API_KEY is unset. "
                "Set it only for the final demo."
            )
        return ChatAnthropic(
            model=settings.claude.model,
            temperature=settings.claude.temperature,
            api_key=settings.claude.api_key,
        )
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'ollama' or 'claude')")
