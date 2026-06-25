"""LangGraph single-agent ReAct system: LLM factory, tools, graph, validation.

Imports are lazy: ``agent.analytics`` (pure-Python KPIs/risks) and the eval
harness can be used with NO langchain/langgraph installed. The LLM-dependent
symbols (``get_chat_model``, ``build_agent``, ``run_agent``) are resolved on first
access via module ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_chat_model", "build_agent", "run_agent"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute loading
    if name == "get_chat_model":
        from .llm import get_chat_model

        return get_chat_model
    if name in {"build_agent", "run_agent"}:
        from . import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
