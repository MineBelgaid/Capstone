"""Central configuration for the AI Project Workflow Intelligence Agent.

The single most important thing here is ``LLM_BACKEND``: it switches the whole
system between the local development model (Ollama) and the Claude API used only
for the final live demo. Swapping backends is meant to be a one-line change here
(or one env var), never a refactor.

All values can be overridden with environment variables so nothing secret is
committed to the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
CHROMA_DIR = DATA_DIR / "chroma"          # persistent vector store lives here
SQLITE_PATH = DATA_DIR / "app.db"         # local review/approval state

for _p in (DATA_DIR, SYNTHETIC_DIR, CHROMA_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# LLM backend switch  --  the heart of the "swap to Claude for demo" rule
# --------------------------------------------------------------------------- #
# "ollama"  -> fully local, free, no API key (development + eval)
# "claude"  -> Claude API, used ONLY for the final demo
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()


@dataclass(frozen=True)
class OllamaConfig:
    """Local development backend."""

    # qwen2.5:14b is the preferred default (better tool-use / reasoning).
    # Override to "llama3.1:8b" on RAM-constrained machines (<16GB):
    #   export OLLAMA_MODEL=llama3.1:8b
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    # Local embedding model (free). Falls back to sentence-transformers if the
    # Ollama embedding model is unavailable (see retrieval/embeddings.py).
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass(frozen=True)
class ClaudeConfig:
    """Final-demo backend. Costs a few dollars total -- demo runs only."""

    model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    temperature: float = float(os.getenv("CLAUDE_TEMPERATURE", "0.1"))
    # Even on the Claude demo we keep embeddings LOCAL (no paid embeddings).
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass(frozen=True)
class RetrievalConfig:
    collection_name: str = "project_knowledge"
    chunk_size: int = 800          # characters
    chunk_overlap: int = 120
    top_k: int = 5
    # sentence-transformers fallback model (local, free)
    st_fallback_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class AgentConfig:
    max_react_steps: int = 8          # hard cap on reason->act->observe loops
    max_validation_retries: int = 2   # retries when Pydantic validation fails
    workflow_timeout_s: int = 60      # success criterion: < 60s per scenario
    # ReAct implementation:
    #   "custom"   -> explicit constrained reason/act/observe loop (robust on
    #                 local models; recommended for Ollama dev)
    #   "prebuilt" -> LangGraph create_react_agent native tool calling
    #                 (best with Claude for the final demo)
    react_mode: str = os.getenv("REACT_MODE", "custom").lower()
    # Reflection / self-critique: after the loop produces a final answer, the
    # model re-checks every claim against the exact tool observations and revises
    # any unsupported/contradicted statement. Reduces hallucination in the
    # narrative (numbers/risks stay tied to the deterministic tool outputs).
    reflection_enabled: bool = os.getenv("REACT_REFLECTION", "true").lower() == "true"


@dataclass(frozen=True)
class LangSmithConfig:
    enabled: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project: str = os.getenv("LANGCHAIN_PROJECT", "pwi-agent")
    # API key read from LANGCHAIN_API_KEY env var by langchain itself.


@dataclass(frozen=True)
class Settings:
    backend: str = LLM_BACKEND
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)

    # Hard design rule from the brief: never act on an external system without
    # explicit human approval in the dashboard. Tools consult this flag.
    require_human_approval: bool = True

    def active_model_name(self) -> str:
        return self.claude.model if self.backend == "claude" else self.ollama.model

    def embed_model_name(self) -> str:
        cfg = self.claude if self.backend == "claude" else self.ollama
        return cfg.embed_model


settings = Settings()


if __name__ == "__main__":
    print(f"LLM backend      : {settings.backend}")
    print(f"Active model     : {settings.active_model_name()}")
    print(f"Embedding model  : {settings.embed_model_name()}")
    print(f"Chroma dir       : {CHROMA_DIR}")
    print(f"Human approval   : {settings.require_human_approval}")
