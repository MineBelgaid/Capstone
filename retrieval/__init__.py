"""Local RAG layer: embeddings + persistent ChromaDB store."""

from .embeddings import get_embedder
from .store import KnowledgeStore

__all__ = ["get_embedder", "KnowledgeStore"]
