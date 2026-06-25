"""Local, free embedding functions.

Primary: Ollama ``nomic-embed-text`` (no API key, runs locally).
Fallback: sentence-transformers ``all-MiniLM-L6-v2`` (local, free) when Ollama or
the embed model isn't available -- so eval/dev never blocks on Ollama being up.

Both expose the same minimal interface ChromaDB expects: a callable taking a list
of strings and returning a list of float vectors. We deliberately never use a
paid embedding provider.
"""

from __future__ import annotations

from typing import Protocol

from config import settings


class Embedder(Protocol):
    name: str

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class _OllamaEmbedder:
    def __init__(self, model: str, base_url: str) -> None:
        import ollama  # local import so import works without the package at import time

        self._client = ollama.Client(host=base_url)
        self.model = model
        self.name = f"ollama:{model}"
        # probe once so we can fail fast and fall back
        self._client.embeddings(model=model, prompt="ping")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            resp = self._client.embeddings(model=self.model, prompt=t)
            out.append(list(resp["embedding"]))
        return out


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def get_embedder() -> Embedder:
    """Return a working local embedder, preferring Ollama and falling back to ST."""
    try:
        return _OllamaEmbedder(
            model=settings.embed_model_name(),
            base_url=settings.ollama.base_url,
        )
    except Exception as exc:  # noqa: BLE001 - any failure -> fall back to ST
        print(f"[embeddings] Ollama unavailable ({exc!r}); using sentence-transformers")
        return _SentenceTransformerEmbedder(settings.retrieval.st_fallback_model)
