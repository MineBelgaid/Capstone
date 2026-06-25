"""Persistent ChromaDB knowledge store with metadata filtering.

Holds task history, meeting notes and sprint data as embedded chunks. On query
we do a similarity search, optionally filtered by metadata (e.g. doc_type,
sprint), and return the most relevant chunks for the ReAct loop to reason over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_DIR, settings
from retrieval.embeddings import get_embedder
from schemas import Task


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float


class KnowledgeStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = get_embedder()
        self._collection = self._client.get_or_create_collection(
            name=settings.retrieval.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- ingestion ------------------------------------------------------- #
    def add_meeting_note(self, source: str, text: str, meeting_date: str | None = None) -> int:
        return self._add_document(
            text=text,
            base_meta={"doc_type": "meeting_note", "source": source,
                       "meeting_date": meeting_date or ""},
            id_prefix=f"note::{source}",
        )

    def add_tasks(self, tasks: list[Task]) -> int:
        """Embed one short document per task for retrieval over task history."""
        if not tasks:
            return 0
        docs, metas, ids = [], [], []
        for t in tasks:
            docs.append(
                f"Task {t.task_id}: {t.title}. Status={t.status.value}. "
                f"Assignee={t.assignee or 'unassigned'}. Points={t.story_points or 0}. "
                f"Sprint={t.sprint or 'n/a'}. Due={t.due_date or 'n/a'}. "
                f"Labels={', '.join(t.labels) or 'none'}."
            )
            metas.append({
                "doc_type": "task", "source": "task_export", "task_id": t.task_id,
                "status": t.status.value, "assignee": t.assignee or "",
                "sprint": t.sprint or "",
            })
            ids.append(f"task::{t.task_id}")
        self._upsert(docs, metas, ids)
        return len(docs)

    def _add_document(self, text: str, base_meta: dict[str, Any], id_prefix: str) -> int:
        chunks = _chunk_text(
            text, settings.retrieval.chunk_size, settings.retrieval.chunk_overlap
        )
        if not chunks:
            return 0
        ids = [f"{id_prefix}::chunk{i}" for i in range(len(chunks))]
        metas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]
        self._upsert(chunks, metas, ids)
        return len(chunks)

    def _upsert(self, docs: list[str], metas: list[dict], ids: list[str]) -> None:
        embeddings = self._embedder(docs)
        self._collection.upsert(
            documents=docs, embeddings=embeddings, metadatas=metas, ids=ids
        )

    # ---- query ----------------------------------------------------------- #
    def query(
        self,
        text: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval.top_k
        q_emb = self._embedder([text])
        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(text=doc, metadata=meta or {}, distance=float(dist)))
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (used between eval scenarios)."""
        self._client.delete_collection(settings.retrieval.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.retrieval.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
