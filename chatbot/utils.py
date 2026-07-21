"""Vector store access + text chunking for the RAG layer.

Everything here fails soft: if chromadb is missing or the store can't be
opened, the app still runs -- retrieval just returns nothing instead of
raising AttributeError on a None collection.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from .config import Config

logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None
    logger.info("chromadb not installed -- falling back to built-in keyword index")


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Collapse runs of whitespace so chunk boundaries stay predictable."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Overlap matters: without it, a fact that straddles a chunk edge gets
    cut in half and neither chunk retrieves well.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Back off to the nearest sentence end, then word boundary.
        window = text[start:end]
        split_at = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if split_at < chunk_size * 0.5:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = chunk_size

        chunks.append(text[start : start + split_at + 1].strip())
        start += max(split_at + 1 - overlap, 1)

    return [c for c in chunks if c]


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------

class VectorStore:
    """Thin lazy wrapper around a persistent Chroma collection."""

    def __init__(self) -> None:
        self._collection = None
        self._fallback = None
        self._backend = "none"
        self._initialised = False
        self._error: str | None = None

    # -- internals ---------------------------------------------------------

    def _embedding_function(self):
        """Prefer OpenAI embeddings; fall back to Chroma's bundled model.

        The default fallback downloads an ONNX MiniLM on first use, which
        silently fails on an offline machine -- hence the preference order.
        """
        if not Config.uses_openai_embeddings():
            # Third-party providers (Groq etc.) have no embeddings endpoint --
            # use Chroma's local model, which needs no API key or quota.
            logger.info("Using Chroma's local embedding model (no API quota needed)")
            return None
        try:
            from chromadb.utils import embedding_functions

            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=Config.OPENAI_API_KEY,
                model_name=Config.EMBEDDING_MODEL,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenAI embeddings unavailable (%s); using default", exc)
            return None

    def _use_fallback(self, reason: str) -> None:
        """Switch to the dependency-free BM25 index."""
        from pathlib import Path

        from .fallback_store import FallbackStore

        path = Path(Config.CHROMA_PATH).parent / "fallback_index.json"
        self._fallback = FallbackStore(path)
        self._backend = "bm25"
        logger.warning(
            "chromadb unavailable (%s) -- using built-in BM25 index at %s. "
            "Keyword-based rather than semantic; `pip install chromadb` for "
            "embedding search.",
            reason,
            path,
        )

    def _ensure(self) -> None:
        if self._initialised:
            return
        self._initialised = True

        if chromadb is None:
            self._use_fallback("not installed")
            return

        try:
            from pathlib import Path

            Path(Config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            kwargs: dict[str, Any] = {"name": Config.COLLECTION_NAME}
            embed_fn = self._embedding_function()
            if embed_fn is not None:
                kwargs["embedding_function"] = embed_fn
            self._collection = client.get_or_create_collection(**kwargs)
            self._backend = "chromadb"
            logger.info("Vector store ready at %s", Config.CHROMA_PATH)
        except Exception as exc:  # pragma: no cover
            logger.error("Could not open Chroma: %s", exc)
            self._use_fallback(str(exc))

    # -- public API --------------------------------------------------------

    @property
    def available(self) -> bool:
        self._ensure()
        return self._collection is not None or self._fallback is not None

    @property
    def backend(self) -> str:
        """'chromadb' (semantic) or 'bm25' (keyword fallback)."""
        self._ensure()
        return self._backend

    @property
    def error(self) -> str | None:
        self._ensure()
        return self._error

    def count(self) -> int:
        if not self.available:
            return 0
        if self._fallback is not None:
            return self._fallback.count()
        try:
            return self._collection.count()
        except Exception:  # pragma: no cover
            return 0

    def add_documents(
        self,
        documents: Iterable[str],
        ids: Iterable[str],
        metadatas: Iterable[dict] | None = None,
    ) -> int:
        """Upsert chunks. Upsert (not add) so re-ingesting isn't an error."""
        if not self.available:
            raise RuntimeError(f"Vector store unavailable: {self._error}")
        if self._fallback is not None:
            return self._fallback.add_documents(documents, ids, metadatas)

        docs = list(documents)
        id_list = list(ids)
        if not docs:
            return 0
        if len(docs) != len(id_list):
            raise ValueError("documents and ids must be the same length")

        meta_list = list(metadatas) if metadatas is not None else [{} for _ in docs]
        # Chroma rejects empty metadata dicts on some versions.
        meta_list = [m or {"source": "unknown"} for m in meta_list]

        # Batch to stay under embedding-API request limits.
        batch = 100
        for i in range(0, len(docs), batch):
            self._collection.upsert(
                documents=docs[i : i + batch],
                ids=id_list[i : i + batch],
                metadatas=meta_list[i : i + batch],
            )
        return len(docs)

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        """Chunk a single document and store every chunk."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{doc_id}::chunk-{i}" for i in range(len(chunks))]
        metas = [{**(metadata or {}), "source": doc_id, "chunk": i} for i in range(len(chunks))]
        return self.add_documents(chunks, ids, metas)

    def query(self, query_text: str, n_results: int | None = None) -> list[dict]:
        """Return matching chunks as dicts. Never raises."""
        n_results = n_results or Config.RAG_RESULTS
        if not self.available or not (query_text or "").strip():
            return []
        if self._fallback is not None:
            return self._fallback.query(query_text, n_results)

        try:
            raw = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_results, max(self.count(), 1)),
            )
        except Exception as exc:  # pragma: no cover
            logger.error("RAG query failed: %s", exc)
            return []

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        out = []
        for i, doc in enumerate(docs):
            out.append(
                {
                    "text": doc,
                    "source": (metas[i] if i < len(metas) else {}).get("source", "unknown"),
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    def list_sources(self) -> list[dict]:
        """Every distinct document in the store, with its chunk count."""
        if not self.available or self.count() == 0:
            return []
        if self._fallback is not None:
            return self._fallback.list_sources()
        try:
            raw = self._collection.get(include=["metadatas"])
        except Exception as exc:  # pragma: no cover
            logger.error("Could not list sources: %s", exc)
            return []

        counts: dict[str, dict] = {}
        for meta in raw.get("metadatas") or []:
            if not meta:
                continue
            source = meta.get("source", "unknown")
            entry = counts.setdefault(source, {"source": source, "chunks": 0, "suffix": ""})
            entry["chunks"] += 1
            if meta.get("suffix"):
                entry["suffix"] = meta["suffix"]
        return sorted(counts.values(), key=lambda d: d["source"].lower())

    def get_document_text(self, source: str) -> str:
        """Reassemble one document's full text from its chunks, in order."""
        if not self.available:
            return ""
        if self._fallback is not None:
            return self._fallback.get_document_text(source)
        try:
            raw = self._collection.get(
                where={"source": source}, include=["documents", "metadatas"]
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Could not read %r: %s", source, exc)
            return ""

        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        pairs = [
            ((metas[i] or {}).get("chunk", i), docs[i]) for i in range(len(docs))
        ]
        pairs.sort(key=lambda p: p[0])
        return "\n\n".join(text for _, text in pairs)

    def delete_by_source(self, source: str) -> int:
        """Delete every chunk belonging to one document. Returns how many."""
        if not self.available:
            return 0
        if self._fallback is not None:
            return self._fallback.delete_by_source(source)
        try:
            existing = self._collection.get(where={"source": source}, include=[])
            ids = existing.get("ids") or []
            if not ids:
                return 0
            self._collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # pragma: no cover
            logger.error("Delete failed for %r: %s", source, exc)
            return 0

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._ensure()
        if self._fallback is not None:
            self._fallback.reset()
            return
        if self._collection is None or chromadb is None:
            return
        try:
            client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            client.delete_collection(name=Config.COLLECTION_NAME)
        except Exception as exc:  # pragma: no cover
            logger.warning("Reset failed: %s", exc)
        self._initialised = False
        self._collection = None


vector_store = VectorStore()


def query_rag(query_text: str, n_results: int | None = None) -> str:
    """Retrieval formatted for prompt injection. Returns '' when empty."""
    hits = vector_store.query(query_text, n_results)
    if not hits:
        return ""
    return "\n---\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
