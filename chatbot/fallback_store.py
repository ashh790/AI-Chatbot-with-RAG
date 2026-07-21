"""A dependency-free document index, used when chromadb isn't available.

chromadb is a large install that fails on plenty of machines (native build
steps, Python version mismatches, corporate proxies). RAG shouldn't be gated
behind it, so this provides the same interface backed by BM25 ranking over a
JSON file.

BM25 is lexical, not semantic: it matches words rather than meaning, so it
won't connect "car" to "automobile" the way embeddings would. For question
answering over your own documents -- where questions usually reuse the
document's own vocabulary -- it performs well and needs nothing but the
standard library.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# BM25 tuning. k1 controls term-frequency saturation, b controls how much
# document length is penalised. These are the standard defaults.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "what", "which", "who", "how", "when", "where", "do", "does",
    "did", "can", "could", "would", "should", "i", "you", "we", "my", "me",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, drop stopwords and 1-char tokens."""
    return [
        t for t in _TOKEN.findall((text or "").lower())
        if len(t) > 1 and t not in STOPWORDS
    ]


class FallbackStore:
    """BM25 index over chunks, persisted as JSON. Thread-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._docs: dict[str, dict] = {}   # id -> {text, metadata, tokens}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for doc_id, entry in raw.get("documents", {}).items():
                self._docs[doc_id] = {
                    "text": entry["text"],
                    "metadata": entry.get("metadata", {}),
                    "tokens": entry.get("tokens") or tokenize(entry["text"]),
                }
            logger.info("Fallback store loaded %d chunk(s)", len(self._docs))
        except Exception as exc:  # pragma: no cover
            logger.error("Could not read %s: %s -- starting empty", self.path, exc)
            self._docs = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "documents": {
                doc_id: {
                    "text": e["text"],
                    "metadata": e["metadata"],
                    "tokens": e["tokens"],
                }
                for doc_id, e in self._docs.items()
            },
        }
        # Write to a temp file then replace, so an interrupted save can't
        # leave a truncated index behind.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)

    # -- interface mirroring VectorStore -----------------------------------

    @property
    def available(self) -> bool:
        return True

    @property
    def error(self) -> str | None:
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._docs)

    def add_documents(self, documents, ids, metadatas=None) -> int:
        docs = list(documents)
        id_list = list(ids)
        if not docs:
            return 0
        if len(docs) != len(id_list):
            raise ValueError("documents and ids must be the same length")
        metas = list(metadatas) if metadatas is not None else [{}] * len(docs)

        with self._lock:
            for text, doc_id, meta in zip(docs, id_list, metas):
                self._docs[doc_id] = {
                    "text": text,
                    "metadata": meta or {"source": "unknown"},
                    "tokens": tokenize(text),
                }
            self._save()
        return len(docs)

    def query(self, query_text: str, n_results: int = 4) -> list[dict]:
        terms = tokenize(query_text)
        with self._lock:
            if not self._docs or not terms:
                return []
            docs = list(self._docs.items())

        n_docs = len(docs)
        avg_len = sum(len(e["tokens"]) for _, e in docs) / n_docs or 1.0

        # Document frequency per query term.
        df = Counter()
        for _, entry in docs:
            seen = set(entry["tokens"])
            for term in set(terms):
                if term in seen:
                    df[term] += 1

        scored = []
        for doc_id, entry in docs:
            tf = Counter(entry["tokens"])
            dl = len(entry["tokens"]) or 1
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                # BM25 idf, +1 inside the log keeps it non-negative.
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avg_len))
            if score > 0:
                scored.append((score, doc_id, entry))

        scored.sort(key=lambda t: -t[0])
        top = scored[: max(1, n_results)]
        if not top:
            return []

        best = top[0][0] or 1.0
        return [
            {
                "text": entry["text"],
                "source": entry["metadata"].get("source", "unknown"),
                # Present as a pseudo-distance so callers see the same shape
                # they'd get from Chroma (lower = closer).
                "distance": round(1.0 - score / best, 4),
            }
            for score, _, entry in top
        ]

    def list_sources(self) -> list[dict]:
        with self._lock:
            counts: dict[str, dict] = {}
            for entry in self._docs.values():
                meta = entry["metadata"]
                source = meta.get("source", "unknown")
                rec = counts.setdefault(
                    source, {"source": source, "chunks": 0, "suffix": ""}
                )
                rec["chunks"] += 1
                if meta.get("suffix"):
                    rec["suffix"] = meta["suffix"]
        return sorted(counts.values(), key=lambda d: d["source"].lower())

    def get_document_text(self, source: str) -> str:
        """Reassemble one document's full text from its chunks, in order."""
        with self._lock:
            pairs = [
                (e["metadata"].get("chunk", 0), e["text"])
                for e in self._docs.values()
                if e["metadata"].get("source") == source
            ]
        pairs.sort(key=lambda p: p[0])
        return "\n\n".join(text for _, text in pairs)

    def delete_by_source(self, source: str) -> int:
        with self._lock:
            doomed = [
                doc_id for doc_id, e in self._docs.items()
                if e["metadata"].get("source") == source
            ]
            for doc_id in doomed:
                del self._docs[doc_id]
            if doomed:
                self._save()
        return len(doomed)

    def reset(self) -> None:
        with self._lock:
            self._docs.clear()
            self._save()
