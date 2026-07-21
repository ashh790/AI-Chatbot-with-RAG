"""Load documents from docs/ into the Chroma vector store.

    python ingest.py               # ingest everything in docs/
    python ingest.py --reset       # wipe the store first
    python ingest.py --path X.pdf  # ingest one file or folder
    python ingest.py --stats       # how many chunks are stored

Supported: .pdf .txt .md .markdown .csv .json .log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatbot.config import Config  # noqa: E402
from chatbot.utils import chunk_text, vector_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.error("Could not open %s: %s", path.name, exc)
        return ""

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("%s page %d unreadable: %s", path.name, i + 1, exc)
    return "\n\n".join(pages)


def read_file(path: Path) -> str:
    if path.suffix.lower() in PDF_SUFFIXES:
        return read_pdf(path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.error("Could not read %s: %s", path.name, exc)
            return ""
    logger.error("Could not decode %s with any known encoding", path.name)
    return ""


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED else []
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def ingest_path(target: Path) -> tuple[int, int]:
    """Returns (files_ingested, chunks_stored)."""
    files = collect_files(target)
    if not files:
        logger.warning("No supported files found in %s", target)
        logger.info("Supported extensions: %s", ", ".join(sorted(SUPPORTED)))
        return 0, 0

    total_files = 0
    total_chunks = 0

    for path in files:
        raw = read_file(path)
        if not raw.strip():
            logger.warning("SKIP %s (no extractable text -- scanned PDF?)", path.name)
            continue

        chunks = chunk_text(raw)
        if not chunks:
            logger.warning("SKIP %s (nothing left after chunking)", path.name)
            continue

        try:
            rel = path.relative_to(target if target.is_dir() else target.parent)
        except ValueError:
            rel = path.name

        source = str(rel).replace("\\", "/")
        ids = [f"{source}::chunk-{i}" for i in range(len(chunks))]
        metas = [
            {"source": source, "chunk": i, "suffix": path.suffix.lower()}
            for i in range(len(chunks))
        ]

        try:
            stored = vector_store.add_documents(chunks, ids, metas)
        except Exception as exc:
            logger.error("FAIL %s: %s", path.name, exc)
            continue

        total_files += 1
        total_chunks += stored
        logger.info("OK   %s -> %d chunks", source, stored)

    return total_files, total_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store")
    parser.add_argument("--path", type=str, default=None, help="File or folder (default: docs/)")
    parser.add_argument("--reset", action="store_true", help="Clear the store first")
    parser.add_argument("--stats", action="store_true", help="Show store stats and exit")
    args = parser.parse_args()

    if not vector_store.available:
        logger.error("Vector store unavailable: %s", vector_store.error)
        logger.error("Try: pip install chromadb")
        return 1

    if args.stats:
        print(f"Collection : {Config.COLLECTION_NAME}")
        print(f"Location   : {Config.CHROMA_PATH}")
        print(f"Chunks     : {vector_store.count()}")
        return 0

    if args.reset:
        logger.info("Resetting collection...")
        vector_store.reset()

    target = Path(args.path).expanduser().resolve() if args.path else Config.DOCS_DIR
    if not target.exists():
        logger.error("Path does not exist: %s", target)
        if not args.path:
            target.mkdir(parents=True, exist_ok=True)
            logger.info("Created %s -- drop your PDFs/text files there and re-run.", target)
        return 1

    if not Config.has_valid_api_key():
        logger.warning(
            "No valid OPENAI_API_KEY -- falling back to Chroma's bundled "
            "embedding model, which downloads ~80MB on first use."
        )

    files, chunks = ingest_path(target)
    print()
    print(f"Ingested {files} file(s), {chunks} chunk(s).")
    print(f"Store now holds {vector_store.count()} chunk(s).")
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
