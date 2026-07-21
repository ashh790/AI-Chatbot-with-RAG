"""HTTP routes."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, make_response, render_template, request

from .config import Config
from .documents import DocumentError, delete_document, save_and_ingest
from .memory import memory_store
from .services import UpstreamError, generate_chat_response, is_live
from .utils import vector_store

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint("chatbot", __name__)

MAX_MESSAGE_LENGTH = 8000


@chatbot_bp.route("/", methods=["GET"])
def index():
    # Explicit no-store: without it Chrome will happily serve a cached copy
    # of the page after you've edited the template, which looks exactly like
    # "my changes aren't applying".
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@chatbot_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "llm_connected": is_live(),
            "demo_mode": not is_live(),
            "provider": Config.provider_name(),
            "model": Config.MODEL_NAME,
            "key_problem": Config.key_problem(),
            "vector_store_available": vector_store.available,
            "vector_store_error": vector_store.error,
            "vector_store_backend": vector_store.backend,
            "documents_indexed": vector_store.count(),
            "active_sessions": memory_store.session_count(),
        }
    )


@chatbot_bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = str(data.get("session_id") or "default_user")[:128]

    if not message:
        return jsonify({"error": "Message is required"}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message exceeds {MAX_MESSAGE_LENGTH} characters"}), 413

    try:
        result = generate_chat_response(session_id, message)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except UpstreamError as exc:
        logger.error("Upstream failure (%s): %s", exc.kind, exc)
        return jsonify({"error": str(exc), "kind": exc.kind}), exc.status
    except RuntimeError as exc:
        logger.error("Upstream failure: %s", exc)
        return jsonify({"error": str(exc), "kind": "unknown"}), 502
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled error in /api/chat")
        return jsonify({"error": f"Internal error: {exc}"}), 500

    return jsonify(
        {
            "response": result["reply"],
            "session_id": result["session_id"],
            "used_rag": result["used_rag"],
            "demo_mode": result["demo_mode"],
        }
    )


@chatbot_bp.route("/api/reset", methods=["POST"])
def reset_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    memory_store.clear(session_id)
    scope = f"session '{session_id}'" if session_id else "all sessions"
    return jsonify({"status": "cleared", "scope": scope})


@chatbot_bp.route("/api/ingest", methods=["POST"])
def ingest_text():
    """Add a document to the knowledge base without touching the CLI."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    doc_id = (data.get("doc_id") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400
    if not doc_id:
        return jsonify({"error": "Field 'doc_id' is required"}), 400
    if not vector_store.available:
        return jsonify({"error": f"Vector store unavailable: {vector_store.error}"}), 503

    try:
        stored = vector_store.add_document(doc_id, text, data.get("metadata") or {})
    except Exception as exc:
        logger.exception("Ingest failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", "doc_id": doc_id, "chunks_added": stored,
                    "total_chunks": vector_store.count()})


@chatbot_bp.route("/api/upload", methods=["POST"])
def upload_documents():
    """Accept one or more files, save them under docs/, and index them."""
    files = request.files.getlist("files") or request.files.getlist("file")
    files = [f for f in files if f and f.filename]

    if not files:
        return jsonify({"error": "No files were uploaded"}), 400
    if not vector_store.available:
        return jsonify(
            {"error": f"Vector store unavailable: {vector_store.error}. "
                      "Run: pip install chromadb"}
        ), 503

    uploaded, failed = [], []
    for storage in files:
        logger.info("Upload received: %s", storage.filename)
        try:
            result = save_and_ingest(storage)
            logger.info(
                "Indexed %s -> %d chunk(s)", result["saved_as"], result["chunks"]
            )
            uploaded.append(result)
        except DocumentError as exc:
            logger.warning("Rejected %s: %s", storage.filename, exc)
            failed.append({"filename": storage.filename, "error": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("Upload failed for %s", storage.filename)
            failed.append({"filename": storage.filename, "error": str(exc)})

    status = 200 if uploaded else 400
    return jsonify(
        {
            "uploaded": uploaded,
            "failed": failed,
            "total_chunks": vector_store.count(),
        }
    ), status


@chatbot_bp.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify(
        {
            "documents": vector_store.list_sources(),
            "total_chunks": vector_store.count(),
            "vector_store_available": vector_store.available,
        }
    )


@chatbot_bp.route("/api/documents/<path:source>", methods=["DELETE"])
def remove_document(source):
    if not vector_store.available:
        return jsonify({"error": f"Vector store unavailable: {vector_store.error}"}), 503
    try:
        result = delete_document(source)
    except Exception as exc:  # pragma: no cover
        logger.exception("Delete failed")
        return jsonify({"error": str(exc)}), 500

    if result["chunks_removed"] == 0 and not result["file_removed"]:
        return jsonify({"error": f"No document named {source!r}"}), 404

    result["total_chunks"] = vector_store.count()
    return jsonify(result)


@chatbot_bp.errorhandler(404)
def not_found(_exc):
    return jsonify({"error": "Not found"}), 404


@chatbot_bp.errorhandler(413)
def too_large(_exc):
    from .documents import MAX_UPLOAD_BYTES

    return jsonify(
        {
            "error": f"Upload too large. Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                     "per file. Try fewer or smaller files."
        }
    ), 413
