"""Central configuration, loaded once from the .env file next to app.py."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to THIS file, never the current working directory.
# This is what makes `python app.py`, `flask run`, and `python -m chatbot`
# all behave identically no matter where you launch them from.
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

ENV_FILE = BASE_DIR / ".env"

# override=True is deliberate. By default python-dotenv refuses to overwrite a
# variable that already exists in the OS environment, so a stale
# `set OPENAI_API_KEY=...` left over in a terminal silently beats the .env file
# and the app drops into demo mode while the file looks perfectly correct.
# The file on disk is the source of truth.
load_dotenv(ENV_FILE, override=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_PLACEHOLDER_KEYS = {
    "",
    "your_actual_openai_key",
    "your_openai_api_key",
    "sk-your-key-here",
    "placeholder",
    "changeme",
}


class Config:
    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    DEBUG = _env_bool("FLASK_DEBUG", default=False)
    PORT = _env_int("PORT", 5000)
    # Auto-restart on code edits. OFF by default: the reloader watches the
    # project tree, and uploads write into docs/ inside that tree, so it can
    # restart the server mid-request and reset the connection. Only turn this
    # on (FLASK_RELOAD=true) while actively editing the code.
    USE_RELOADER = _env_bool("FLASK_RELOAD", default=False)

    # --- LLM provider ---
    # Any OpenAI-protocol-compatible provider works. Leave BASE_URL empty for
    # OpenAI itself; set it to switch to a free provider (see .env.example).
    OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
    OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip()
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_TOOL_ROUNDS = _env_int("MAX_TOOL_ROUNDS", 5)

    # --- RAG / Chroma ---
    CHROMA_PATH = str(BASE_DIR / "vectorstore" / "chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
    DOCS_DIR = BASE_DIR / "docs"
    RAG_RESULTS = _env_int("RAG_RESULTS", 4)
    CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)

    # --- Memory ---
    MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 10)

    @classmethod
    def has_valid_api_key(cls) -> bool:
        """True for a key that looks real from ANY provider.

        Deliberately not 'startswith sk-': Groq keys begin gsk_, Gemini keys
        AIza, OpenRouter keys sk-or-. Only placeholders are rejected.
        """
        key = cls.OPENAI_API_KEY
        if key.lower() in _PLACEHOLDER_KEYS:
            return False
        return len(key) >= 20

    @classmethod
    def key_problem(cls) -> str | None:
        """Explain WHY the key was rejected, or None if it's fine.

        'Demo mode' with no reason attached is the least useful error message
        possible -- this turns it into something actionable.
        """
        key = cls.OPENAI_API_KEY
        if not ENV_FILE.exists():
            return f"No .env file found at {ENV_FILE}. Copy .env.example to .env."
        if not key:
            return (
                f"OPENAI_API_KEY is empty. Add it to {ENV_FILE} "
                "(no quotes, no spaces around the '=')."
            )
        if key.lower() in _PLACEHOLDER_KEYS:
            return f"OPENAI_API_KEY is still the placeholder value {key!r}."
        if len(key) < 20:
            return (
                f"OPENAI_API_KEY is only {len(key)} characters -- it looks "
                "truncated. Copy the whole key."
            )
        return None

    @classmethod
    def provider_name(cls) -> str:
        """Best-effort label for whichever endpoint we're pointed at."""
        url = cls.OPENAI_BASE_URL.lower()
        if not url:
            return "OpenAI"
        for fragment, label in (
            ("groq.com", "Groq"),
            ("generativelanguage.googleapis.com", "Google Gemini"),
            ("openrouter.ai", "OpenRouter"),
            ("mistral.ai", "Mistral"),
            ("together.xyz", "Together AI"),
            ("localhost", "local model"),
            ("127.0.0.1", "local model"),
        ):
            if fragment in url:
                return label
        return url

    @classmethod
    def uses_openai_embeddings(cls) -> bool:
        """Only OpenAI's own endpoint serves text-embedding-3-*.

        Groq has no embeddings endpoint at all, so pointing Chroma at it would
        fail every ingest. When we're on a third-party endpoint, fall back to
        Chroma's local model instead.
        """
        return cls.has_valid_api_key() and not cls.OPENAI_BASE_URL
