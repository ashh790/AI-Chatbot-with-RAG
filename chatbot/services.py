"""Chat orchestration: history + retrieval + tool-calling loop."""

from __future__ import annotations

import logging

from .config import Config
from .memory import memory_store
from .prompts import SYSTEM_PROMPT, build_system_prompt
from .tools import TOOLS, execute_tool_call
from .utils import query_rag, vector_store

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
    logger.warning("openai package not installed -- demo mode only")


def _build_client():
    """Build the client, honouring OPENAI_BASE_URL for non-OpenAI providers."""
    if OpenAI is None or not Config.has_valid_api_key():
        return None
    kwargs = {"api_key": Config.OPENAI_API_KEY}
    if Config.OPENAI_BASE_URL:
        kwargs["base_url"] = Config.OPENAI_BASE_URL
    try:
        c = OpenAI(**kwargs)
        logger.info("LLM provider: %s (model %s)", Config.provider_name(), Config.MODEL_NAME)
        return c
    except Exception as exc:  # pragma: no cover
        logger.error("Could not construct client: %s", exc)
        return None


client = _build_client()


def is_live() -> bool:
    """False means we're echoing in demo mode rather than calling the API."""
    return client is not None


class UpstreamError(RuntimeError):
    """An OpenAI failure, classified so the UI can say something useful."""

    def __init__(self, message: str, kind: str, status: int = 502) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


def classify_openai_error(exc: Exception) -> UpstreamError:
    """Turn a raw SDK exception into a plain-language explanation.

    A bare 'OpenAI request failed: Error code: 429' tells the user nothing
    actionable. These are the failures that actually happen in practice.
    """
    name = type(exc).__name__
    text = str(exc)
    status = getattr(exc, "status_code", None)

    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return UpstreamError(
            "Your OpenAI account has no available credit, so the API rejected "
            "the request. The code is working -- this is a billing issue. Add "
            "credit at https://platform.openai.com/settings/organization/billing "
            "then try again.",
            kind="insufficient_quota",
            status=402,
        )

    if name == "RateLimitError" or status == 429:
        return UpstreamError(
            "OpenAI is rate-limiting this key. Wait a few seconds and retry.",
            kind="rate_limit",
            status=429,
        )

    if name == "AuthenticationError" or status == 401:
        return UpstreamError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env -- "
            "the key may be revoked, or copied incompletely.",
            kind="bad_key",
            status=401,
        )

    if name == "PermissionDeniedError" or status == 403:
        return UpstreamError(
            f"This key isn't allowed to use model '{Config.MODEL_NAME}'. "
            "Try MODEL_NAME=gpt-4o-mini in your .env.",
            kind="forbidden",
            status=403,
        )

    if name == "NotFoundError" or status == 404:
        return UpstreamError(
            f"Model '{Config.MODEL_NAME}' does not exist or isn't available to "
            "this account. Check MODEL_NAME in your .env.",
            kind="bad_model",
            status=404,
        )

    if name in {"APIConnectionError", "APITimeoutError"}:
        return UpstreamError(
            "Could not reach api.openai.com. Check your internet connection, "
            "VPN, or firewall.",
            kind="network",
            status=504,
        )

    return UpstreamError(f"OpenAI request failed: {text}", kind="unknown", status=502)


def _run_tool_rounds(messages: list) -> str:
    """Call the model, resolving tool calls until it returns prose.

    The original bug: only tool_calls[0] was answered. OpenAI returns a
    400 if an assistant message with tool_calls isn't followed by a tool
    message for EVERY tool_call_id, so parallel calls broke the request.
    """
    for round_num in range(Config.MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=messages,
            tools=TOOLS or None,
            tool_choice="auto" if TOOLS else None,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return (msg.content or "").strip() or "(empty response)"

        # Echo the assistant turn back via model_dump rather than rebuilding it
        # by hand. Hand-building drops provider-specific fields -- notably
        # Gemini's `thought_signature` (carried in tool_calls[].extra_content),
        # without which Gemini rejects round 2 with:
        #   400 "Function call is missing a thought_signature"
        # model_dump keeps those fields and still yields plain JSON-safe dicts.
        try:
            messages.append(msg.model_dump(exclude_none=True))
        except AttributeError:  # non-pydantic client (tests, older SDKs)
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

        # One tool message per call id -- all of them, in order.
        for tc in tool_calls:
            logger.info("Round %d: tool %s", round_num + 1, tc.function.name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": execute_tool_call(tc),
                }
            )

    logger.warning("Hit MAX_TOOL_ROUNDS (%d)", Config.MAX_TOOL_ROUNDS)
    final = client.chat.completions.create(model=Config.MODEL_NAME, messages=messages)
    return (final.choices[0].message.content or "").strip() or "(empty response)"


def generate_chat_response(session_id: str, user_message: str) -> dict:
    """Produce a reply. Returns a dict so routes can surface RAG metadata."""
    user_message = (user_message or "").strip()
    if not user_message:
        raise ValueError("Message cannot be empty")

    rag_context = query_rag(user_message)
    history = memory_store.get_history(session_id)

    # Tell the model what's indexed, not just what matched. Without this it
    # denies that uploads exist whenever keyword retrieval comes back empty.
    try:
        documents = vector_store.list_sources()
    except Exception:  # pragma: no cover
        documents = []

    messages = [
        {"role": "system", "content": build_system_prompt(rag_context, documents)},
        *history,
        {"role": "user", "content": user_message},
    ]

    if client is None:
        why = Config.key_problem() or (
            "The 'openai' package isn't installed (pip install openai)."
            if OpenAI is None
            else "The API client could not be constructed."
        )
        reply = (
            f"Demo mode -- I'm not connected to a model, so I can't answer.\n\n"
            f"Reason: {why}\n\n"
            f"After fixing it, restart the server (Ctrl+C, then python app.py). "
            f"Config is read once at startup, so edits to .env need a restart."
        )
        used_rag = False
    else:
        try:
            reply = _run_tool_rounds(messages)
            used_rag = bool(rag_context)
        except Exception as exc:
            err = classify_openai_error(exc)
            logger.error("Chat failed (%s): %s", err.kind, exc)
            # Don't poison history with a failed turn.
            raise err from exc

    memory_store.add_message(session_id, "user", user_message)
    memory_store.add_message(session_id, "assistant", reply)

    return {
        "reply": reply,
        "session_id": session_id,
        "used_rag": used_rag,
        "demo_mode": client is None,
    }


__all__ = [
    "generate_chat_response",
    "is_live",
    "classify_openai_error",
    "UpstreamError",
    "SYSTEM_PROMPT",
]
