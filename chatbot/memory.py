"""In-process conversation memory, keyed by session id.

Note: this lives in the Flask process's RAM. It resets on restart and is not
shared across workers -- swap for Redis or a database before running more
than one worker in production.
"""

from __future__ import annotations

import threading

from .config import Config


class ChatMemory:
    def __init__(self, max_messages: int | None = None) -> None:
        self._sessions: dict[str, list[dict]] = {}
        self._max = max_messages or Config.MAX_HISTORY_MESSAGES
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[dict]:
        """Return a copy, so callers can't mutate stored history by accident."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            if len(history) > self._max:
                # Trim in whole user/assistant pairs so the log never starts
                # on an orphaned assistant turn.
                excess = len(history) - self._max
                self._sessions[session_id] = history[excess + (excess % 2) :]

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


memory_store = ChatMemory()
