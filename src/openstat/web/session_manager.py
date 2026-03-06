"""Multi-user session management with TTL cleanup."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from openstat.session import Session
from openstat.logging_config import get_logger

log = get_logger("web.sessions")


class SessionManager:
    """Manages multiple user sessions with automatic cleanup."""

    def __init__(self, max_sessions: int = 100, ttl_minutes: int = 60) -> None:
        self._sessions: dict[str, Session] = {}
        self._last_access: dict[str, datetime] = {}
        self._max_sessions = max_sessions
        self._ttl = timedelta(minutes=ttl_minutes)

    def create(self) -> str:
        """Create a new session, return its ID."""
        self._cleanup()
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = Session()
        self._last_access[session_id] = datetime.now()
        log.info("Created session: %s", session_id)
        return session_id

    def get(self, session_id: str) -> Session | None:
        """Get an existing session by ID."""
        session = self._sessions.get(session_id)
        if session is not None:
            self._last_access[session_id] = datetime.now()
        return session

    def get_or_create(self, session_id: str) -> Session:
        """Get existing session or create new one."""
        session = self.get(session_id)
        if session is None:
            self._sessions[session_id] = Session()
            self._last_access[session_id] = datetime.now()
            log.info("Created session: %s", session_id)
            session = self._sessions[session_id]
        return session

    def remove(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def _cleanup(self) -> None:
        """Remove expired sessions and enforce max count."""
        now = datetime.now()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > self._ttl
        ]
        for sid in expired:
            self.remove(sid)
            log.info("Expired session: %s", sid)

        # If still over limit, remove oldest
        while len(self._sessions) >= self._max_sessions:
            oldest = min(self._last_access, key=self._last_access.get)
            self.remove(oldest)
            log.info("Evicted session: %s", oldest)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
