"""Tests for web GUI (F8)."""

import pytest

from openstat.web.session_manager import SessionManager


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        sid = mgr.create()
        assert sid is not None
        assert mgr.active_count == 1

    def test_get_session(self):
        mgr = SessionManager()
        sid = mgr.create()
        session = mgr.get(sid)
        assert session is not None

    def test_get_nonexistent(self):
        mgr = SessionManager()
        session = mgr.get("nonexistent")
        assert session is None

    def test_get_or_create(self):
        mgr = SessionManager()
        s1 = mgr.get_or_create("test1")
        s2 = mgr.get_or_create("test1")
        assert s1 is s2

    def test_remove_session(self):
        mgr = SessionManager()
        sid = mgr.create()
        mgr.remove(sid)
        assert mgr.active_count == 0

    def test_max_sessions(self):
        mgr = SessionManager(max_sessions=3)
        for _ in range(5):
            mgr.create()
        assert mgr.active_count <= 3

    def test_session_isolation(self):
        """Sessions are independent."""
        import polars as pl
        mgr = SessionManager()
        s1 = mgr.get_or_create("a")
        s2 = mgr.get_or_create("b")
        s1.df = pl.DataFrame({"x": [1]})
        assert s2.df is None


try:
    from fastapi.testclient import TestClient
    from openstat.web.app import app as web_app
    HAS_FASTAPI = web_app is not None
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestWebAPI:
    def test_index(self):
        client = TestClient(web_app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_create_session_api(self):
        client = TestClient(web_app)
        resp = client.post("/api/session")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_status(self):
        client = TestClient(web_app)
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
