"""FastAPI application for OpenStat web GUI."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from openstat.web.session_manager import SessionManager
from openstat.logging_config import get_logger

log = get_logger("web")

if HAS_FASTAPI:
    app = FastAPI(title="OpenStat Web", version="0.3.0")
    sessions = SessionManager()

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main HTML page."""
        html_path = static_dir / "index.html"
        if html_path.exists():
            return html_path.read_text()
        return "<h1>OpenStat Web</h1><p>Static files not found.</p>"

    @app.post("/api/session")
    async def create_session():
        """Create a new analysis session."""
        session_id = sessions.create()
        return {"session_id": session_id}

    @app.post("/api/upload/{session_id}")
    async def upload_file(session_id: str, file: UploadFile = File(...)):
        """Upload a data file to a session."""
        session = sessions.get(session_id)
        if session is None:
            return {"error": "Session not found"}

        # Save to temp file
        suffix = Path(file.filename).suffix if file.filename else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load into session
        from openstat.repl import _dispatch
        result = _dispatch(session, f"load {tmp_path}")
        return {
            "result": result,
            "shape": session.shape_str,
            "filename": file.filename,
        }

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket REPL for interactive commands."""
        await websocket.accept()
        session = sessions.get_or_create(session_id)
        log.info("WebSocket connected: %s", session_id)

        try:
            while True:
                command = await websocket.receive_text()
                from openstat.repl import _dispatch
                result = _dispatch(session, command)

                if result == "__QUIT__":
                    await websocket.send_json({
                        "type": "quit",
                        "content": "Session ended.",
                    })
                    break

                # Check for new plot files
                plot_data = None
                if session.plot_paths:
                    last_plot = session.plot_paths[-1]
                    plot_path = Path(last_plot)
                    if plot_path.exists():
                        with open(plot_path, "rb") as f:
                            plot_data = base64.b64encode(f.read()).decode()

                await websocket.send_json({
                    "type": "result",
                    "content": result or "",
                    "shape": session.shape_str,
                    "plot": plot_data,
                })
        except WebSocketDisconnect:
            log.info("WebSocket disconnected: %s", session_id)
        except Exception as e:
            log.error("WebSocket error: %s", e)

    @app.get("/api/status")
    async def status():
        """Server status."""
        return {
            "active_sessions": sessions.active_count,
            "status": "running",
        }
else:
    app = None
