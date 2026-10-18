"""
HY Memory - Lightweight HTTP Server

Zero-dependency HTTP server (stdlib only) that wraps HyMemoryClient
for local use by the OpenClaw plugin.

Usage:
    python -m hy_memory.server                    # default port 19527
    python -m hy_memory.server --port 19528
    HY_MEMORY_SERVER_PORT=19528 python -m hy_memory.server

API Endpoints:
    POST /api/v1/add          — Write memory
    POST /api/v1/search       — Search memories
    POST /api/v1/user_profile — Get assembled user profile
    GET  /api/v1/memories/:id — Get single memory
    POST /api/v1/list         — List memories
    PUT  /api/v1/memories/:id — Update memory
    DELETE /api/v1/memories/:id — Delete memory
    POST /api/v1/delete_all   — Delete all user memories
    GET  /healthz             — Health check (static, process alive)
    GET  /health              — Liveness probe (pings internal event loop; 503 if loop blocked)
    GET  /info                — Server info
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

logger = logging.getLogger("hy_memory.server")

# Lazy-init global client
_client = None
_client_lock = None


def _get_client():
    """Lazy-initialize the HyMemoryClient singleton."""
    global _client
    if _client is not None:
        return _client
    from .client import HyMemoryClient
    from .config import MemoryConfig

    logger.info("[server] Initializing HyMemoryClient...")

    config = MemoryConfig.from_env()

    # ---- glue: thinking mode + temperature enforcement ----
    # when the HY_MEMORY_THINKING_MODE env var is set, inject control parameters into models that support thinking
    thinking_mode = os.getenv("HY_MEMORY_THINKING_MODE")
    if thinking_mode:
        _llm_model = (config.llm.model or "").lower()
        # models that need extra_body.thinking to be controlled
        _needs_thinking_body = (
            "kimi" in _llm_model
            or "deepseek" in _llm_model
            or "qwen" in _llm_model
            or "hy3" in _llm_model
            or "hunyuan" in _llm_model
        )
        if _needs_thinking_body:
            extra = config.llm.extra_body or {}
            extra["thinking"] = {"type": thinking_mode}
            config.llm.extra_body = extra
            # kimi/deepseek have hard constraints on temperature
            if "kimi" in _llm_model or "deepseek" in _llm_model:
                required_temp = 0.6 if thinking_mode == "disabled" else 1.0
                config.llm.temperature = required_temp
            logger.info(f"[server] thinking adapter: model={config.llm.model}, thinking={thinking_mode}")

    _client = HyMemoryClient(config=config)
    logger.info("[server] HyMemoryClient ready")
    return _client


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any):
    """Send a JSON response."""
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Optional[Dict]:
    """Read and parse JSON request body."""
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        return {}
    raw = handler.rfile.read(content_length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


class MemoryHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for HY Memory API."""

    # Suppress default access log
    def log_message(self, format, *args):
        logger.debug(f"[http] {args[0] if args else ''}")

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/healthz":
                _json_response(self, 200, {"status": "ok"})
                return

            if path == "/health":
                self._handle_health_liveness()
                return

            if path == "/api/v1/status":
                self._handle_healthz()
                return

            if path == "/info":
                from . import __version__
                _json_response(self, 200, {
                    "name": "hy-memory-server",
                    "version": __version__,
                    "status": "running",
                })
                return

            # GET /api/v1/metrics?minutes=5
            if path == "/api/v1/metrics":
                minutes = 5
                if "?" in self.path:
                    from urllib.parse import parse_qs, urlparse
                    qs = parse_qs(urlparse(self.path).query)
                    minutes = int(qs.get("minutes", ["5"])[0])
                client = _get_client()
                result = client.get_metrics(minutes=minutes)
                _json_response(self, 200, result)
                return

            # GET /api/v1/memories/:id
            m = re.match(r"^/api/v1/memories/([^/]+)$", path)
            if m:
                memory_id = m.group(1)
                client = _get_client()
                result = client.get(memory_id)
                if result is None:
                    _json_response(self, 404, {"error": "not_found", "memory_id": memory_id})
                else:
                    _json_response(self, 200, result)
                return

            _json_response(self, 404, {"error": "not_found", "path": path})

        except Exception as e:
            logger.error(f"[server] GET {path} error: {e}", exc_info=True)
            _json_response(self, 500, {"error": str(e)})

    def do_POST(self):
        path = self.path.rstrip("/")
        body = _read_json_body(self)
        if body is None:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return

        try:
            if path == "/api/v1/add":
                self._handle_add(body)
                return

            if path == "/api/v1/add_extracted":
                self._handle_add_extracted(body)
                return

            if path == "/api/v1/search":
                self._handle_search(body)
                return

            if path == "/api/v1/user_profile":
                self._handle_user_profile(body)
                return

            if path == "/api/v1/list":
                self._handle_list(body)
                return

            if path == "/api/v1/delete_all":
                self._handle_delete_all(body)
                return

            _json_response(self, 404, {"error": "not_found", "path": path})

        except Exception as e:
            logger.error(f"[server] POST {path} error: {e}", exc_info=True)
            _json_response(self, 500, {"error": str(e)})

    def do_PUT(self):
        path = self.path.rstrip("/")
        body = _read_json_body(self)
        if body is None:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return

        try:
            m = re.match(r"^/api/v1/memories/([^/]+)$", path)
            if m:
                memory_id = m.group(1)
                content = body.get("content", "")
                if not content:
                    _json_response(self, 400, {"error": "content is required"})
                    return
                client = _get_client()
                result = client.update(memory_id, content)
                _json_response(self, 200, result)
                return

            _json_response(self, 404, {"error": "not_found", "path": path})

        except Exception as e:
            logger.error(f"[server] PUT {path} error: {e}", exc_info=True)
            _json_response(self, 500, {"error": str(e)})

    def do_DELETE(self):
        path = self.path.rstrip("/")
        try:
            m = re.match(r"^/api/v1/memories/([^/]+)$", path)
            if m:
                memory_id = m.group(1)
                client = _get_client()
                result = client.delete(memory_id)
                _json_response(self, 200, result)
                return

            _json_response(self, 404, {"error": "not_found", "path": path})

        except Exception as e:
            logger.error(f"[server] DELETE {path} error: {e}", exc_info=True)
            _json_response(self, 500, {"error": str(e)})

    # ================================================================
    # Route handlers
    # ================================================================

    def _handle_health_liveness(self):
        """Lightweight liveness probe: checks whether the internal event loop (_LoopThread) is still alive.

        Unlike /api/v1/status (which deep-checks VDB/embed/LLM and opens its own temporary loop), this
        endpoint specifically probes the one _LoopThread loop that actually runs all business coroutines, to see whether it is blocked or deadlocked.

        Approach: submit an almost zero-cost ping coroutine to _loop_thread's loop and await it with a short timeout.
          - loop scheduling normally → returns 200 {"status":"alive"} quickly
          - loop stuck (e.g. blocked synchronously) → timeout → 503 {"status":"loop_blocked"}
        It never calls any business method, never touches VDB/LLM, and adds no timeout to any business request. That way external probing
        can clearly tell "the process is alive" apart from "the loop is alive" and restart the sidecar accordingly, without hurting normal long writes.
        """

        try:
            client = _get_client()
        except Exception as e:
            _json_response(self, 503, {"status": "error", "error": f"client init failed: {e}"})
            return

        loop_thread = getattr(client, "_loop_thread", None)
        loop = getattr(loop_thread, "_loop", None)
        if loop is None:
            _json_response(self, 503, {"status": "error", "error": "no internal loop"})
            return

        if not loop.is_running():
            _json_response(self, 503, {"status": "loop_stopped"})
            return

        async def _ping():
            return True

        try:
            fut = asyncio.run_coroutine_threadsafe(_ping(), loop)
            # short timeout: while the loop is healthy the ping takes microseconds; if it is stuck the timeout marks it unavailable
            fut.result(timeout=2.0)
            _json_response(self, 200, {"status": "alive"})
        except Exception:
            # timeout or scheduling failure → the loop is blocked/deadlocked
            try:
                fut.cancel()
            except Exception:
                pass
            _json_response(self, 503, {"status": "loop_blocked"})

    def _handle_healthz(self):
        """Deep health check: verify VDB, embed, and LLM connectivity."""

        checks = {"status": "ok", "vdb": "ok", "embed": "ok", "llm": "ok"}
        has_error = False

        try:
            client = _get_client()
        except Exception as e:
            _json_response(self, 503, {"status": "error", "error": f"client init failed: {e}"})
            return

        # 1. VDB check — get stats
        try:
            loop = asyncio.new_event_loop()
            stats = loop.run_until_complete(client._vector_store.get_stats())
            loop.close()
            checks["vdb"] = "ok"
            checks["vdb_provider"] = client._config.vector_store.provider
            checks["vdb_collection"] = client._vector_store._collection_name
            checks["vdb_points"] = stats.get("points_count", stats.get("vectors_count", "?"))
        except Exception as e:
            checks["vdb"] = f"error: {e}"
            checks["vdb_provider"] = client._config.vector_store.provider
            has_error = True

        # 2. Embed check — embed a single test string
        try:
            loop = asyncio.new_event_loop()
            vectors = loop.run_until_complete(client._embed_service.embed("health check"))
            loop.close()
            if vectors and len(vectors) > 0:
                checks["embed"] = "ok"
                checks["embed_dims"] = len(vectors)
            else:
                checks["embed"] = "error: empty response"
                has_error = True
        except Exception as e:
            checks["embed"] = f"error: {e}"
            has_error = True

        # 3. LLM check — simple completion
        try:
            from openai import OpenAI
            llm_cfg = client._config.llm
            llm_client = OpenAI(
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url or None,
                timeout=10,
            )
            resp = llm_client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            if resp.choices:
                checks["llm"] = "ok"
            else:
                checks["llm"] = "error: no choices"
                has_error = True
        except Exception as e:
            checks["llm"] = f"error: {e}"
            has_error = True

        checks["status"] = "degraded" if has_error else "ok"
        status_code = 200 if not has_error else 503
        _json_response(self, status_code, checks)

    def _handle_add(self, body: Dict):
        """POST /api/v1/add"""
        data = body.get("data")
        if data is None:
            # Support both "data" (str or messages list) and "messages" key
            messages = body.get("messages")
            text = body.get("text", "")
            if messages:
                data = messages
            elif text:
                data = text
            else:
                _json_response(self, 400, {"error": "data, text, or messages is required"})
                return

        client = _get_client()
        kwargs = {}
        for key in ("user_id", "agent_id", "session_id", "metadata"):
            if key in body:
                kwargs[key] = body[key]

        # memory_at
        if body.get("memory_at"):
            try:
                kwargs["memory_at"] = datetime.fromisoformat(body["memory_at"])
            except Exception:
                pass

        result = client.add(data, **kwargs)
        _json_response(self, 200, result)

    def _handle_add_extracted(self, body: Dict):
        """POST /api/v1/add_extracted — write memories that are already extracted (skips extract, goes straight to reconcile)"""
        # accepts contents (str or list) / content / text
        contents = body.get("contents")
        if contents is None:
            single = body.get("content") or body.get("text")
            if single:
                contents = single
        if not contents:
            _json_response(self, 400, {"error": "contents, content, or text is required"})
            return

        client = _get_client()
        kwargs = {}
        for key in ("user_id", "agent_id", "session_id", "metadata"):
            if key in body:
                kwargs[key] = body[key]
        if body.get("memory_at"):
            try:
                kwargs["memory_at"] = datetime.fromisoformat(body["memory_at"])
            except Exception:
                pass

        result = client.add_extracted(contents, **kwargs)
        _json_response(self, 200, result)

    def _handle_search(self, body: Dict):
        """POST /api/v1/search"""
        query = body.get("query", "")
        if not query:
            _json_response(self, 400, {"error": "query is required"})
            return

        client = _get_client()
        kwargs = {}

        # user_ids: required by SDK, but for openclaw convenience
        # accept both user_ids (list) and user_id (str)
        user_ids = body.get("user_ids")
        if not user_ids:
            user_id = body.get("user_id", "")
            if user_id:
                user_ids = [user_id]
        if not user_ids:
            _json_response(self, 400, {"error": "user_id or user_ids is required"})
            return
        kwargs["user_ids"] = user_ids

        for key in ("agent_ids", "session_ids"):
            if key in body:
                kwargs[key] = body[key]

        for key in ("limit", "min_score", "schema_min_score", "schema_limit", "reader"):
            if key in body:
                kwargs[key] = body[key]

        result = client.search(query, **kwargs)
        _json_response(self, 200, result)

    def _handle_user_profile(self, body: Dict):
        """POST /api/v1/user_profile"""
        user_id = str(body.get("user_id") or "").strip()
        if not user_id:
            _json_response(self, 400, {"error": "user_id is required"})
            return
        client = _get_client()
        result = client.get_user_profile(
            user_id=user_id,
        )
        _json_response(self, 200, result)

    def _handle_list(self, body: Dict):
        """POST /api/v1/list"""
        client = _get_client()
        kwargs = {}
        for key in ("user_id", "agent_id", "limit", "offset", "order"):
            if key in body:
                kwargs[key] = body[key]
        result = client.list_memories(**kwargs)
        _json_response(self, 200, result)

    def _handle_delete_all(self, body: Dict):
        """POST /api/v1/delete_all"""
        client = _get_client()
        kwargs = {}
        for key in ("user_id", "agent_ids", "session_ids"):
            if key in body:
                kwargs[key] = body[key]
        result = client.delete_all(**kwargs)
        _json_response(self, 200, result)


def run_server(port: int = 19527, host: str = "127.0.0.1"):
    """Start the HTTP server (blocking)."""
    # Configure logging — basicConfig for root; hy_memory level set after client init
    log_level = getattr(logging, os.getenv("MEMORY_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Pre-initialize client to fail fast (this calls setup_logging internally)
    logger.info(f"[server] Starting HY Memory Server on {host}:{port}")
    _get_client()

    # After setup_logging has run, force correct levels
    hy_logger = logging.getLogger("hy_memory")
    hy_logger.setLevel(log_level)
    for h in hy_logger.handlers:
        h.setLevel(log_level)
    # Suppress noisy third-party debug logs (must be after setup_logging)
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Use ThreadingHTTPServer so /healthz is never blocked by long-running requests
    server = ThreadingHTTPServer((host, port), MemoryHTTPHandler)
    server.daemon_threads = True  # Threads die with main process
    logger.info(f"[server] Listening on http://{host}:{port}")
    logger.info(f"[server] Health check: http://{host}:{port}/healthz")

    # Graceful shutdown
    def _shutdown(signum, frame):
        logger.info("[server] Shutting down...")
        server.shutdown()
        if _client is not None:
            _client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutdown(None, None)


def main():
    """Entry point for `python -m hy_memory.server`."""
    import argparse
    parser = argparse.ArgumentParser(description="HY Memory HTTP Server")
    parser.add_argument(
        "--port", type=int,
        default=int(os.getenv("HY_MEMORY_SERVER_PORT", "19527")),
        help="Port to listen on (default: 19527)",
    )
    parser.add_argument(
        "--host", type=str,
        default=os.getenv("HY_MEMORY_SERVER_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
