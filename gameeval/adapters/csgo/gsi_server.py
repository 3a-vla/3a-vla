"""CSGO Game State Integration (GSI) async HTTP server.

CSGO sends periodic POST requests with JSON game state to a configured
HTTP endpoint.  This module implements an asynchronous receiver using
``aiohttp`` that:

1. Listens for GSI payloads on a configurable port.
2. Parses each payload into a :class:`GameState`.
3. Stores the latest state in a thread-safe buffer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable

from aiohttp import web

from gameeval.adapters.csgo.gamestate import GameState

logger = logging.getLogger("gameeval.csgo.gsi")


class GSIServer:
    """Asynchronous HTTP server that receives CSGO GSI payloads.

    Parameters
    ----------
    host : str
        Bind address (default ``"127.0.0.1"``).
    port : int
        Listen port (default ``3000``).
    auth_token : str | None
        If set, reject payloads without matching ``auth.token``.
    on_state : Callable[[GameState], None] | None
        Optional callback invoked on each new state.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
        auth_token: str | None = None,
        on_state: Callable[[GameState], None] | None = None,
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.on_state = on_state

        # State buffer
        self._latest_state: GameState | None = None
        self._lock = threading.Lock()

        # Server internals
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Stats
        self._payloads_received = 0

    # ---- Public API ----------------------------------------------------------

    def start(self) -> None:
        """Start the GSI server in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._site is not None:
                break
            time.sleep(0.05)

        logger.info("GSI server started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stop the GSI server."""
        if not self._running:
            return
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("GSI server stopped.")

    @property
    def latest_state(self) -> GameState | None:
        """Return the most recently received game state."""
        with self._lock:
            return self._latest_state

    @property
    def payloads_received(self) -> int:
        return self._payloads_received

    # ---- Internals -----------------------------------------------------------

    def _run_server(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_app())
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._cleanup())
            self._loop.close()

    async def _start_app(self) -> None:
        self._app = web.Application()
        self._app.router.add_post("/", self._handle_post)
        self._app.router.add_get("/health", self._handle_health)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def _cleanup(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle_post(self, request: web.Request) -> web.Response:
        """Handle incoming GSI POST."""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        # Auth check
        if self.auth_token:
            auth = payload.get("auth", {})
            if auth.get("token") != self.auth_token:
                return web.Response(status=403, text="Unauthorized")

        # Parse state
        try:
            state = GameState.from_gsi_payload(payload)
        except Exception as e:
            logger.warning("Failed to parse GSI payload: %s", e)
            return web.Response(status=500, text="Parse error")

        # Update buffer
        with self._lock:
            self._latest_state = state
        self._payloads_received += 1

        # Callback
        if self.on_state:
            try:
                self.on_state(state)
            except Exception as e:
                logger.warning("on_state callback error: %s", e)

        return web.Response(text="OK")

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "payloads_received": self._payloads_received,
            "has_state": self._latest_state is not None,
        })
