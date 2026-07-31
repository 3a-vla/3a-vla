"""Public adapter for a private GP evaluation interface.

The repository contains only a small HTTP bridge contract.  Tencent-internal
endpoints, credentials, schemas, and state normalization remain in a private
bridge process.  The agent still receives pixels only; normalized state is
available through ``StateProvider`` exclusively to the evaluator.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from gameeval.core.game_adapter import AdapterConfig, GameAdapter


class BridgeTransport(Protocol):
    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...


class HTTPBridgeTransport:
    """Minimal JSON transport with bearer credentials supplied at runtime."""

    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"GP bridge request failed: {method} {path}: {exc}") from exc
        result = json.loads(body.decode("utf-8")) if body else {}
        if not isinstance(result, dict):
            raise RuntimeError("GP bridge response must be a JSON object")
        return result


class GPBridgeConfig(AdapterConfig):
    """Runtime configuration without any built-in private credentials."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(game="gp", **kwargs)
        self.base_url = base_url or os.getenv("GAMEEVAL_GP_BRIDGE_URL", "http://127.0.0.1:8765")
        self.token = token if token is not None else os.getenv("GAMEEVAL_GP_TOKEN", "")
        self.timeout = float(timeout)


class GPBridgeAdapter(GameAdapter):
    """State-backed GP adapter over the public bridge protocol."""

    def __init__(self, transport: BridgeTransport | None = None) -> None:
        self._transport_override = transport
        self._transport: BridgeTransport | None = None
        self._config: GPBridgeConfig | None = None
        self._state: dict[str, Any] = {}
        self._frame: np.ndarray | None = None
        self._connected = False

    def connect(self, config: AdapterConfig) -> None:
        extra = getattr(config, "extra", {}) or {}
        self._config = (
            config
            if isinstance(config, GPBridgeConfig)
            else GPBridgeConfig(
                base_url=extra.get("base_url"),
                token=extra.get("token"),
                timeout=extra.get("timeout", 30.0),
            )
        )
        self._transport = self._transport_override or HTTPBridgeTransport(
            self._config.base_url,
            self._config.token,
            self._config.timeout,
        )
        self._transport.request("GET", "/v1/health")
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def reset(self, task_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        response = self._request("POST", "/v1/reset", {"task": task_config})
        self._update(response)
        return self.screenshot(), self.get_state()

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        response = self._request("POST", "/v1/step", {"action": action})
        self._update(response)
        info = response.get("info", {})
        if not isinstance(info, dict):
            raise RuntimeError("GP bridge 'info' must be an object")
        return self.screenshot(), self.get_state(), info

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def screenshot(self) -> np.ndarray:
        if self._frame is None:
            response = self._request("GET", "/v1/screenshot")
            self._update(response)
        if self._frame is None:
            raise RuntimeError("GP bridge did not provide a frame")
        return self._frame.copy()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._transport is None:
            raise RuntimeError("GP adapter is not connected")
        return self._transport.request(method, path, payload)

    def _update(self, response: dict[str, Any]) -> None:
        state = response.get("state")
        if state is not None:
            if not isinstance(state, dict):
                raise RuntimeError("GP bridge 'state' must be an object")
            self._state = dict(state)
        frame_b64 = response.get("frame_b64")
        if frame_b64:
            raw = base64.b64decode(frame_b64, validate=True)
            self._frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    @property
    def game_name(self) -> str:
        return "gp"

    @property
    def is_connected(self) -> bool:
        return self._connected
