"""Open Pixel2Play 150M policy adapter for the Windows evaluation loop."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from gameeval.core.observation import Observation


class OpenP2PTransport(Protocol):
    def health(self) -> dict[str, Any]: ...

    def reset(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def act(self, frame: np.ndarray, frame_id: int) -> dict[str, Any]: ...


class HTTPOpenP2PTransport:
    """Small standard-library client for the WSL open-p2p sidecar."""

    def __init__(self, endpoint: str, timeout: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return self._request(method, path, data, headers)

    def _request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request = Request(
            f"{self.endpoint}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"open-p2p sidecar request failed: {method} {path}: {exc}") from exc
        result = json.loads(body.decode("utf-8")) if body else {}
        if not isinstance(result, dict):
            raise RuntimeError("open-p2p sidecar response must be a JSON object")
        return result

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/v1/health")

    def reset(self, context: dict[str, Any]) -> dict[str, Any]:
        return self._json_request("POST", "/v1/reset", context)

    def act(self, frame: np.ndarray, frame_id: int) -> dict[str, Any]:
        height, width = frame.shape[:2]
        return self._request(
            "POST",
            "/v1/act",
            np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
            {
                "Accept": "application/json",
                "Content-Type": "application/octet-stream",
                "X-Frame-Width": str(width),
                "X-Frame-Height": str(height),
                "X-Frame-Id": str(frame_id),
            },
        )


_P2P_KEY_TO_TOKEN = {
    "Space": "space",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "a": "a",
    "d": "d",
    "e": "e",
    "f": "f",
    "q": "q",
    "w": "w",
    "s": "s",
    "z": "z",
    "LeftShift": "shift",
    "RightShift": "shift",
}
_P2P_MOUSE_TO_TOKEN = {"0": "L", "1": "R", "2": "M"}


def adapt_open_p2p_response(
    response: dict[str, Any],
    *,
    frame_duration_ms: int = 50,
) -> dict[str, Any]:
    """Translate one published open-p2p action response to GameEval."""
    source_keys = [str(value) for value in response.get("keys", [])]
    source_buttons = [str(value) for value in response.get("mouse_buttons", [])]
    tokens = [token for key in source_keys if (token := _P2P_KEY_TO_TOKEN.get(key))]
    tokens.extend(token for button in source_buttons if (token := _P2P_MOUSE_TO_TOKEN.get(button)))
    dropped_keys = [key for key in source_keys if key not in _P2P_KEY_TO_TOKEN]
    dropped_buttons = [button for button in source_buttons if button not in _P2P_MOUSE_TO_TOKEN]
    raw: dict[str, Any] = {
        "policy": "open-p2p",
        "model_size": "150M",
        "source_frame_id": response.get("frame_id"),
        "source_keys": source_keys,
        "source_mouse_buttons": source_buttons,
        "dropped_keys": dropped_keys,
        "dropped_mouse_buttons": dropped_buttons,
    }
    # Transports that measure their own latency (e.g. the in-process backend)
    # report it so the artifact records the achieved control rate.
    inference_ms = response.get("inference_ms")
    if isinstance(inference_ms, (int, float)) and not isinstance(inference_ms, bool):
        raw["inference_ms"] = float(inference_ms)
    return {
        "mouse_dx": float(response.get("mouse_dx", 0.0)),
        "mouse_dy": float(response.get("mouse_dy", 0.0)),
        "duration_ms": int(frame_duration_ms),
        "frames": [{"inputs": tokens}],
        "raw": raw,
    }


class OpenP2P150MAgent:
    """Convert open-p2p 150M actions into GameEval's Windows action schema."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:9876",
        timeout: float = 10.0,
        width: int = 192,
        height: int = 192,
        frame_duration_ms: int = 50,
        check_health: bool = True,
        transport: OpenP2PTransport | None = None,
        **_: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame_duration_ms = int(frame_duration_ms)
        if self.width < 1 or self.height < 1:
            raise ValueError("open-p2p input dimensions must be positive")
        self.transport = transport or HTTPOpenP2PTransport(endpoint, timeout)
        self.frame_id = 0
        self.context: dict[str, Any] = {"model": "open-p2p", "model_size": "150M"}
        if check_health:
            self.transport.health()

    def set_task_context(
        self,
        *,
        game: str,
        task: str,
        instruction: str | None = None,
    ) -> None:
        self.context.update(
            {"game": game, "task_id": task, "instruction": instruction or ""}
        )

    def reset(self) -> None:
        self.frame_id = 0
        self.transport.reset(dict(self.context))

    def act(self, observation: Observation) -> dict[str, Any]:
        if observation.screenshot is None:
            raise ValueError("open-p2p requires an RGB screenshot observation")
        frame = self._prepare_frame(observation.screenshot)
        response = self.transport.act(frame, self.frame_id)
        response_id = int(response.get("frame_id", self.frame_id))
        if response_id != self.frame_id:
            raise RuntimeError(
                f"open-p2p returned frame_id={response_id}, expected {self.frame_id}"
            )
        self.frame_id += 1

        return adapt_open_p2p_response(
            {**response, "frame_id": response_id},
            frame_duration_ms=self.frame_duration_ms,
        )

    def _prepare_frame(self, value: np.ndarray) -> np.ndarray:
        frame = np.asarray(value, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("open-p2p screenshot must have shape (H, W, 3)")
        if frame.shape[:2] != (self.height, self.width):
            frame = np.asarray(
                Image.fromarray(frame).resize(
                    (self.width, self.height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        return np.ascontiguousarray(frame)


__all__ = [
    "OpenP2P150MAgent",
    "HTTPOpenP2PTransport",
    "OpenP2PTransport",
    "adapt_open_p2p_response",
]
