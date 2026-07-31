"""CSGO runtime backed by screen capture, desktop input, and hidden GSI state.

The adapter deliberately does not define scenarios or success rules. Operators
or private setup hooks prepare the game, task files bound rollout duration, and
the state evaluator judges the latest Game State Integration (GSI) snapshot.
Privileged GSI values are never returned as public per-step metadata.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from gameeval.adapters.csgo.gamestate import GameState
from gameeval.adapters.csgo.gsi_server import GSIServer
from gameeval.control.windows import WindowsInputController
from gameeval.core.action_space import Action
from gameeval.core.game_adapter import AdapterConfig, GameAdapter
from gameeval.utils.cs_screenshot import DEFAULT_PROCESS, CSScreenCapture
from gameeval.utils.screenshot import capture_screen

logger = logging.getLogger("gameeval.csgo.adapter")


class CSGOAdapterConfig(AdapterConfig):
    """Runtime settings for a local or streamed CSGO client."""

    def __init__(
        self,
        gsi_host: str = "0.0.0.0",
        gsi_port: int = 3000,
        gsi_auth_token: str | None = None,
        screenshot_region: dict | None = None,
        screenshot_crop_size: tuple[int, int] | None = None,
        screenshot_target_size: tuple[int, int] | None = None,
        screenshot_process_name: str | None = DEFAULT_PROCESS,
        screenshot_debug: bool = False,
        enable_input_injection: bool = False,
        mouse_scale: float = 1.0,
        hold_left_mouse: bool = False,
        **kwargs: Any,
    ):
        super().__init__(game="csgo", **kwargs)
        self.gsi_host = gsi_host
        self.gsi_port = int(gsi_port)
        self.gsi_auth_token = gsi_auth_token
        self.screenshot_region = screenshot_region
        self.screenshot_crop_size = (
            tuple(screenshot_crop_size) if screenshot_crop_size else None
        )
        self.screenshot_target_size = (
            tuple(screenshot_target_size) if screenshot_target_size else None
        )
        self.screenshot_process_name = screenshot_process_name
        self.screenshot_debug = bool(screenshot_debug)
        self.enable_input_injection = bool(enable_input_injection)
        self.mouse_scale = float(mouse_scale)
        self.hold_left_mouse = bool(hold_left_mouse)


class CSGOGameAdapter(GameAdapter):
    """Capture/control runtime plus evaluator-only GSI state provider."""

    def __init__(self) -> None:
        self._gsi: GSIServer | None = None
        self._config: CSGOAdapterConfig | None = None
        self._latest_game_state: GameState | None = None
        self._connected = False
        self._screenshot_region: dict | None = None
        self._cs_capture: CSScreenCapture | None = None
        self._injector: Any = None

    def connect(self, config: AdapterConfig) -> None:
        if isinstance(config, CSGOAdapterConfig):
            self._config = config
        else:
            extra = config.extra or {}
            self._config = CSGOAdapterConfig(
                gsi_host=extra.get("gsi_host", "0.0.0.0"),
                gsi_port=extra.get("gsi_port", 3000),
                gsi_auth_token=extra.get("gsi_auth_token"),
                screenshot_region=extra.get("screenshot_region"),
                screenshot_crop_size=extra.get("screenshot_crop_size"),
                screenshot_target_size=extra.get("screenshot_target_size"),
                screenshot_process_name=extra.get(
                    "screenshot_process_name", DEFAULT_PROCESS
                ),
                screenshot_debug=extra.get("screenshot_debug", False),
                enable_input_injection=extra.get("enable_input_injection", False),
                mouse_scale=extra.get("mouse_scale", 1.0),
                hold_left_mouse=extra.get("hold_left_mouse", False),
            )

        self._gsi = GSIServer(
            host=self._config.gsi_host,
            port=self._config.gsi_port,
            auth_token=self._config.gsi_auth_token,
            on_state=self._on_gsi_state,
        )
        self._gsi.start()
        self._screenshot_region = self._config.screenshot_region

        try:
            self._cs_capture = CSScreenCapture(
                crop_size=self._config.screenshot_crop_size,
                target_size=self._config.screenshot_target_size,
                process_name=self._config.screenshot_process_name,
                debug=self._config.screenshot_debug,
            )
            logger.info(
                "CSGO capture initialized (process=%s, window_found=%s)",
                self._config.screenshot_process_name,
                self._cs_capture.has_window,
            )
        except Exception as exc:  # pragma: no cover - depends on desktop runtime
            logger.warning("Window capture unavailable; using monitor capture: %s", exc)
            self._cs_capture = None

        if self._config.enable_input_injection:
            self._injector = WindowsInputController(
                mouse_scale=self._config.mouse_scale,
                enabled=True,
                hold_left_mouse=self._config.hold_left_mouse,
            )
            backend = "SendInput"
            logger.info("CSGO input enabled through %s", backend)

        self._connected = True
        logger.info("CSGO runtime connected; GSI is evaluator-only")

    def close(self) -> None:
        if self._injector is not None:
            self._injector.release_all()
        if self._gsi is not None:
            self._gsi.stop()
        if self._cs_capture is not None:
            self._cs_capture.close()
        self._injector = None
        self._cs_capture = None
        self._connected = False

    def reset(self, task_config: dict) -> tuple[np.ndarray, dict]:
        """Start sampling the operator- or hook-prepared scene."""
        if self._injector is not None:
            self._injector.release_all()
        frame = self.screenshot()
        state = self.get_state()
        if not state:
            logger.warning(
                "No CSGO GSI snapshot received yet; state judging may return unknown"
            )
        logger.info("CSGO episode ready: task=%s", task_config.get("task_id", ""))
        return frame, state

    def step(self, action: dict) -> tuple[np.ndarray, dict, dict]:
        """Apply one action and return a frame plus public input diagnostics."""
        info: dict[str, Any] = {}
        if self._injector is not None:
            try:
                act_obj = action if isinstance(action, Action) else Action.from_dict(action)
                info["input"] = self._injector.apply(act_obj)
            except Exception as exc:
                logger.warning("CSGO input injection failed: %s", exc)
                info["input"] = {"error": str(exc)}

        time.sleep(0.125)
        return self.screenshot(), self.get_state(), info

    def get_state(self) -> dict:
        """Return the latest normalized GSI snapshot for the state evaluator."""
        if self._latest_game_state is None:
            return {}
        return self._latest_game_state.to_dict()

    def screenshot(self) -> np.ndarray:
        """Capture a CSGO frame as an RGB array."""
        if self._cs_capture is not None:
            try:
                image = self._cs_capture.capture()
                if image is not None:
                    processed = self._cs_capture.process_to_pil(image)
                    if processed is not None:
                        return np.asarray(processed, dtype=np.uint8)
            except Exception as exc:
                logger.debug("CSGO window capture failed; using monitor capture: %s", exc)
        return capture_screen(region=self._screenshot_region)

    @property
    def game_name(self) -> str:
        return "csgo"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def execute_command(self, command: str) -> str | None:
        """CSGO has no command-injection channel in the public adapter."""
        logger.debug("Ignoring unsupported CSGO command: %r", command)
        return None

    def _on_gsi_state(self, state: GameState) -> None:
        self._latest_game_state = state
