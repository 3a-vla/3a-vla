"""GTA5 runtime based only on desktop pixels and keyboard/mouse control.

GTA5 exposes no privileged state to GameEval.  The adapter deliberately keeps
the runtime thin: an operator or a private reset hook prepares the scene, the
agent receives screenshots, actions are injected at the OS boundary, and the
completed rollout is evaluated by a VLM judge.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from gameeval.core.action_space import Action
from gameeval.core.runtime import GameRuntime, RuntimeConfig
from gameeval.utils.screenshot import capture_screen


class GTA5AdapterConfig(RuntimeConfig):
    """Configuration for a local, captured, or streamed GTA5 client."""

    def __init__(
        self,
        *,
        screenshot_region: dict[str, int] | None = None,
        monitor_index: int = 1,
        enable_input: bool = False,
        mouse_scale: float = 1.0,
        frame_duration_ms: int = 50,
        use_key: str = "f",
        step_delay: float = 0.05,
        reset_delay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(game="gta5", **kwargs)
        self.screenshot_region = screenshot_region
        self.monitor_index = int(monitor_index)
        self.enable_input = bool(enable_input)
        self.mouse_scale = float(mouse_scale)
        self.frame_duration_ms = int(frame_duration_ms)
        self.use_key = str(use_key)
        self.step_delay = float(step_delay)
        self.reset_delay = float(reset_delay)


class GTA5GameAdapter(GameRuntime):
    """Screen/control-only runtime; it never implements ``StateProvider``."""

    def __init__(
        self,
        *,
        capture: Callable[[], np.ndarray] | None = None,
        controller: Any | None = None,
        reset_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._capture_override = capture
        self._controller_override = controller
        self._reset_hook = reset_hook
        self._controller: Any | None = None
        self._config: GTA5AdapterConfig | None = None
        self._connected = False

    def connect(self, config: RuntimeConfig) -> None:
        extra = getattr(config, "extra", {}) or {}
        self._config = (
            config
            if isinstance(config, GTA5AdapterConfig)
            else GTA5AdapterConfig(
                screenshot_region=extra.get("screenshot_region"),
                monitor_index=extra.get("monitor_index", 1),
                enable_input=extra.get("enable_input", False),
                mouse_scale=extra.get("mouse_scale", 1.0),
                frame_duration_ms=extra.get("frame_duration_ms", 50),
                use_key=extra.get("use_key", "f"),
                step_delay=extra.get("step_delay", 0.05),
                reset_delay=extra.get("reset_delay", 0.0),
            )
        )
        self._controller = self._controller_override or self._make_controller()
        self._connected = True

    def _make_controller(self) -> Any:
        assert self._config is not None
        from gameeval.control.windows import WindowsInputController

        return WindowsInputController(
            enabled=self._config.enable_input,
            mouse_scale=self._config.mouse_scale,
            use_key=self._config.use_key,
        )

    def close(self) -> None:
        if self._controller is not None:
            self._controller.release_all()
        self._connected = False

    def reset_runtime(self, task_config: dict[str, Any]) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("GTA5 adapter is not connected")
        if self._controller is not None:
            self._controller.release_all()
        if self._reset_hook is not None:
            self._reset_hook(task_config)
        delay = float(task_config.get("setup", {}).get("reset_delay", 0.0))
        if not delay and self._config is not None:
            delay = self._config.reset_delay
        if delay > 0:
            time.sleep(delay)
        return self.screenshot()

    def step_runtime(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("GTA5 adapter is not connected")
        parsed = Action.from_dict(action)
        input_info = self._controller.apply(parsed) if self._controller is not None else {}
        if self._config is not None and self._config.step_delay > 0:
            time.sleep(self._config.step_delay)
        return self.screenshot(), {"input": input_info}

    def screenshot(self) -> np.ndarray:
        if self._capture_override is not None:
            return self._capture_override()
        if self._config is None:
            raise RuntimeError("GTA5 adapter is not connected")
        return capture_screen(
            region=self._config.screenshot_region,
            monitor_index=self._config.monitor_index,
        )

    @property
    def game_name(self) -> str:
        return "gta5"

    @property
    def is_connected(self) -> bool:
        return self._connected
