"""Public GP-Visual runtime using only Windows pixels and desktop input."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from gameeval.control.windows import WindowsInputController
from gameeval.core.action_space import Action
from gameeval.core.runtime import GameRuntime, RuntimeConfig
from gameeval.utils.screenshot import capture_screen


class GPVisualConfig(RuntimeConfig):
    """Configuration for a Windows Game for Peace client or emulator."""

    def __init__(
        self,
        *,
        screenshot_region: dict[str, int] | None = None,
        monitor_index: int = 1,
        window_title: str | None = None,
        focus_window: bool = False,
        require_unobstructed: bool = True,
        enable_input: bool = False,
        mouse_scale: float = 1.0,
        use_key: str = "f",
        step_delay: float = 0.05,
        reset_delay: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(game="gp", **kwargs)
        self.screenshot_region = screenshot_region
        self.monitor_index = int(monitor_index)
        self.window_title = str(window_title) if window_title else None
        self.focus_window = bool(focus_window)
        self.require_unobstructed = bool(require_unobstructed)
        self.enable_input = bool(enable_input)
        self.mouse_scale = float(mouse_scale)
        self.use_key = str(use_key)
        self.step_delay = float(step_delay)
        self.reset_delay = float(reset_delay)


class GPVisualAdapter(GameRuntime):
    """GP runtime with no serialized-state channel.

    Initialization is deliberately external: the task YAML records the exact
    version, resolution, keymap, and operator/reset-hook checklist, while this
    runtime captures RGB frames and emits ordinary Windows input events.
    """

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
        self._config: GPVisualConfig | None = None
        self._cached_hwnd: int | None = None
        self._connected = False

    def connect(self, config: RuntimeConfig) -> None:
        extra = getattr(config, "extra", {}) or {}
        self._config = (
            config
            if isinstance(config, GPVisualConfig)
            else GPVisualConfig(
                screenshot_region=extra.get("screenshot_region"),
                monitor_index=extra.get("monitor_index", 1),
                window_title=extra.get("window_title"),
                focus_window=extra.get("focus_window", False),
                require_unobstructed=extra.get("require_unobstructed", True),
                enable_input=extra.get("enable_input", False),
                mouse_scale=extra.get("mouse_scale", 1.0),
                use_key=extra.get("use_key", "f"),
                step_delay=extra.get("step_delay", 0.05),
                reset_delay=extra.get("reset_delay", 5.0),
            )
        )
        self._controller = self._controller_override or WindowsInputController(
            enabled=self._config.enable_input,
            mouse_scale=self._config.mouse_scale,
            use_key=self._config.use_key,
        )
        self._connected = True

    def close(self) -> None:
        if self._controller is not None:
            self._controller.release_all()
        self._connected = False

    def reset_runtime(self, task_config: dict[str, Any]) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("GP-Visual adapter is not connected")
        if self._controller is not None:
            self._controller.release_all()
        self._resolve_window_region(task_config)
        if self._reset_hook is not None:
            self._reset_hook(task_config)
        delay = float((task_config.get("setup", {}) or {}).get("reset_delay", 0.0))
        if not delay and self._config is not None:
            delay = self._config.reset_delay
        if delay > 0:
            time.sleep(delay)
        return self.screenshot()

    def _resolve_window_region(self, task_config: dict[str, Any]) -> None:
        """Bind the capture region to the game window's client area.

        Resolved per episode because the operator may move the window between
        episodes. The task's declared resolution is enforced so a mis-sized
        window fails the run instead of silently changing the observation, and
        an obstructed window is rejected because screen capture would otherwise
        record whatever is painted on top of the game.
        """
        if self._config is None or not self._config.window_title:
            return
        from gameeval.utils.window import find_window_region, resolve_capture_region

        window = find_window_region(self._config.window_title)
        self._cached_hwnd = int(window["hwnd"])
        if self._config.focus_window:
            self._focus_window(self._cached_hwnd)

        declared = (task_config.get("setup", {}) or {}).get("resolution")
        expected = None
        if isinstance(declared, (list, tuple)) and len(declared) == 2:
            expected = (int(declared[0]), int(declared[1]))
        self._config.screenshot_region = resolve_capture_region(
            self._config.window_title,
            expected_size=expected,
            require_unobstructed=self._config.require_unobstructed,
        )

    @staticmethod
    def _focus_window(hwnd: int) -> None:
        """Restore and raise the game window before capturing it."""
        import ctypes

        user32 = ctypes.windll.user32
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        # Compositing and the game's own redraw need a moment to settle.
        time.sleep(1.0)

    def step_runtime(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("GP-Visual adapter is not connected")
        parsed = Action.from_dict(action)
        focus_error = self._focus_violation()
        if focus_error is not None:
            # SendInput targets whichever window holds focus. Injecting now would
            # type into that window instead of the game, so stop the episode
            # rather than corrupt both the run and the operator's desktop.
            if self._controller is not None:
                self._controller.release_all()
            raise RuntimeError(focus_error)
        input_info = self._controller.apply(parsed) if self._controller is not None else {}
        if self._config is not None and self._config.step_delay > 0:
            time.sleep(self._config.step_delay)
        return self.screenshot(), {"input": input_info}

    def _focus_violation(self) -> str | None:
        """Return why input injection is unsafe this step, or None if it is safe."""
        if self._config is None or not self._config.enable_input:
            return None
        if not self._config.window_title:
            return None
        if self._cached_hwnd is None:
            return None

        import ctypes

        foreground = int(ctypes.windll.user32.GetForegroundWindow())
        if foreground == self._cached_hwnd:
            return None

        buffer = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetWindowTextW(foreground, buffer, 256)
        return (
            f"Game window lost focus to {buffer.value or foreground!r}; "
            "keyboard and mouse events would be delivered there instead of to "
            "the game. Refocus the game window before rerunning."
        )

    def screenshot(self) -> np.ndarray:
        if self._capture_override is not None:
            return self._capture_override()
        if self._config is None:
            raise RuntimeError("GP-Visual adapter is not connected")
        return capture_screen(
            region=self._config.screenshot_region,
            monitor_index=self._config.monitor_index,
        )

    @property
    def game_name(self) -> str:
        return "gp"

    @property
    def is_connected(self) -> bool:
        return self._connected
