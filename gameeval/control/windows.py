"""Generic Windows keyboard/mouse backend using ``SendInput``.

Listen-Server mode has no engine-level action API; the only way to
drive the in-game character from GameEval is to synthesize OS-level
keyboard + mouse events so a focused game receives ordinary device input
like human input. This module exposes a single class,
:class:`WindowsInputController`, that translates an
:class:`gameeval.core.action_space.Action` into a sequence of
``SendInput`` calls while maintaining the "currently pressed" state
across ticks so keys are released cleanly at episode boundaries.

The low-level Win32 glue is intentionally self-contained.

Design notes
------------
* Keyboard events use **scancodes** (``KEYEVENTF_SCANCODE``). Games —
  especially DirectInput-era engines — respond more reliably to
  scancodes than to virtual-key codes.
* Mouse motion uses **relative** events (``MOUSEEVENTF_MOVE`` without
  ``MOUSEEVENTF_ABSOLUTE``). ``Action.mouse_dx / mouse_dy`` are
  interpreted as **pixels** for the current tick; an optional
  ``mouse_scale`` can multiply the incoming deltas (useful when an
  agent emits VLA-scaled values instead of raw pixels).
* Diff-based key handling: the injector keeps a ``_pressed_keys`` /
  ``_pressed_mouse`` set and only emits the DOWN / UP transitions
  needed to reach the target state. This mirrors how a human would
  hold ``W`` across many frames without re-pressing it.
* On non-Windows platforms the module still imports; the injector
  raises ``RuntimeError`` on construction, and the adapter is
  expected to gate instantiation on ``enable_input_injection``.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes
from typing import Any

from gameeval.core.action_space import Action, DiscreteAction

logger = logging.getLogger("gameeval.control.windows")


# ---------------------------------------------------------------------------
# Win32 SendInput glue (Windows only)
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

# Input types
_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1

# Mouse event flags
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040

# Keyboard event flags
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

# DirectInput scancodes for game-relevant keys.
_SCAN_CODE: dict[str, int] = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20,
    "space": 0x39, "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    "e": 0x12, "r": 0x13, "f": 0x21, "g": 0x22,
    "b": 0x30, "q": 0x10, "c": 0x2E, "z": 0x2C,
    "x": 0x2D, "v": 0x2F, "t": 0x14,
    "tab": 0x0F,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
}


if _IS_WINDOWS:

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]

    _user32 = ctypes.windll.user32

    def _send_input(*inputs: "_INPUT") -> None:
        n = len(inputs)
        arr = (_INPUT * n)(*inputs)
        _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))

    def _key_event(key: str, down: bool) -> None:
        """Press or release a key by scancode."""
        scan = _SCAN_CODE.get(key)
        if scan is None:
            logger.debug("Unknown key %r — ignored.", key)
            return
        inp = _INPUT()
        inp.type = _INPUT_KEYBOARD
        inp.union.ki.wVk = 0
        inp.union.ki.wScan = scan
        flags = _KEYEVENTF_SCANCODE
        if not down:
            flags |= _KEYEVENTF_KEYUP
        inp.union.ki.dwFlags = flags
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = None
        _send_input(inp)

    def _mouse_button(button: str, down: bool) -> None:
        """Press or release a mouse button."""
        if button == "left":
            flag = _MOUSEEVENTF_LEFTDOWN if down else _MOUSEEVENTF_LEFTUP
        elif button == "right":
            flag = _MOUSEEVENTF_RIGHTDOWN if down else _MOUSEEVENTF_RIGHTUP
        elif button == "middle":
            flag = _MOUSEEVENTF_MIDDLEDOWN if down else _MOUSEEVENTF_MIDDLEUP
        else:
            logger.debug("Unknown mouse button %r — ignored.", button)
            return
        inp = _INPUT()
        inp.type = _INPUT_MOUSE
        inp.union.mi.dx = 0
        inp.union.mi.dy = 0
        inp.union.mi.mouseData = 0
        inp.union.mi.dwFlags = flag
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = None
        _send_input(inp)

    def _mouse_move_relative(dx: int, dy: int) -> None:
        """Move the mouse cursor by (dx, dy) pixels."""
        inp = _INPUT()
        inp.type = _INPUT_MOUSE
        inp.union.mi.dx = dx
        inp.union.mi.dy = dy
        inp.union.mi.mouseData = 0
        inp.union.mi.dwFlags = _MOUSEEVENTF_MOVE
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = None
        _send_input(inp)

else:  # pragma: no cover — non-Windows fallbacks for import safety.

    def _key_event(key: str, down: bool) -> None:  # noqa: D401
        raise RuntimeError("WindowsInputController requires Windows (SendInput).")

    def _mouse_button(button: str, down: bool) -> None:
        raise RuntimeError("WindowsInputController requires Windows (SendInput).")

    def _mouse_move_relative(dx: int, dy: int) -> None:
        raise RuntimeError("WindowsInputController requires Windows (SendInput).")


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class WindowsInputController:
    """Translate :class:`Action` into Windows SendInput events.

    Parameters
    ----------
    mouse_scale :
        Multiplier applied to ``action.mouse_dx / mouse_dy`` before
        emitting ``MOUSEEVENTF_MOVE``. Default ``1.0`` treats the
        action deltas as raw pixels. Set to ``10.0`` if the upstream
        agent emits VLA-scaled values (``dx = pixels * 0.1``).
    enabled :
        When ``False`` ``apply`` is a no-op that still returns a
        stats dict — useful for dry runs / CI where no game is
        attached.

    Notes
    -----
    The controller is stateful: pressed keys and mouse buttons persist
    between ``apply`` calls, so a held ``W`` is not re-pressed every
    tick. Always call :meth:`release_all` at episode boundaries and
    on ``close`` to avoid leaving keys stuck on the operator's
    desktop.
    """

    # DiscreteAction → scancode key name
    _DISCRETE_TO_KEY: dict[DiscreteAction, str] = {
        DiscreteAction.MOVE_FORWARD:  "w",
        DiscreteAction.MOVE_BACKWARD: "s",
        DiscreteAction.MOVE_LEFT:     "a",
        DiscreteAction.MOVE_RIGHT:    "d",
        DiscreteAction.JUMP:          "space",
        DiscreteAction.CROUCH:        "ctrl",
        DiscreteAction.RELOAD:        "r",
        DiscreteAction.WEAPON_SLOT_1: "1",
        DiscreteAction.WEAPON_SLOT_2: "2",
        DiscreteAction.WEAPON_SLOT_3: "3",
        DiscreteAction.USE:           "e",
    }

    # Timed action-chunk tokens that live in ActionFrame.inputs
    # but don't map to a DiscreteAction. The canonical set lives in
    # gameeval.core.action_space.ACTION_INPUT_TOKENS.
    _FRAME_TOKEN_TO_KEY: dict[str, str] = {
        "shift": "shift",
        "4": "4",
        "q": "q",
        "z": "z",
    }
    _FRAME_TOKEN_TO_MOUSE: dict[str, str] = {
        "L": "left",
        "R": "right",
        "M": "middle",
    }

    def __init__(
        self,
        *,
        mouse_scale: float = 1.0,
        enabled: bool = True,
        hold_left_mouse: bool = False,
        use_key: str = "e",
    ) -> None:
        if enabled and not _IS_WINDOWS:
            raise RuntimeError(
                "WindowsInputController(enabled=True) requires Windows "
                "(SendInput). Run with enabled=False for dry-run on "
                "other platforms."
            )
        self._mouse_scale = float(mouse_scale)
        self._enabled = bool(enabled)
        # When True, the left mouse button uses hold-diff semantics
        # (same as the right button) instead of edge-triggered tap.
        # This lets a SHOOT pressed in the last frame of a chunk stay
        # held across the multi-second VLM inference gap → continuous
        # autofire between predictions.  Default False keeps the safer
        # tap behaviour.
        self._hold_left_mouse = bool(hold_left_mouse)
        self._use_key = str(use_key)
        self._pressed_keys: set[str] = set()
        self._pressed_mouse: set[str] = set()

    # ---- public API -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply(self, action: Action) -> dict[str, Any]:
        """Send the events needed to realise ``action`` this tick.

        Returns a dict summarising what was emitted, suitable for
        logging into the ``info`` channel of the environment step.
        """
        if not self._enabled:
            return {
                "skipped": True,
                "keys": [],
                "mouse": [],
                "dx": 0,
                "dy": 0,
            }

        want_keys, want_mouse = self._translate(action)

        # Keyboard diff — release first, then press. Releasing first
        # avoids transient "both A and D held" states when switching
        # strafe direction.
        for key in list(self._pressed_keys):
            if key not in want_keys:
                _key_event(key, down=False)
                self._pressed_keys.discard(key)
        for key in want_keys:
            if key not in self._pressed_keys:
                _key_event(key, down=True)
                self._pressed_keys.add(key)

        # Mouse-button handling.
        #
        # Left button ("L", SHOOT) is edge-triggered by default: one
        # full down+up per tick that requests it.  Without this a
        # SHOOT in the last frame of a VLM chunk would stay held
        # across the multi-second inference gap.  When the injector
        # is constructed with ``hold_left_mouse=True`` the left
        # button instead uses hold-diff semantics (same as right) —
        # useful when you WANT the trigger to keep firing between
        # inferences.
        #
        # Right button ("R", ADS / AWP scope) keeps hold-diff
        # semantics so scoping works as expected.
        HOLD_BUTTONS = {"right", "left"} if self._hold_left_mouse else {"right"}

        # 1) Release any previously-held hold-type button that is no
        #    longer requested.
        for btn in list(self._pressed_mouse):
            if btn in HOLD_BUTTONS and btn not in want_mouse:
                _mouse_button(btn, down=False)
                self._pressed_mouse.discard(btn)

        # 2) Press any newly-requested hold-type button.
        for btn in want_mouse:
            if btn in HOLD_BUTTONS and btn not in self._pressed_mouse:
                _mouse_button(btn, down=True)
                self._pressed_mouse.add(btn)

        # 3) Tap edge-triggered buttons (everything not in HOLD_BUTTONS,
        #    i.e. the left button) down+up in the same tick. We insert
        #    a short sleep between DOWN and UP so the click spans at
        #    least one the game engine tick (~15.6 ms at 64 tickrate);
        #    otherwise the game can see DOWN+UP in the same tick and the
        #    shot is lost. 20 ms is safely above one tick and well
        #    below the adapter's per-step sleep of 50 ms.
        _TAP_HOLD_SECONDS = 0.125
        for btn in want_mouse:
            if btn in HOLD_BUTTONS:
                continue
            _mouse_button(btn, down=True)
            time.sleep(_TAP_HOLD_SECONDS)
            _mouse_button(btn, down=False)

        # Relative mouse motion — rounded to int pixels. Skip the
        # SendInput call when the rounded delta is zero to avoid
        # spurious sub-pixel events.
        dx = int(round(action.mouse_dx * self._mouse_scale))
        dy = int(round(action.mouse_dy * self._mouse_scale))
        if dx or dy:
            _mouse_move_relative(dx, dy)

        return {
            "skipped": False,
            "keys": sorted(want_keys),
            "mouse": sorted(want_mouse),
            "dx": dx,
            "dy": dy,
        }

    def release_all(self) -> None:
        """Release every key / button the injector believes is held.

        Call at episode boundaries and on ``close`` — otherwise the
        operator's desktop will be left with ``W`` or the left mouse
        button stuck down after an evaluation ends.
        """
        if not self._enabled:
            self._pressed_keys.clear()
            self._pressed_mouse.clear()
            return

        for key in list(self._pressed_keys):
            try:
                _key_event(key, down=False)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("Failed to release key %r: %s", key, e)
        for btn in list(self._pressed_mouse):
            try:
                _mouse_button(btn, down=False)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("Failed to release mouse %r: %s", btn, e)
        self._pressed_keys.clear()
        self._pressed_mouse.clear()

    # ---- internals ------------------------------------------------------

    def _translate(self, action: Action) -> tuple[set[str], set[str]]:
        """Collapse ``action`` into (keys_to_hold, mouse_buttons_to_hold).

        Both ``action.discrete`` and ``action.frames[*].inputs`` are
        considered — the former is the canonical GameEval channel,
        the latter covers action chunk tokens like ``shift`` /
        ``R`` that have no DiscreteAction counterpart.
        """
        want_keys: set[str] = set()
        want_mouse: set[str] = set()

        for da in action.discrete:
            if da == DiscreteAction.SHOOT:
                want_mouse.add("left")
            elif da == DiscreteAction.USE:
                want_keys.add(self._use_key)
            else:
                key = self._DISCRETE_TO_KEY.get(da)
                if key is not None:
                    want_keys.add(key)

        # Frame tokens. We take the union across all frames because
        # Action corresponds to a single env tick; per-frame timing
        # within a chunk is the agent's responsibility.
        for frame in action.frames:
            for token in frame.inputs:
                key = self._FRAME_TOKEN_TO_KEY.get(token)
                if key is not None:
                    want_keys.add(key)
                btn = self._FRAME_TOKEN_TO_MOUSE.get(token)
                if btn is not None:
                    want_mouse.add(btn)

        return want_keys, want_mouse
