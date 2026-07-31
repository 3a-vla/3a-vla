"""Resolve a Windows window's client area into a screen-capture region.

GP-Visual runs against an emulator window rather than a full screen. Capturing
the whole monitor would fold the desktop and window chrome into the policy's
observation, so the capture region must be the game's client area.

Two coordinate systems meet here. Win32 reports geometry in the calling process's
coordinate space, which is *logical* (DPI-scaled) for a DPI-unaware process,
while screen grabbers such as ``mss`` always address *physical* pixels. Rather
than assuming a scale factor, this module derives one by comparing the desktop
size in both spaces, then verifies the result lands inside the physical desktop.

The region is returned in the ``{left, top, width, height}`` form used by
``gameeval.utils.screenshot.capture_screen``.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any

MIN_CAPTURE_EDGE = 64

_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


class WindowNotFoundError(RuntimeError):
    """Raised when no visible window matches the requested title."""


def _require_windows() -> Any:
    if sys.platform != "win32":
        raise RuntimeError("Window-region capture requires Windows")
    return ctypes.windll.user32


def physical_desktop_size() -> tuple[int, int] | None:
    """Return the virtual desktop size in physical pixels, if obtainable."""
    try:
        import mss

        with mss.mss() as sct:
            desktop = sct.monitors[0]
            return int(desktop["width"]), int(desktop["height"])
    except Exception:  # pragma: no cover - mss is optional at import time
        return None


def display_scale() -> float:
    """Return the factor converting this process's coordinates to physical pixels.

    A DPI-unaware process sees a virtualized desktop, so the ratio of physical to
    reported width gives the scale. A DPI-aware process already reports physical
    pixels and the ratio is 1.
    """
    user32 = _require_windows()
    logical_width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    physical = physical_desktop_size()
    if logical_width <= 0 or physical is None or physical[0] <= 0:
        return 1.0
    return physical[0] / logical_width


def _looks_physical(region: dict[str, int]) -> bool:
    """Whether a region is already expressed in physical pixels.

    A DPI-unaware process is shown a virtualized desktop, so any rect larger than
    that virtual desktop cannot be in virtualized coordinates and must already be
    physical. This happens when the emulator renders above the logical desktop
    size. Scaling such a rect again would run off-screen.
    """
    user32 = _require_windows()
    logical_width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    logical_height = int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    if logical_width <= 0 or logical_height <= 0:
        return False
    return region["width"] > logical_width or region["height"] > logical_height


def _scale_region(region: dict[str, int], scale: float) -> dict[str, int]:
    if abs(scale - 1.0) < 1e-6:
        return dict(region)
    return {
        "left": int(round(region["left"] * scale)),
        "top": int(round(region["top"] * scale)),
        "width": int(round(region["width"] * scale)),
        "height": int(round(region["height"] * scale)),
    }


def list_windows() -> list[dict[str, Any]]:
    """Return visible, titled windows with their reported client-area rects."""
    user32 = _require_windows()
    results: list[dict[str, Any]] = []

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):  # pragma: no cover - requires a live desktop
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)

        client = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(client))
        origin = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))

        results.append(
            {
                "hwnd": int(hwnd),
                "title": buffer.value,
                "class_name": class_buffer.value,
                "region": {
                    "left": int(origin.x),
                    "top": int(origin.y),
                    "width": int(client.right),
                    "height": int(client.bottom),
                },
            }
        )
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return results


def find_window_region(title_contains: str) -> dict[str, Any]:
    """Find one visible window whose title contains ``title_contains``.

    A substring match keeps task files stable across emulator version strings,
    but an ambiguous match is an error: silently capturing the wrong window
    would invalidate every episode in the run.
    """
    needle = str(title_contains).strip()
    if not needle:
        raise ValueError("title_contains must be a non-empty string")

    lowered = needle.lower()
    matches = [
        window
        for window in list_windows()
        if lowered in window["title"].lower()
        and window["region"]["width"] >= MIN_CAPTURE_EDGE
        and window["region"]["height"] >= MIN_CAPTURE_EDGE
    ]
    if not matches:
        raise WindowNotFoundError(
            f"No visible window title contains {needle!r}. Start the game, "
            "leave it windowed, and keep it un-minimized."
        )
    if len(matches) > 1:
        titles = ", ".join(repr(window["title"]) for window in matches)
        raise WindowNotFoundError(
            f"Window title {needle!r} is ambiguous across: {titles}"
        )
    return matches[0]


def occluding_window(window: dict[str, Any], region: dict[str, int]) -> str | None:
    """Return the title of a window covering ``region``'s centre, if any.

    Screen capture reads the desktop composite, so a covered game window yields
    whatever is painted on top of it. Detecting that is essential: otherwise an
    entire run silently substitutes unrelated pixels for the policy's
    observation and the judge's evidence.
    """
    user32 = _require_windows()
    hwnd = int(window["hwnd"])

    if user32.IsIconic(hwnd):
        return "(minimized)"

    centre = wintypes.POINT(
        int(region["left"] + region["width"] // 2),
        int(region["top"] + region["height"] // 2),
    )
    # WindowFromPoint expects this process's coordinate space, so undo any
    # physical-pixel scaling applied to the capture region.
    scale = display_scale()
    if abs(scale - 1.0) > 1e-6:
        centre.x = int(round(centre.x / scale))
        centre.y = int(round(centre.y / scale))

    top = user32.WindowFromPoint(centre)
    if not top:
        return None
    root = user32.GetAncestor(top, 2)  # GA_ROOT
    if int(root) == hwnd:
        return None

    buffer = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(root, buffer, 256)
    return buffer.value or f"hwnd {int(root)}"


def resolve_capture_region(
    title_contains: str,
    *,
    expected_size: tuple[int, int] | None = None,
    scale: float | None = None,
    require_unobstructed: bool = False,
) -> dict[str, int]:
    """Resolve a physical-pixel capture region for a window's client area.

    ``expected_size`` is checked against the client size actually used for
    capture, so a task pins the pixel geometry the policy and judge observe.
    ``require_unobstructed`` additionally refuses to capture a covered window.
    """
    window = find_window_region(title_contains)
    reported = window["region"]

    if scale is not None:
        region = _scale_region(reported, float(scale))
    elif _looks_physical(reported):
        region = dict(reported)
    else:
        region = _scale_region(reported, display_scale())

    physical = physical_desktop_size()
    if physical is not None:
        width, height = physical
        if (
            region["left"] < 0
            or region["top"] < 0
            or region["left"] + region["width"] > width
            or region["top"] + region["height"] > height
        ):
            raise RuntimeError(
                f"Resolved capture region {region} falls outside the "
                f"{width}x{height} desktop. Move the game window fully on-screen."
            )

    if expected_size is not None:
        want_width, want_height = (int(expected_size[0]), int(expected_size[1]))
        if (region["width"], region["height"]) != (want_width, want_height):
            raise RuntimeError(
                f"Window {window['title']!r} captures at "
                f"{region['width']}x{region['height']}, but the task declares "
                f"{want_width}x{want_height}. Fix the emulator resolution or "
                "update the task's setup.resolution before reporting results."
            )

    if require_unobstructed:
        blocker = occluding_window(window, region)
        if blocker is not None:
            raise RuntimeError(
                f"Window {window['title']!r} is covered by {blocker!r}. Screen "
                "capture reads the desktop composite, so the rollout would "
                "record the wrong pixels. Bring the game window to the front "
                "and keep it unobstructed."
            )
    return region


__all__ = [
    "WindowNotFoundError",
    "display_scale",
    "find_window_region",
    "list_windows",
    "occluding_window",
    "physical_desktop_size",
    "resolve_capture_region",
]
