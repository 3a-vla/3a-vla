"""CSGO window capture with monitor fallback and optional image preprocessing."""

from __future__ import annotations

import ctypes
from typing import Optional

from PIL import Image

try:
    import mss
except ImportError:  # pragma: no cover - checked when a runtime connects
    mss = None

try:
    import psutil
    import win32con
    import win32gui
    import win32process
    import win32ui

    _HAS_WIN32 = True
except ImportError:  # pragma: no cover - monitor capture remains available
    _HAS_WIN32 = False


DEFAULT_PROCESS = "csgo.exe"


class CSScreenCapture:
    """Capture a CSGO window and optionally center-crop or resize it."""

    def __init__(
        self,
        crop_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        process_name: str | None = DEFAULT_PROCESS,
        debug: bool = False,
    ) -> None:
        self.crop_size = tuple(crop_size) if crop_size else None
        self.target_size = tuple(target_size) if target_size else None
        self.process_name = process_name or None
        self.debug = bool(debug)

        if mss is None:
            raise RuntimeError("CSGO capture requires: pip install 'gameeval[csgo]'")
        self._sct = mss.mss()
        self.hwnd: Optional[int] = None
        if self.process_name and _HAS_WIN32:
            self.hwnd = self._find_window_by_process(self.process_name)

    @staticmethod
    def _find_window_by_process(process_name: str) -> Optional[int]:
        if not _HAS_WIN32:
            return None

        target_pids = set()
        for process in psutil.process_iter(["pid", "name"]):
            try:
                name = process.info["name"]
                if name and name.lower() == process_name.lower():
                    target_pids.add(process.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not target_pids:
            return None

        result_hwnd = None

        def enum_callback(hwnd, _):
            nonlocal result_hwnd
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid in target_pids:
                    result_hwnd = hwnd
                    return not bool(win32gui.GetWindowText(hwnd))
            except Exception:
                return True
            return True

        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception:
            pass
        return result_hwnd

    def refresh_window(self) -> bool:
        if not (self.process_name and _HAS_WIN32):
            return False
        self.hwnd = self._find_window_by_process(self.process_name)
        return self.hwnd is not None

    def capture(self) -> Optional[Image.Image]:
        if self.hwnd and _HAS_WIN32:
            image = self._capture_window(self.hwnd)
            if image is not None:
                return image
            if self.refresh_window() and self.hwnd:
                image = self._capture_window(self.hwnd)
                if image is not None:
                    return image
        return self._capture_fullscreen()

    def _capture_fullscreen(self) -> Optional[Image.Image]:
        try:
            monitor = self._sct.monitors[1]
            shot = self._sct.grab(monitor)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as exc:
            if self.debug:
                print(f"[CSScreenCapture] monitor capture failed: {exc}")
            return None

    def _capture_window(self, hwnd: int) -> Optional[Image.Image]:
        if not _HAS_WIN32:
            return None
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                return None

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)

            result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
            if result == 0:
                save_dc.BitBlt(
                    (0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY
                )

            bitmap_info = bitmap.GetInfo()
            bitmap_bytes = bitmap.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (bitmap_info["bmWidth"], bitmap_info["bmHeight"]),
                bitmap_bytes,
                "raw",
                "BGRX",
                0,
                1,
            )

            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            return image
        except Exception as exc:
            if self.debug:
                print(f"[CSScreenCapture] window capture failed: {exc}")
            return None

    def process_to_pil(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if self.crop_size is not None:
                crop_width, crop_height = self.crop_size
                width, height = image.size
                if width >= crop_width and height >= crop_height:
                    left = (width - crop_width) // 2
                    top = (height - crop_height) // 2
                    image = image.crop(
                        (left, top, left + crop_width, top + crop_height)
                    )
            if self.target_size is not None and image.size != self.target_size:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as exc:
            if self.debug:
                print(f"[CSScreenCapture] preprocessing failed: {exc}")
            return None

    @property
    def has_window(self) -> bool:
        return self.hwnd is not None

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


__all__ = ["CSScreenCapture", "DEFAULT_PROCESS"]
