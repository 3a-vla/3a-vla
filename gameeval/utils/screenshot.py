"""Screenshot capture utilities.

Uses `mss` for cross-platform screen capture and Pillow for processing.
"""

from __future__ import annotations

import numpy as np


def capture_screen(
    region: dict[str, int] | None = None,
    monitor_index: int = 1,
) -> np.ndarray:
    """Capture a screenshot and return as an RGB numpy array.

    Parameters
    ----------
    region : dict | None
        Capture region ``{"left": int, "top": int, "width": int, "height": int}``.
        If *None*, captures the full monitor.
    monitor_index : int
        Which monitor to capture (1-based; 0 = all monitors combined).

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``, RGB order.
    """
    import mss

    with mss.mss() as sct:
        if region is not None:
            monitor = region
        else:
            monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
        # mss returns BGRA; convert to RGB
        img = np.array(raw)[:, :, :3][:, :, ::-1].copy()
    return img


def encode_image_base64(img: np.ndarray, fmt: str = "jpeg") -> str:
    """Encode an RGB array to a base64 string (for VLM API calls).

    Parameters
    ----------
    img : np.ndarray
        RGB image.
    fmt : str
        Image format: ``"jpeg"`` or ``"png"``.

    Returns
    -------
    str
        Base64-encoded image string.
    """
    import base64
    import io

    from PIL import Image

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
