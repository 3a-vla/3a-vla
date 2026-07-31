"""Unified observation space for GameEval.

An observation in FPS games typically consists of:

1. **Visual**: an RGB screenshot (the primary modality for VLA models).
2. **Structured state**: an optional debug field that remains empty for normal
   evaluation; evaluator-only state travels through a separate provider.

This module defines :class:`Observation` as a container for both modalities
and :class:`ObservationConfig` to control which fields are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObservationConfig:
    """Controls what is included in each :class:`Observation`.

    Attributes
    ----------
    include_screenshot : bool
        Whether to capture and include the RGB screenshot.
    screenshot_width : int
        Target width; the adapter may resize.
    screenshot_height : int
        Target height.
    include_state : bool
        Whether to include structured game state.
    state_fields : list[str] | None
        If given, only include these top-level state keys.
        ``None`` means include everything the adapter provides.
    include_minimap : bool
        Whether to include a minimap / radar image (game-dependent).
    """

    include_screenshot: bool = True
    screenshot_width: int = 640
    screenshot_height: int = 360
    # Privileged engine state is hidden from agents by default.  It is
    # consumed through StateProvider by the evaluator instead.
    include_state: bool = False
    state_fields: list[str] | None = None
    include_minimap: bool = False


@dataclass
class Observation:
    """A single observation from the FPS environment.

    Attributes
    ----------
    screenshot : np.ndarray | None
        RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
    state : dict[str, Any]
        Structured game state.  Contents are game-dependent.
        Common keys (when available):

        - ``player_health`` (int)
        - ``player_armor`` (int)
        - ``player_position`` (dict with x, y, z)
        - ``player_velocity`` (dict with x, y, z)
        - ``player_yaw`` (float)
        - ``player_pitch`` (float)
        - ``current_weapon`` (str)
        - ``ammo_clip`` (int)
        - ``ammo_reserve`` (int)
        - ``enemies`` (list[dict])  – per-enemy info if available
        - ``score`` (dict)
        - ``round`` (int)
        - ``time_left`` (float)
    minimap : np.ndarray | None
        Optional minimap / radar image.
    timestamp : float
        Wall-clock time of capture (``time.time()``).
    step_index : int
        The step index within the current episode.
    metadata : dict[str, Any]
        Extra per-observation info.
    """

    screenshot: np.ndarray | None = None
    state: dict[str, Any] = field(default_factory=dict)
    minimap: np.ndarray | None = None
    timestamp: float = 0.0
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- Convenience --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (screenshots → shape only)."""
        d: dict[str, Any] = {
            "state": self.state,
            "timestamp": self.timestamp,
            "step_index": self.step_index,
            "metadata": self.metadata,
        }
        if self.screenshot is not None:
            d["screenshot_shape"] = list(self.screenshot.shape)
        if self.minimap is not None:
            d["minimap_shape"] = list(self.minimap.shape)
        return d

    @property
    def has_screenshot(self) -> bool:
        return self.screenshot is not None

    @property
    def image_for_vlm(self) -> np.ndarray:
        """Return the screenshot as-is, or raise if unavailable."""
        if self.screenshot is None:
            raise ValueError("No screenshot in this observation.")
        return self.screenshot


def resize_screenshot(
    img: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize an RGB image to the target resolution.

    Uses Pillow for high-quality resizing.
    """
    from PIL import Image

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    return np.array(pil_img)
