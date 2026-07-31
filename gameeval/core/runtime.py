"""Game execution and privileged-state interfaces.

The agent-facing runtime is intentionally separate from the evaluator's state
provider. This prevents evaluator-only telemetry from leaking into VLA observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RuntimeConfig:
    game: str = ""
    host: str = "127.0.0.1"
    port: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class GameRuntime(ABC):
    """Agent-facing lifecycle for a local, remote, or streamed game client."""

    @abstractmethod
    def connect(self, config: RuntimeConfig) -> None:
        """Connect to or initialize the game runtime."""

    @abstractmethod
    def close(self) -> None:
        """Release capture, control, and game resources."""

    @abstractmethod
    def reset_runtime(self, task_config: dict[str, Any]) -> np.ndarray | None:
        """Reset an episode and return the first visual frame."""

    @abstractmethod
    def step_runtime(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Execute one action and return the next frame plus public metadata."""

    @abstractmethod
    def screenshot(self) -> np.ndarray | None:
        """Capture a frame without advancing the environment."""

    @property
    @abstractmethod
    def game_name(self) -> str:
        """Stable short game identifier."""

    @property
    def is_connected(self) -> bool:
        return False


class StateProvider(ABC):
    """Privileged state channel available only to state evaluators."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return the latest normalized game state."""
