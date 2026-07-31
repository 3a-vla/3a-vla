"""Combined runtime and state-provider interface for state-backed games.

CSGO and GP implement this combined runtime/state-provider bridge so privileged
telemetry stays separate from agent observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from gameeval.core.runtime import GameRuntime, RuntimeConfig, StateProvider


@dataclass
class AdapterConfig(RuntimeConfig):
    """Configuration passed to :meth:`GameAdapter.connect`.

    Subclasses may extend this with game-specific fields.
    """

    pass


class GameAdapter(GameRuntime, StateProvider, ABC):
    """Abstract base for adapters that expose an evaluator-only state channel.

    Lifecycle
    ---------
    1. ``connect(config)``  – establish connection to game server / engine.
    2. ``reset(task_config)`` – set up the scene for a specific task.
    3. Loop: ``step(action)`` → new observation + raw state.
    4. ``close()`` – tear down gracefully.

    Subclass implementors can use ``gameeval/adapters/csgo/`` as an
    interface-backed reference implementation.
    """

    # ---- Connection lifecycle ------------------------------------------------

    @abstractmethod
    def connect(self, config: AdapterConfig) -> None:
        """Connect to the game server or engine.

        Parameters
        ----------
        config : AdapterConfig
            Connection parameters (host, port, credentials, etc.).
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources and disconnect."""

    # ---- Episode lifecycle ---------------------------------------------------

    @abstractmethod
    def reset(self, task_config: dict) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        The adapter may invoke a private setup hook or attach to a scene that
        an operator prepared, then returns the initial frame and state.

        Parameters
        ----------
        task_config : dict
            Parsed task configuration (from YAML).

        Returns
        -------
        screenshot : np.ndarray
            Initial RGB screenshot, shape ``(H, W, 3)``, dtype ``uint8``.
        state : dict
            Initial structured game state.
        """

    @abstractmethod
    def step(self, action: dict) -> tuple[np.ndarray, dict, dict]:
        """Execute one action tick in the game.

        Parameters
        ----------
        action : dict
            Unified action dictionary (see :mod:`gameeval.core.action_space`).

        Returns
        -------
        screenshot : np.ndarray
            New RGB screenshot after executing the action.
        state : dict
            New structured game state.
        info : dict
            Extra per-step metadata (e.g. reward from engine, raw events).
        """

    # ---- State access --------------------------------------------------------

    @abstractmethod
    def get_state(self) -> dict:
        """Return the latest structured game state without advancing a tick."""

    # ---- Split runtime / state-provider views -------------------------------

    def reset_runtime(self, task_config: dict[str, Any]) -> np.ndarray | None:
        """Expose only the initial frame through the agent-facing runtime."""
        screenshot, _state = self.reset(task_config)
        return screenshot

    def step_runtime(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Execute a step without exposing privileged state to the agent."""
        screenshot, _state, info = self.step(action)
        return screenshot, info

    def snapshot(self) -> dict[str, Any]:
        """Expose ``get_state`` only through the privileged state provider."""
        return self.get_state()

    @abstractmethod
    def screenshot(self) -> np.ndarray:
        """Capture the current screen and return an RGB array.

        Returns
        -------
        np.ndarray
            Shape ``(H, W, 3)``, dtype ``uint8``.
        """

    # ---- Metadata ------------------------------------------------------------

    @property
    @abstractmethod
    def game_name(self) -> str:
        """Short identifier, e.g. ``'csgo'`` or ``'gp'``."""

    @property
    def is_connected(self) -> bool:  # noqa: D401
        """Whether the adapter currently holds a live connection."""
        return False

    # ---- Optional helpers ----------------------------------------------------

    def execute_command(self, command: str) -> str | None:
        """Send a raw command to the game engine (e.g. RCON).

        Not all adapters need to support this.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support raw commands.")
