"""GameEvalEnv — the unified Gym-like environment for GameEval.

This is the main entry point that agents interact with. It delegates to a
concrete :class:`~gameeval.core.runtime.GameRuntime`; privileged state is
kept on a separate evaluator-only channel.

Example
-------
>>> from gameeval import GameEvalEnv
>>> env = GameEvalEnv(adapter=runtime, adapter_config=config)
>>> obs = env.reset({"task_id": "example"})
>>> for _ in range(env.max_steps):
...     action = my_agent.act(obs)
...     obs, reward, done, info = env.step(action)
...     if done:
...         break
>>> env.close()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from gameeval.core.action_space import Action, GameActionSpace
from gameeval.core.game_adapter import AdapterConfig
from gameeval.core.observation import Observation, ObservationConfig, resize_screenshot
from gameeval.core.runtime import GameRuntime, RuntimeConfig, StateProvider

logger = logging.getLogger("gameeval.env")


@dataclass
class EpisodeInfo:
    """Metadata accumulated during a single episode."""

    task_id: str = ""
    game: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_steps: int = 0
    done: bool = False


class GameEvalEnv:
    """Unified environment wrapping an agent-facing :class:`GameRuntime`.

    Parameters
    ----------
    adapter : GameRuntime
        A concrete game runtime.
    adapter_config : RuntimeConfig | dict | None
        Config passed to ``runtime.connect()``.
    obs_config : ObservationConfig | None
        Observation settings (resolution, included fields, …).
    action_space : GameActionSpace | None
        Override the default action space.
    max_steps : int
        Hard episode step limit.
    """

    def __init__(
        self,
        adapter: GameRuntime,
        adapter_config: AdapterConfig | dict | None = None,
        state_provider: StateProvider | None = None,
        obs_config: ObservationConfig | None = None,
        action_space: GameActionSpace | None = None,
        max_steps: int = 1000,
    ):
        self.runtime = adapter
        self.state_provider = state_provider
        if self.state_provider is None and isinstance(adapter, StateProvider):
            self.state_provider = adapter
        self.obs_config = obs_config or ObservationConfig()
        self.action_space = action_space or GameActionSpace()
        self.max_steps = max_steps

        # Internal state
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_info: EpisodeInfo | None = None
        self._task_config: dict = {}
        self._screenshots: list[np.ndarray] = []
        self._connected: bool = False
        self._oracle_state: dict[str, Any] = {}

        # Connect adapter
        if adapter_config is not None:
            cfg = (
                adapter_config
                if isinstance(adapter_config, AdapterConfig)
                else RuntimeConfig(**adapter_config)
            )
            self.runtime.connect(cfg)
            self._connected = True

    # ---- Gym-like API --------------------------------------------------------

    def reset(self, task_config: dict | None = None, **kwargs: Any) -> Observation:
        """Reset the environment and start a new episode.

        Parameters
        ----------
        task_config : dict | None
            Parsed task configuration.  If *None*, re-uses the last config.

        Returns
        -------
        Observation
            Initial observation.
        """
        if task_config is not None:
            self._task_config = task_config

        # Reset only the agent-facing runtime. Privileged state is pulled
        # independently and is never placed in the observation unless a
        # caller explicitly enables it for debugging.
        screenshot = self.runtime.reset_runtime(self._task_config)
        state = self._snapshot_oracle()

        # Resize if needed
        if (
            self.obs_config.include_screenshot
            and screenshot is not None
            and (
                screenshot.shape[1] != self.obs_config.screenshot_width
                or screenshot.shape[0] != self.obs_config.screenshot_height
            )
        ):
            screenshot = resize_screenshot(
                screenshot,
                self.obs_config.screenshot_width,
                self.obs_config.screenshot_height,
            )

        # Filter state fields
        if self.obs_config.state_fields is not None:
            state = {k: v for k, v in state.items() if k in self.obs_config.state_fields}

        # Build observation
        obs = Observation(
            screenshot=screenshot if self.obs_config.include_screenshot else None,
            state=state if self.obs_config.include_state else {},
            timestamp=time.time(),
            step_index=0,
        )

        # Init episode tracking
        self._step_count = 0
        self._episode_count += 1
        self._screenshots = []
        if screenshot is not None:
            self._screenshots.append(screenshot)

        self._episode_info = EpisodeInfo(
            task_id=self._task_config.get("task_id", ""),
            game=self.runtime.game_name,
            start_time=time.time(),
        )

        logger.info(
            "Episode %d started — task=%s game=%s",
            self._episode_count,
            self._episode_info.task_id,
            self._episode_info.game,
        )
        return obs

    def step(self, action: Action | dict) -> tuple[Observation, float, bool, dict]:
        """Execute one action tick.

        Parameters
        ----------
        action : Action | dict
            The agent's action.

        Returns
        -------
        obs : Observation
        reward : float
            Reserved for runtime-provided online reward. Authoritative task
            evaluation happens after the rollout.
        done : bool
            Whether the episode has ended.
        info : dict
            Public runtime metadata. Privileged state is not included.
        """
        if isinstance(action, dict):
            action = Action.from_dict(action)

        self._step_count += 1

        # Execute in game
        screenshot, step_info = self.runtime.step_runtime(action.to_dict())
        state = self._snapshot_oracle()

        # Resize
        if (
            self.obs_config.include_screenshot
            and screenshot is not None
            and (
                screenshot.shape[1] != self.obs_config.screenshot_width
                or screenshot.shape[0] != self.obs_config.screenshot_height
            )
        ):
            screenshot = resize_screenshot(
                screenshot,
                self.obs_config.screenshot_width,
                self.obs_config.screenshot_height,
            )

        if screenshot is not None:
            self._screenshots.append(screenshot)

        # Filter state
        if self.obs_config.state_fields is not None:
            state = {k: v for k, v in state.items() if k in self.obs_config.state_fields}

        obs = Observation(
            screenshot=screenshot if self.obs_config.include_screenshot else None,
            state=state if self.obs_config.include_state else {},
            timestamp=time.time(),
            step_index=self._step_count,
        )

        # Check termination: max steps
        done = self._step_count >= self.max_steps
        reward = 0.0
        info: dict[str, Any] = {**step_info}

        # Check termination from adapter info (e.g. player dead)
        if step_info.get("done", False):
            done = True

        info["step"] = self._step_count
        info["max_steps"] = self.max_steps

        if done and self._episode_info is not None:
            self._episode_info.end_time = time.time()
            self._episode_info.total_steps = self._step_count
            self._episode_info.done = True
            info["episode"] = {
                "task_id": self._episode_info.task_id,
                "total_steps": self._step_count,
                "duration": self._episode_info.end_time - self._episode_info.start_time,
            }
            logger.info(
                "Episode %d ended — steps=%d",
                self._episode_count,
                self._step_count,
            )

        return obs, reward, done, info

    def close(self) -> None:
        """Close the environment and release resources."""
        if self._connected:
            self.runtime.close()
            self._connected = False
        logger.info("GameEvalEnv closed.")

    # ---- Properties ----------------------------------------------------------

    @property
    def game_name(self) -> str:
        return self.runtime.game_name

    @property
    def oracle_state(self) -> dict[str, Any]:
        """Latest privileged state for evaluators and metrics only."""
        return dict(self._oracle_state)

    @property
    def current_step(self) -> int:
        return self._step_count

    @property
    def episode_screenshots(self) -> list[np.ndarray]:
        """All screenshots captured during the current episode."""
        return self._screenshots

    @property
    def episode_info(self) -> EpisodeInfo | None:
        return self._episode_info

    def _snapshot_oracle(self) -> dict[str, Any]:
        if self.state_provider is None:
            self._oracle_state = {}
        else:
            self._oracle_state = self.state_provider.snapshot() or {}
        return dict(self._oracle_state)

    # ---- Context manager -----------------------------------------------------

    def __enter__(self) -> GameEvalEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
