"""Game-independent evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gameeval.core.evaluation import EvaluationResult, EvaluatorType


@dataclass
class EpisodeContext:
    """Evidence bundle passed to exactly one evaluator per task."""

    task: dict[str, Any]
    step_index: int = 0
    initial_state: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    frames: list[np.ndarray] = field(default_factory=list)
    video_path: Path | None = None
    actions_path: Path | None = None
    episode_dir: Path | None = None
    runtime_info: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Evaluate one completed episode."""

    @property
    @abstractmethod
    def evaluator_type(self) -> EvaluatorType:
        """Evaluator family used in reports and result artifacts."""

    @abstractmethod
    def evaluate(self, episode: EpisodeContext) -> EvaluationResult:
        """Evaluate a completed episode synchronously."""

    async def evaluate_async(self, episode: EpisodeContext) -> EvaluationResult:
        """Default async wrapper for synchronous evaluators."""
        return self.evaluate(episode)
