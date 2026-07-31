"""Core abstractions for GameEval."""

from gameeval.core.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorType,
    Evidence,
)
from gameeval.core.runtime import GameRuntime, RuntimeConfig, StateProvider

__all__ = [
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluatorType",
    "Evidence",
    "GameRuntime",
    "RuntimeConfig",
    "StateProvider",
]
