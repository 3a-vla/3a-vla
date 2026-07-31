"""GameEval: extensible VLA evaluation for CSGO, GTA5, and GP."""

__version__ = "0.3.0"

from gameeval.core.action_space import Action, GameActionSpace
from gameeval.core.evaluation import EvaluationResult, EvaluationStatus, EvaluatorType
from gameeval.core.game_adapter import GameAdapter
from gameeval.core.game_env import GameEvalEnv
from gameeval.core.observation import Observation, ObservationConfig
from gameeval.core.runtime import GameRuntime, RuntimeConfig, StateProvider

__all__ = [
    "GameEvalEnv",
    "GameAdapter",
    "GameRuntime",
    "RuntimeConfig",
    "StateProvider",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluatorType",
    "Action",
    "GameActionSpace",
    "Observation",
    "ObservationConfig",
    "__version__",
]
