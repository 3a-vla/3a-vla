"""Deterministic evaluator for games with a privileged state interface."""

from __future__ import annotations

from typing import Any

from gameeval.core.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorType,
    Evidence,
)
from gameeval.evaluators.base import BaseEvaluator, EpisodeContext
from gameeval.evaluators.conditions import ConditionEngine, ConditionError


class StateEvaluator(BaseEvaluator):
    """Evaluate a task condition tree against normalized hidden state."""

    def __init__(self) -> None:
        self._condition_engine = ConditionEngine()

    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.STATE

    def evaluate(self, episode: EpisodeContext) -> EvaluationResult:
        state = episode.state
        if state is None:
            return EvaluationResult(
                status=EvaluationStatus.UNKNOWN,
                reason="State evaluator selected but no StateProvider is available",
                confidence=1.0,
                evaluator=self.evaluator_type,
            )

        evaluator_cfg = episode.task.get("evaluator", {}) or {}
        conditions = evaluator_cfg.get("conditions")
        if conditions is None:
            return EvaluationResult(
                status=EvaluationStatus.ERROR,
                reason="State evaluator has no condition tree",
                confidence=1.0,
                evaluator=self.evaluator_type,
            )

        try:
            condition_result = self._condition_engine.evaluate(
                state,
                conditions,
                initial_state=episode.initial_state,
            )
        except ConditionError as exc:
            return EvaluationResult(
                status=EvaluationStatus.ERROR,
                reason=str(exc),
                confidence=1.0,
                evaluator=self.evaluator_type,
            )
        return EvaluationResult(
            status=(
                EvaluationStatus.SUCCESS
                if condition_result.success
                else EvaluationStatus.FAIL
            ),
            reason=condition_result.reason,
            confidence=1.0,
            evaluator=self.evaluator_type,
            score=1.0 if condition_result.success else 0.0,
            evidence=[
                Evidence(
                    kind="state_conditions",
                    value={
                        "types": sorted(condition_result.condition_types),
                        "trace": condition_result.trace,
                    },
                    description="Condition types evaluated against the hidden state provider",
                )
            ],
            metrics=self._extract_metrics(state),
        )

    @staticmethod
    def _extract_metrics(state: dict[str, Any]) -> dict[str, Any]:
        metrics = state.get("metrics", {})
        return dict(metrics) if isinstance(metrics, dict) else {}
