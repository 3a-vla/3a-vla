"""Game-independent condition DSL for privileged-state evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ConditionError(ValueError):
    """Raised when a condition tree is malformed or uses an unknown operator."""


@dataclass
class ConditionEvaluation:
    success: bool
    reason: str
    condition_types: set[str] = field(default_factory=set)
    trace: list[dict[str, Any]] = field(default_factory=list)


def get_path(value: Any, path: str) -> Any:
    """Resolve a dotted path through dictionaries and list indices."""
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


class ConditionEngine:
    """Evaluate declarative conditions against a normalized state snapshot.

    Supported composite operators are ``AND``, ``OR``, and ``NOT``. Leaf
    operators are ``field_exists``, ``field_equals``, ``field_in``, numeric
    comparisons (``field_gt/gte/lt/lte``), numeric change comparisons
    (``field_delta_gt/gte/lt/lte``), and ``event_count``.
    """

    def evaluate(
        self,
        state: dict[str, Any],
        condition: dict[str, Any],
        initial_state: dict[str, Any] | None = None,
    ) -> ConditionEvaluation:
        if not isinstance(condition, dict):
            raise ConditionError("Condition must be a mapping")
        condition_type = str(condition.get("type", "")).strip()
        if not condition_type:
            raise ConditionError("Condition is missing 'type'")

        upper = condition_type.upper()
        if upper in {"AND", "OR"}:
            children = condition.get("conditions")
            if not isinstance(children, list) or not children:
                raise ConditionError(f"{upper} requires a non-empty 'conditions' list")
            results = [self.evaluate(state, child, initial_state) for child in children]
            success = all(item.success for item in results) if upper == "AND" else any(
                item.success for item in results
            )
            types = {upper}
            trace: list[dict[str, Any]] = []
            for item in results:
                types.update(item.condition_types)
                trace.extend(item.trace)
            return ConditionEvaluation(
                success=success,
                reason=f"{upper} {'passed' if success else 'failed'}",
                condition_types=types,
                trace=trace,
            )

        if upper == "NOT":
            child = condition.get("condition")
            if not isinstance(child, dict):
                raise ConditionError("NOT requires a 'condition' mapping")
            result = self.evaluate(state, child, initial_state)
            success = not result.success
            return ConditionEvaluation(
                success=success,
                reason=f"NOT {'passed' if success else 'failed'}",
                condition_types={"NOT", *result.condition_types},
                trace=result.trace,
            )

        success, actual, expected = self._evaluate_leaf(
            state,
            condition_type,
            condition,
            initial_state,
        )
        trace = {
            "type": condition_type,
            "path": condition.get("path"),
            "actual": actual,
            "expected": expected,
            "success": success,
        }
        return ConditionEvaluation(
            success=success,
            reason=f"{condition_type} {'passed' if success else 'failed'}",
            condition_types={condition_type},
            trace=[trace],
        )

    def _evaluate_leaf(
        self,
        state: dict[str, Any],
        condition_type: str,
        condition: dict[str, Any],
        initial_state: dict[str, Any] | None,
    ) -> tuple[bool, Any, Any]:
        if condition_type == "event_count":
            event_name = str(condition.get("event", ""))
            minimum = int(condition.get("min_count", 1))
            events = get_path(state, str(condition.get("path", "events"))) or []
            actual = sum(
                1
                for event in events
                if isinstance(event, dict) and str(event.get("type", "")) == event_name
            )
            return actual >= minimum, actual, minimum

        path = str(condition.get("path", ""))
        if not path:
            raise ConditionError(f"{condition_type} requires 'path'")
        actual = get_path(state, path)
        expected = condition.get("value")

        delta_comparisons = {
            "field_delta_gt": lambda left, right: left > right,
            "field_delta_gte": lambda left, right: left >= right,
            "field_delta_lt": lambda left, right: left < right,
            "field_delta_lte": lambda left, right: left <= right,
        }
        delta_comparison = delta_comparisons.get(condition_type)
        if delta_comparison is not None:
            initial = get_path(initial_state or {}, path)
            try:
                delta = float(actual) - float(initial)
                threshold = float(expected)
            except (TypeError, ValueError) as exc:
                raise ConditionError(
                    f"{condition_type} requires numeric initial/final/value at '{path}'"
                ) from exc
            return (
                bool(delta_comparison(delta, threshold)),
                {"initial": initial, "final": actual, "delta": delta},
                expected,
            )

        if condition_type == "field_exists":
            expected = bool(condition.get("value", True))
            return (actual is not None) is expected, actual, expected
        if condition_type == "field_equals":
            return actual == expected, actual, expected
        if condition_type == "field_in":
            values = condition.get("values")
            if not isinstance(values, list):
                raise ConditionError("field_in requires a 'values' list")
            return actual in values, actual, values

        comparisons = {
            "field_gt": lambda left, right: left > right,
            "field_gte": lambda left, right: left >= right,
            "field_lt": lambda left, right: left < right,
            "field_lte": lambda left, right: left <= right,
        }
        comparison = comparisons.get(condition_type)
        if comparison is not None:
            try:
                left = float(actual)
                right = float(expected)
            except (TypeError, ValueError) as exc:
                raise ConditionError(
                    f"{condition_type} requires numeric actual/value at '{path}'"
                ) from exc
            return bool(comparison(left, right)), actual, expected

        raise ConditionError(f"Unsupported condition type: {condition_type}")
