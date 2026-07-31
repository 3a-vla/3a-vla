"""Game-independent result types for state and visual evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationStatus(str, Enum):
    """Terminal outcome of an evaluation pass."""

    SUCCESS = "success"
    FAIL = "fail"
    UNKNOWN = "unknown"
    ERROR = "error"

    @classmethod
    def from_value(cls, value: str | bool | "EvaluationStatus") -> "EvaluationStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.SUCCESS if value else cls.FAIL
        normalized = str(value).strip().lower()
        aliases = {
            "pass": cls.SUCCESS,
            "passed": cls.SUCCESS,
            "succeeded": cls.SUCCESS,
            "failure": cls.FAIL,
            "failed": cls.FAIL,
            "abstain": cls.UNKNOWN,
            "ambiguous": cls.UNKNOWN,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class EvaluatorType(str, Enum):
    """Supported evaluator family."""

    STATE = "state"
    VLM = "vlm"


@dataclass
class Evidence:
    """One piece of evidence supporting an evaluation outcome."""

    kind: str
    value: Any
    timestamp: float | None = None
    frame_index: int | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "description": self.description,
        }


@dataclass
class EvaluationResult:
    """Game-independent result emitted by state and visual evaluators."""

    status: EvaluationStatus = EvaluationStatus.UNKNOWN
    reason: str = ""
    confidence: float = 0.0
    evaluator: EvaluatorType = EvaluatorType.STATE
    score: float | None = None
    terminal: bool = True
    evidence: list[Evidence | dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether evaluation produced an authoritative success."""
        return self.status is EvaluationStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        evidence = [
            item.to_dict() if isinstance(item, Evidence) else item
            for item in self.evidence
        ]
        return {
            "status": self.status.value,
            "success": self.success,
            "reason": self.reason,
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "evaluator": self.evaluator.value,
            "score": self.score,
            "terminal": self.terminal,
            "evidence": evidence,
            "metrics": self.metrics,
            "details": self.details,
        }
