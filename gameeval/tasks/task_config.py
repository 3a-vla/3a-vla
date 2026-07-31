"""Portable, user-defined episode specifications.

GameEval intentionally ships no canonical task suite.  A task file describes
one reproducible episode and selects exactly one authoritative judge backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gameeval.protocols import expected_evaluator, resolve_protocol

SUPPORTED_GAMES = frozenset({"csgo", "gta5", "gp"})


def _validate_gp_research_contract(value: dict[str, Any], time_limit: float) -> None:
    """Require the reproducibility fields shared by both GP protocols."""
    setup = value.get("setup")
    if not isinstance(setup, dict):
        raise ValueError("GP tasks require a 'setup' mapping")
    required_setup = {"game_version", "resolution", "initialization", "keymap"}
    missing_setup = sorted(required_setup - set(setup))
    if missing_setup:
        raise ValueError(f"GP task setup is missing reproducibility fields: {missing_setup}")
    resolution = setup.get("resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or any(int(value) < 1 for value in resolution)
    ):
        raise ValueError("GP task setup.resolution must be [width, height]")
    if time_limit <= 0:
        raise ValueError("GP tasks require a positive time_limit")

    termination = value.get("termination")
    if not isinstance(termination, dict) or not {"success", "failure", "timeout"}.issubset(
        termination
    ):
        raise ValueError("GP tasks require success/failure/timeout termination definitions")
    milestones = value.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("GP tasks require a non-empty milestones list")
    reporting = value.get("reporting")
    if not isinstance(reporting, dict) or not {"success_rate", "progress"}.issubset(reporting):
        raise ValueError("GP tasks require reporting.success_rate and reporting.progress")


@dataclass
class EvaluatorConfig:
    """Configuration for one state or VLM judge."""

    type: str = "state"
    conditions: dict[str, Any] | None = None
    rubric: str | None = None
    model: str | None = None
    sample_frames: int = 8
    min_confidence: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluatorConfig":
        if not isinstance(value, dict):
            raise ValueError("'evaluator' must be a mapping")
        evaluator_type = str(value.get("type", "state")).strip().lower()
        known = {
            "type",
            "conditions",
            "rubric",
            "model",
            "sample_frames",
            "min_confidence",
        }
        if evaluator_type == "state":
            conditions = value.get("conditions")
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError("State evaluator requires a non-empty 'conditions' mapping")
            if value.get("rubric") is not None:
                raise ValueError("State evaluator cannot also declare a VLM rubric")
            return cls(
                type="state",
                conditions=conditions,
                extra={key: item for key, item in value.items() if key not in known},
            )
        if evaluator_type == "vlm":
            rubric = value.get("rubric")
            if not isinstance(rubric, str) or not rubric.strip():
                raise ValueError("VLM evaluator requires a non-empty 'rubric' string")
            if value.get("conditions") is not None:
                raise ValueError("VLM evaluator cannot also declare state conditions")
            sample_frames = int(value.get("sample_frames", 8))
            min_confidence = float(value.get("min_confidence", 0.5))
            if sample_frames < 2:
                raise ValueError("VLM evaluator sample_frames must be at least 2")
            if not 0.0 <= min_confidence <= 1.0:
                raise ValueError("VLM evaluator min_confidence must be in [0, 1]")
            return cls(
                type="vlm",
                rubric=rubric.strip(),
                model=value.get("model"),
                sample_frames=sample_frames,
                min_confidence=min_confidence,
                extra={key: item for key, item in value.items() if key not in known},
            )
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.type == "state":
            result["conditions"] = self.conditions
        else:
            result.update(
                {
                    "rubric": self.rubric,
                    "sample_frames": self.sample_frames,
                    "min_confidence": self.min_confidence,
                }
            )
            if self.model:
                result["model"] = self.model
        result.update(self.extra)
        return result


@dataclass
class TaskConfig:
    """One project-supplied evaluation episode definition."""

    task_id: str
    game: str
    protocol: str
    description: str
    evaluator: EvaluatorConfig
    max_steps: int = 1000
    time_limit: float = 0.0
    setup: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any], source_path: str = "") -> "TaskConfig":
        if not isinstance(value, dict):
            raise ValueError("Task YAML root must be a mapping")
        task_id = str(value.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("Task requires a non-empty task_id")
        game = str(value.get("game", "")).strip().lower()
        if game not in SUPPORTED_GAMES:
            allowed = ", ".join(sorted(SUPPORTED_GAMES))
            raise ValueError(f"Unsupported game '{game}'; expected one of: {allowed}")
        protocol = resolve_protocol(game, value.get("protocol"))
        description = str(value.get("description", value.get("instruction", ""))).strip()
        if not description:
            raise ValueError("Task requires a non-empty description or instruction")
        if "evaluator" not in value:
            raise ValueError("Task must declare an 'evaluator' mapping")
        evaluator = EvaluatorConfig.from_dict(value["evaluator"])
        required_evaluator = expected_evaluator(protocol)
        if evaluator.type != required_evaluator:
            raise ValueError(
                f"Protocol '{protocol}' requires evaluator.type: {required_evaluator}"
            )

        max_steps = int(value.get("max_steps", 1000))
        time_limit = float(value.get("time_limit", 0.0))
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if time_limit < 0:
            raise ValueError("time_limit cannot be negative")
        if game == "gp":
            _validate_gp_research_contract(value, time_limit)

        known = {
            "task_id",
            "game",
            "protocol",
            "description",
            "instruction",
            "evaluator",
            "max_steps",
            "time_limit",
            "setup",
            "metadata",
        }
        return cls(
            task_id=task_id,
            game=game,
            protocol=protocol,
            description=description,
            evaluator=evaluator,
            max_steps=max_steps,
            time_limit=time_limit,
            setup=dict(value.get("setup", {}) or {}),
            metadata=dict(value.get("metadata", {}) or {}),
            extra={key: item for key, item in value.items() if key not in known},
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskConfig":
        task_path = Path(path)
        with open(task_path, encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        return cls.from_dict(value, source_path=str(task_path))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "game": self.game,
            "protocol": self.protocol,
            "description": self.description,
            "max_steps": self.max_steps,
            "time_limit": self.time_limit,
            "setup": self.setup,
            "evaluator": self.evaluator.to_dict(),
            "metadata": self.metadata,
        }
        result.update(self.extra)
        return result
