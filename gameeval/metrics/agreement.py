"""Agreement statistics for state, VLM, and human episode labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LABELS = ("success", "fail", "unknown")
SOURCES = ("state", "vlm", "human")


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    aliases = {"pass": "success", "passed": "success", "failure": "fail", "failed": "fail"}
    label = aliases.get(label, label)
    return label if label in LABELS else None


class HumanLabelStore:
    """Read expert labels keyed by the exact task/episode identity."""

    def __init__(self, values: dict[tuple[str, int], str] | None = None) -> None:
        self.values = values or {}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "HumanLabelStore":
        values: dict[tuple[str, int], str] = {}
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                task_id = str(item.get("task_id", "")).strip()
                episode = int(item.get("episode"))
                label = normalize_label(item.get("label"))
                if not task_id or label is None:
                    raise ValueError(f"Invalid human label at line {line_number}")
                values[(task_id, episode)] = label
        return cls(values)

    def get(self, task_id: str, episode: int) -> str | None:
        return self.values.get((task_id, int(episode)))


def _pairwise(records: list[dict[str, str | None]], left: str, right: str) -> dict[str, Any]:
    pairs = [
        (record.get(left), record.get(right))
        for record in records
        if record.get(left) in LABELS and record.get(right) in LABELS
    ]
    if not pairs:
        return {"episodes": 0, "agreement_rate": None, "cohen_kappa": None}
    matches = sum(a == b for a, b in pairs)
    total = len(pairs)
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in LABELS
    )
    observed = matches / total
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "episodes": total,
        "agreements": matches,
        "agreement_rate": observed * 100.0,
        "cohen_kappa": kappa,
    }


def calculate_agreement(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report pairwise and exact three-way agreement without imputing labels."""
    normalized = [
        {source: normalize_label(record.get(source)) for source in SOURCES}
        for record in records
    ]
    complete = [record for record in normalized if all(record[source] in LABELS for source in SOURCES)]
    three_way_matches = sum(len({record[source] for source in SOURCES}) == 1 for record in complete)
    distributions = {
        source: dict(Counter(record[source] for record in normalized if record[source] in LABELS))
        for source in SOURCES
    }
    return {
        "episodes": len(normalized),
        "label_coverage": {
            source: sum(record[source] in LABELS for record in normalized) / len(normalized) * 100.0
            if normalized
            else 0.0
            for source in SOURCES
        },
        "label_distribution": distributions,
        "pairwise": {
            "state_vlm": _pairwise(normalized, "state", "vlm"),
            "state_human": _pairwise(normalized, "state", "human"),
            "vlm_human": _pairwise(normalized, "vlm", "human"),
        },
        "three_way": {
            "episodes": len(complete),
            "agreements": three_way_matches,
            "agreement_rate": three_way_matches / len(complete) * 100.0 if complete else None,
        },
    }


__all__ = ["HumanLabelStore", "calculate_agreement", "normalize_label"]
