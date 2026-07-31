"""Game-agnostic episode and run metric aggregation."""

from __future__ import annotations

from collections import Counter
from typing import Any


class MetricsCalculator:
    """Aggregate outcomes without defining a fixed composite score."""

    _RESERVED = {
        "episode",
        "success",
        "evaluation_status",
        "total_steps",
        "duration",
        "task_id",
    }

    def compute_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        status = str(
            episode.get(
                "evaluation_status",
                "success" if episode.get("success", False) else "fail",
            )
        ).lower()
        result: dict[str, Any] = {
            "status": status,
            "success": 1.0 if status == "success" else 0.0,
            "decidable": 1.0 if status in {"success", "fail"} else 0.0,
            "total_steps": float(episode.get("total_steps", 0)),
            "duration_seconds": float(episode.get("duration", 0.0)),
        }
        for key, value in episode.items():
            if key in self._RESERVED or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                result[key] = float(value)
        return result

    def aggregate(
        self,
        episodes: list[dict[str, Any]],
        *,
        scheduled_episodes: int | None = None,
    ) -> dict[str, Any]:
        scheduled = len(episodes) if scheduled_episodes is None else int(scheduled_episodes)
        if scheduled < len(episodes):
            raise ValueError("scheduled_episodes cannot be smaller than completed episodes")
        if scheduled < 0:
            raise ValueError("scheduled_episodes cannot be negative")
        if not episodes:
            return {
                "episodes": scheduled,
                "completed_episodes": 0,
                "not_run": scheduled,
                "successes": 0,
                "sr_failures": scheduled,
                "failures": 0,
                "unknown": 0,
                "errors": 0,
                "success_rate": 0.0,
                "coverage_rate": 0.0,
                "unknown_rate": 0.0,
                "error_rate": 0.0,
            }

        computed = [self.compute_episode(episode) for episode in episodes]
        statuses = Counter(str(item["status"]) for item in computed)
        completed = len(computed)
        total = scheduled
        decidable = statuses["success"] + statuses["fail"]
        result: dict[str, Any] = {
            "episodes": total,
            "completed_episodes": completed,
            "not_run": total - completed,
            "successes": statuses["success"],
            "non_successes": total - statuses["success"],
            "sr_failures": total - statuses["success"],
            "failures": statuses["fail"],
            "unknown": statuses["unknown"],
            "errors": statuses["error"],
            "success_rate": statuses["success"] / total * 100.0 if total else 0.0,
            "coverage_rate": decidable / total * 100.0 if total else 0.0,
            "unknown_rate": statuses["unknown"] / total * 100.0 if total else 0.0,
            "error_rate": statuses["error"] / total * 100.0 if total else 0.0,
            "not_run_rate": (total - completed) / total * 100.0 if total else 0.0,
            "avg_completion_steps": sum(item["total_steps"] for item in computed) / completed,
            "avg_duration_seconds": sum(item["duration_seconds"] for item in computed) / completed,
        }

        numeric_keys = {
            key
            for item in computed
            for key, value in item.items()
            if isinstance(value, (int, float))
            and key not in {"success", "decidable", "total_steps", "duration_seconds"}
        }
        for key in sorted(numeric_keys):
            values = [float(item[key]) for item in computed if key in item]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
        return result
