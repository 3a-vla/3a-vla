"""Minimal game-agnostic evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Write per-task JSON and Markdown summaries without a composite score."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: dict[str, dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        payload = {"metadata": metadata or {}, "results": results}
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)

        markdown_path = self.output_dir / "evaluation_summary.md"
        markdown_path.write_text(
            self._render_markdown(results, metadata or {}), encoding="utf-8"
        )
        return {"json": json_path, "markdown": markdown_path}

    @staticmethod
    def _render_markdown(
        results: dict[str, dict[str, Any]], metadata: dict[str, Any]
    ) -> str:
        lines = ["# GameEval Run Summary", ""]
        for key in ("game", "protocol", "agent", "version", "timestamp"):
            if key in metadata:
                lines.append(f"- {key}: `{metadata[key]}`")
        lines.extend(
            [
                "",
                "Each row is reported independently; GameEval does not average tasks into a leaderboard score.",
                "",
                "Success rate always uses all pre-declared episodes as the denominator; unknown is non-success.",
                "",
                "| Task | Success rate | Coverage | Unknown | Error | Completed / scheduled |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in results.items():
            lines.append(
                "| {name} | {success:.1f}% | {coverage:.1f}% | {unknown:.1f}% | {error:.1f}% | {completed} / {episodes} |".format(
                    name=name,
                    success=float(values.get("success_rate", 0.0)),
                    coverage=float(values.get("coverage_rate", 0.0)),
                    unknown=float(values.get("unknown_rate", 0.0)),
                    error=float(values.get("error_rate", 0.0)),
                    completed=int(values.get("completed_episodes", values.get("episodes", 0))),
                    episodes=int(values.get("episodes", 0)),
                )
            )
        lines.append("")
        return "\n".join(lines)
