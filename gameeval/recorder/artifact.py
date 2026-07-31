"""Standard, replayable episode artifact writer."""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml
from PIL import Image

from gameeval.core.evaluation import EvaluationResult

logger = logging.getLogger("gameeval.recorder.artifact")


class StateRetention(str, Enum):
    """How much privileged state is persisted to disk."""

    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


class EpisodeArtifactWriter:
    """Write one self-contained directory per evaluation episode.

    State is excluded by default, which is suitable for public evaluation
    artifacts. When OpenCV is unavailable the
    writer falls back to JPEG frames instead of silently dropping visuals.
    """

    _DEFAULT_SUMMARY_STATE_FIELDS = {
        "metrics",
        "terminal",
        "success",
    }
    _DEFAULT_PUBLIC_INFO_FIELDS = {
        "done",
        "timeout",
        "step",
        "max_steps",
        "metrics",
        "input",
        "episode",
        "input_injection",
    }

    def __init__(
        self,
        output_dir: str | Path,
        *,
        state_retention: StateRetention | str = StateRetention.NONE,
        video_fps: float = 8.0,
        summary_state_fields: set[str] | None = None,
        public_info_fields: set[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_retention = StateRetention(state_retention)
        self.video_fps = float(video_fps)
        self.summary_state_fields = set(
            summary_state_fields or self._DEFAULT_SUMMARY_STATE_FIELDS
        )
        self.public_info_fields = set(
            public_info_fields or self._DEFAULT_PUBLIC_INFO_FIELDS
        )
        self._episode_dir: Path | None = None
        self._actions: TextIO | None = None
        self._observations: TextIO | None = None
        self._states: TextIO | None = None
        self._video_writer: Any = None
        self._video_path: Path | None = None
        self._fallback_frames_dir: Path | None = None
        self._frame_count = 0
        self._start_time = 0.0
        self._task_id = ""
        self._game = ""
        self._episode_id = 0
        self._judge_backend = "unknown"
        self._protocol = "unknown"
        self._sealed = False

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    @property
    def video_path(self) -> Path | None:
        return self._video_path if self._video_path and self._video_path.exists() else None

    @property
    def actions_path(self) -> Path | None:
        return self._episode_dir / "actions.jsonl" if self._episode_dir else None

    def start_episode(
        self,
        *,
        task: dict[str, Any],
        game: str,
        episode_id: int,
    ) -> Path:
        if self._episode_dir is not None:
            raise RuntimeError("Previous episode artifact was not finalized")
        self._task_id = str(task.get("task_id", "task"))
        self._game = game
        self._episode_id = int(episode_id)
        self._judge_backend = str((task.get("evaluator", {}) or {}).get("type", "unknown"))
        self._protocol = str(task.get("protocol", "unknown"))
        self._start_time = time.time()
        self._frame_count = 0
        self._sealed = False
        self._episode_dir = (
            self.output_dir / self._task_id / f"episode_{self._episode_id:04d}"
        )
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        with open(self._episode_dir / "task.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(task, handle, sort_keys=False, allow_unicode=True)
        self._actions = open(self._episode_dir / "actions.jsonl", "w", encoding="utf-8")
        self._observations = open(
            self._episode_dir / "observations.jsonl", "w", encoding="utf-8"
        )
        if self.state_retention is not StateRetention.NONE:
            self._states = open(self._episode_dir / "state.jsonl", "w", encoding="utf-8")
        return self._episode_dir

    def record_step(
        self,
        *,
        step_index: int,
        action: dict[str, Any],
        frame: np.ndarray | None,
        state: dict[str, Any] | None,
        info: dict[str, Any] | None = None,
        done: bool = False,
    ) -> None:
        if self._episode_dir is None or self._actions is None or self._observations is None:
            raise RuntimeError("start_episode must be called before record_step")
        if self._sealed:
            raise RuntimeError("Cannot record after seal_rollout")
        timestamp = time.time()
        public_info = {
            key: value
            for key, value in (info or {}).items()
            if key in self.public_info_fields
        }
        self._write_jsonl(
            self._actions,
            {
                "step_index": step_index,
                "timestamp": timestamp,
                "action": action,
                "done": bool(done),
                "info": public_info,
            },
        )

        frame_record: dict[str, Any] = {
            "step_index": step_index,
            "timestamp": timestamp,
            "frame_index": None,
        }
        if frame is not None:
            self._frame_count += 1
            frame_record["frame_index"] = self._frame_count - 1
            frame_record["shape"] = list(frame.shape)
            self._write_frame(frame, step_index)
        self._write_jsonl(self._observations, frame_record)

        if self._states is not None and state is not None:
            state_value = state
            if self.state_retention is StateRetention.SUMMARY:
                state_value = {
                    key: value
                    for key, value in state.items()
                    if key in self.summary_state_fields
                }
            self._write_jsonl(
                self._states,
                {"step_index": step_index, "timestamp": timestamp, "state": state_value},
            )

    def seal_rollout(self) -> Path | None:
        """Flush actions and close the video before a post-hoc judge reads it."""
        if self._episode_dir is None:
            raise RuntimeError("No episode artifact is active")
        if not self._sealed:
            for handle in (self._actions, self._observations, self._states):
                if handle is not None:
                    handle.flush()
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self._sealed = True
        return self.video_path

    def write_sidecar(self, filename: str, payload: dict[str, Any]) -> Path:
        """Write a non-authoritative calibration artifact for the active episode."""
        if self._episode_dir is None:
            raise RuntimeError("No episode artifact is active")
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("Sidecar filename must be a plain .json name")
        path = self._episode_dir / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        return path

    def finalize(
        self,
        *,
        result: EvaluationResult,
        metrics: dict[str, Any],
        total_steps: int,
    ) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("No episode artifact is active")
        episode_dir = self._episode_dir
        self.seal_rollout()
        self._close_streams()
        with open(episode_dir / "evaluation.json", "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False, default=str)
        manifest = {
            "schema_version": "1.0",
            "framework": "GameEval",
            "task_id": self._task_id,
            "game": self._game,
            "protocol": self._protocol,
            "episode_id": self._episode_id,
            "start_time": self._start_time,
            "end_time": time.time(),
            "total_steps": int(total_steps),
            "frame_count": self._frame_count,
            "video": "rollout.mp4" if (episode_dir / "rollout.mp4").exists() else None,
            "frames": "frames" if (episode_dir / "frames").exists() else None,
            "state_retention": self.state_retention.value,
            "judge_backend": self._judge_backend,
            "state_source": "privileged" if self._protocol.endswith("-state") else "none",
            "evaluation": "evaluation.json",
            "agreement": "agreement.json" if (episode_dir / "agreement.json").exists() else None,
            "metrics": metrics,
        }
        with open(episode_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
        self._reset_current()
        return episode_dir

    def close(self) -> None:
        self._close_streams()
        self._reset_current()

    def _write_frame(self, frame: np.ndarray, step_index: int) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if self._video_writer is None and self._fallback_frames_dir is None:
            try:
                import cv2

                assert self._episode_dir is not None
                height, width = frame.shape[:2]
                self._video_path = self._episode_dir / "rollout.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self._video_path), fourcc, self.video_fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV VideoWriter could not open rollout.mp4")
                self._video_writer = writer
            except (ImportError, RuntimeError) as exc:
                assert self._episode_dir is not None
                logger.warning("MP4 writer unavailable; storing JPEG frames: %s", exc)
                self._video_path = None
                self._fallback_frames_dir = self._episode_dir / "frames"
                self._fallback_frames_dir.mkdir(exist_ok=True)

        if self._video_writer is not None:
            import cv2

            self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif self._fallback_frames_dir is not None:
            Image.fromarray(frame).save(
                self._fallback_frames_dir / f"frame_{step_index:06d}.jpg",
                quality=90,
            )

    @staticmethod
    def _write_jsonl(handle: TextIO, value: dict[str, Any]) -> None:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()

    def _close_streams(self) -> None:
        for handle in (self._actions, self._observations, self._states):
            if handle is not None:
                handle.close()
        self._actions = None
        self._observations = None
        self._states = None
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _reset_current(self) -> None:
        self._episode_dir = None
        self._video_path = None
        self._fallback_frames_dir = None
        self._frame_count = 0
        self._sealed = False
