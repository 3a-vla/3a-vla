"""Vision-language-model evaluator for games without a state interface."""

from __future__ import annotations

import json
import math
from typing import Any

from gameeval.core.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorType,
    Evidence,
)
from gameeval.evaluators.base import BaseEvaluator, EpisodeContext
from gameeval.utils.vlm_client import VLMClient

DEFAULT_SYSTEM_PROMPT = """You are a strict evaluator of video-game agent rollouts.
Judge only from the ordered screenshots and the supplied task rubric. Do not assume
unseen game state. Return exactly one JSON object with exactly these fields:
{"status":"success|fail|unknown","confidence":0.0,"score":0.0,"reason":"brief visual evidence"}
confidence and score must be numbers in [0, 1]. Return no markdown or extra text."""


class VLMOutputError(ValueError):
    """Raised when a VLM response does not satisfy the judge protocol."""


class VLMJudgeEvaluator(BaseEvaluator):
    """Evaluate an ordered rollout with an OpenAI-compatible vision model.

    The client is injectable so local OpenAI-compatible endpoints and test doubles
    use the same path. No privileged episode state is included in the request.
    """

    def __init__(
        self,
        client: VLMClient | None = None,
        *,
        max_frames: int = 8,
        min_confidence: float = 0.5,
        confidence_threshold: float | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2 to retain first and last frames")
        threshold = min_confidence if confidence_threshold is None else confidence_threshold
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

        self.client = client if client is not None else VLMClient()
        self.max_frames = int(max_frames)
        self.min_confidence = float(threshold)
        self.system_prompt = system_prompt

    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.VLM

    def evaluate(self, episode: EpisodeContext) -> EvaluationResult:
        """Evaluate a completed rollout synchronously."""
        prepared = self._prepare_request(episode)
        if isinstance(prepared, EvaluationResult):
            return prepared
        prompt, images_b64, indices, threshold = prepared

        try:
            response = self.client.chat_sync(
                system_prompt=self.system_prompt,
                user_text=prompt,
                images_b64=images_b64,
                temperature=0.0,
            )
        except Exception as exc:
            return self._request_error(exc, indices)
        return self._parse_result(response, indices, len(episode.frames), threshold)

    async def evaluate_async(self, episode: EpisodeContext) -> EvaluationResult:
        """Evaluate a completed rollout without blocking an existing event loop."""
        prepared = self._prepare_request(episode)
        if isinstance(prepared, EvaluationResult):
            return prepared
        prompt, images_b64, indices, threshold = prepared

        try:
            response = await self.client.chat(
                system_prompt=self.system_prompt,
                user_text=prompt,
                images_b64=images_b64,
                temperature=0.0,
            )
        except Exception as exc:
            return self._request_error(exc, indices)
        return self._parse_result(response, indices, len(episode.frames), threshold)

    def _prepare_request(
        self, episode: EpisodeContext
    ) -> tuple[str, list[str], list[int], float] | EvaluationResult:
        frames = episode.frames
        if not frames:
            return EvaluationResult(
                status=EvaluationStatus.UNKNOWN,
                reason="No rollout frames are available for VLM evaluation",
                confidence=0.0,
                evaluator=self.evaluator_type,
                details={"selected_frame_indices": []},
            )

        try:
            threshold = self._confidence_threshold(episode.task)
            indices = self.sample_frame_indices(len(frames), self.max_frames)
            images_b64 = [VLMClient.encode_image(frames[index], fmt="jpeg") for index in indices]
        except Exception as exc:
            return EvaluationResult(
                status=EvaluationStatus.ERROR,
                reason=f"Could not prepare VLM request: {exc}",
                confidence=0.0,
                evaluator=self.evaluator_type,
                details={"error": str(exc)},
            )

        return self._build_prompt(episode.task, indices, len(frames)), images_b64, indices, threshold

    def _confidence_threshold(self, task: dict[str, Any]) -> float:
        evaluator_cfg = task.get("evaluator", {}) or {}
        configured = evaluator_cfg.get(
            "confidence_threshold",
            evaluator_cfg.get("min_confidence", self.min_confidence),
        )
        threshold = float(configured)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("VLM confidence threshold must be in [0, 1]")
        return threshold

    @staticmethod
    def sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
        """Return ordered, evenly spaced indices, always retaining both ends."""
        if frame_count <= 0:
            return []
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        if frame_count <= max_frames:
            return list(range(frame_count))
        return [
            int(position * (frame_count - 1) / (max_frames - 1))
            for position in range(max_frames)
        ]

    @staticmethod
    def _build_prompt(task: dict[str, Any], indices: list[int], total_frames: int) -> str:
        evaluator_cfg = task.get("evaluator", {}) or {}
        rubric = evaluator_cfg.get("rubric")
        if rubric is None:
            rubric = {
                "description": task.get("description", "Complete the specified game task"),
                "success_criteria": task.get("success_criteria", []),
            }

        task_context = {
            "task_id": task.get("task_id", "unknown"),
            "game": task.get("game", "unknown"),
            "description": task.get("description", ""),
            "rubric": rubric,
        }
        return (
            "Evaluate this rollout against the task and rubric below. The images are in "
            "chronological order. Frame indices are zero-based.\n\n"
            f"Task:\n{json.dumps(task_context, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Sampled frame indices: {indices} (total rollout frames: {total_frames}).\n"
            "Use only visible evidence. If the evidence is insufficient, return status unknown.\n"
            "Return strict JSON with status, confidence, score, and reason."
        )

    def _parse_result(
        self,
        response: Any,
        indices: list[int],
        total_frames: int,
        threshold: float,
    ) -> EvaluationResult:
        try:
            parsed = self._parse_response(response)
        except VLMOutputError as exc:
            return EvaluationResult(
                status=EvaluationStatus.ERROR,
                reason=f"Malformed VLM judge output: {exc}",
                confidence=0.0,
                evaluator=self.evaluator_type,
                details={
                    "raw_response": response if isinstance(response, str) else repr(response),
                    "error": str(exc),
                    "selected_frame_indices": indices,
                },
            )

        model_status, confidence, score, reason = parsed
        status = model_status
        details: dict[str, Any] = {
            "raw_response": response,
            "model_status": model_status.value,
            "confidence_threshold": threshold,
            "selected_frame_indices": indices,
        }
        if confidence < threshold:
            status = EvaluationStatus.UNKNOWN
            details["abstained"] = True
            reason = (
                f"VLM confidence {confidence:.3f} is below threshold {threshold:.3f}: {reason}"
            )

        return EvaluationResult(
            status=status,
            reason=reason,
            confidence=confidence,
            evaluator=self.evaluator_type,
            score=score,
            evidence=[
                Evidence(
                    kind="rollout_frames",
                    value={"indices": indices, "total_frames": total_frames},
                    description="Ordered rollout frames supplied to the VLM judge",
                )
            ],
            details=details,
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Unwrap a ```/```json fenced block around the verdict.

        Many chat models emit JSON inside a markdown fence even when asked not
        to. Unwrapping is purely syntactic and does not relax the field checks
        below, so a malformed verdict is still rejected.
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 2:
            return stripped
        # Drop the opening fence (with any language tag) and a closing fence.
        body = lines[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        return "\n".join(body).strip()

    @staticmethod
    def _parse_response(
        response: Any,
    ) -> tuple[EvaluationStatus, float, float, str]:
        if not isinstance(response, str) or not response.strip():
            raise VLMOutputError("response must be a non-empty JSON string")
        payload = VLMJudgeEvaluator._strip_code_fence(response)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise VLMOutputError("response is not a standalone JSON object") from exc
        if not isinstance(data, dict):
            raise VLMOutputError("top-level JSON value must be an object")

        required_fields = {"status", "confidence", "score", "reason"}
        if set(data) != required_fields:
            missing = sorted(required_fields - set(data))
            extra = sorted(set(data) - required_fields)
            raise VLMOutputError(f"fields must match protocol (missing={missing}, extra={extra})")

        raw_status = data["status"]
        allowed_statuses = {
            EvaluationStatus.SUCCESS.value: EvaluationStatus.SUCCESS,
            EvaluationStatus.FAIL.value: EvaluationStatus.FAIL,
            EvaluationStatus.UNKNOWN.value: EvaluationStatus.UNKNOWN,
        }
        if not isinstance(raw_status, str) or raw_status not in allowed_statuses:
            raise VLMOutputError("status must be one of: success, fail, unknown")

        confidence = VLMJudgeEvaluator._bounded_number(data["confidence"], "confidence")
        score = VLMJudgeEvaluator._bounded_number(data["score"], "score")
        reason = data["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise VLMOutputError("reason must be a non-empty string")
        return allowed_statuses[raw_status], confidence, score, reason.strip()

    @staticmethod
    def _bounded_number(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VLMOutputError(f"{field_name} must be a number in [0, 1]")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise VLMOutputError(f"{field_name} must be a number in [0, 1]")
        return number

    def _request_error(self, exc: Exception, indices: list[int]) -> EvaluationResult:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            reason=f"VLM request failed: {exc}",
            confidence=0.0,
            evaluator=self.evaluator_type,
            details={"error": str(exc), "selected_frame_indices": indices},
        )


__all__ = ["VLMJudgeEvaluator", "VLMOutputError"]
