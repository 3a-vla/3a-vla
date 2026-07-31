"""Manual agent for evaluation-pipeline debugging.

This agent does not inject any actions. It simply paces the evaluation loop so a
human can control the game directly while GameEval keeps collecting
screenshots, GSI state, and judge outputs.
"""

from __future__ import annotations

import time
from typing import Any

from gameeval.core.action_space import Action


class ManualAgent:
    """Human-in-the-loop agent for validating the evaluation pipeline.

    Parameters
    ----------
    step_interval : float
        Seconds to wait between samples.
    start_delay : float
        Extra delay before the first sampled step of each episode so the user
        can focus the CSGO window.
    verbose : bool
        Whether to print operator instructions at episode start.
    """

    def __init__(
        self,
        step_interval: float = 0.1,
        start_delay: float = 3.0,
        verbose: bool = True,
    ):
        self.step_interval = max(0.01, float(step_interval))
        self.start_delay = max(0.0, float(start_delay))
        self.verbose = bool(verbose)
        self._step_count = 0
        self._task_id = ""
        self._instruction = ""
        self._start_delay_pending = True

    def set_task_context(self, game: str, task: str, instruction: str | None = None) -> None:
        self._task_id = task
        self._instruction = instruction or ""

    def reset(self) -> None:
        self._step_count = 0
        self._start_delay_pending = True
        if self.verbose:
            print(
                f"[GameEval manual] task={self._task_id} | "
                f"sample_every={self.step_interval:.2f}s | start_delay={self.start_delay:.1f}s"
            )
            if self._instruction:
                print(f"[GameEval manual] objective: {self._instruction}")

    def act(self, obs: Any) -> Action:
        self._step_count += 1

        if self._start_delay_pending:
            self._start_delay_pending = False
            if self.start_delay > 0:
                time.sleep(self.start_delay)
        else:
            time.sleep(self.step_interval)

        return Action.noop()

    @property
    def name(self) -> str:
        return "ManualAgent"
