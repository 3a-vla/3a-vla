"""Input controller interface shared by desktop game runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gameeval.core.action_space import Action


class InputController(ABC):
    @abstractmethod
    def apply(self, action: Action) -> dict[str, Any]:
        """Execute an action or action chunk."""

    @abstractmethod
    def release_all(self) -> None:
        """Release every held key and mouse button."""

    def close(self) -> None:
        self.release_all()
