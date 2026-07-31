"""Task Registry — discovery and indexing of task configurations.

Scans a user-supplied directory tree for YAML files, parses each into a
:class:`TaskConfig`, and provides lookup by game and task ID.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from gameeval.tasks.task_config import TaskConfig

logger = logging.getLogger("gameeval.tasks.registry")


class TaskRegistry:
    """Registry that auto-discovers and indexes task YAML files.

    Parameters
    ----------
    tasks_dir : str | Path
        Root directory containing task YAML files.
        Any directory layout is accepted; ``game`` comes from each YAML file.
    """

    def __init__(self, tasks_dir: str | Path | None = None):
        self._tasks: dict[tuple[str, str, str], TaskConfig] = {}
        self._tasks_dir: Path | None = None

        if tasks_dir is not None:
            self.scan(tasks_dir)

    def scan(self, tasks_dir: str | Path) -> int:
        """Scan a directory for task YAML files and register them.

        Parameters
        ----------
        tasks_dir : str | Path
            Root task directory.

        Returns
        -------
        int
            Number of tasks registered.
        """
        self._tasks_dir = Path(tasks_dir)
        count = 0

        for yaml_path in sorted(self._tasks_dir.rglob("*.yaml")):
            try:
                config = TaskConfig.from_yaml(yaml_path)
                key = (config.game, config.protocol, config.task_id)
                if key in self._tasks:
                    logger.warning(
                        "Duplicate task_id '%s' — overwriting from %s",
                        config.task_id,
                        yaml_path,
                    )

                self._tasks[key] = config
                count += 1
            except Exception as e:
                logger.warning("Failed to parse task %s: %s", yaml_path, e)

        logger.info("Registered %d tasks from %s", count, self._tasks_dir)
        return count

    def register(self, config: TaskConfig) -> None:
        """Manually register a task configuration."""
        self._tasks[(config.game, config.protocol, config.task_id)] = config

    # ---- Lookup API ----------------------------------------------------------

    def get_task(
        self,
        task_id: str,
        game: str | None = None,
        protocol: str | None = None,
    ) -> TaskConfig:
        """Get a task by its ID.

        Raises
        ------
        KeyError
            If the task ID is not found.
        """
        matches = [
            task
            for (item_game, item_protocol, item_id), task in self._tasks.items()
            if item_id == task_id
            and (game is None or item_game == game)
            and (protocol is None or item_protocol == protocol)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"Task '{task_id}' is ambiguous; pass both game= and protocol="
            )
        available = sorted(
            f"{item_game}:{item_protocol}:{item_id}"
            for item_game, item_protocol, item_id in self._tasks
        )
        raise KeyError(f"Task '{task_id}' not found. Available: {available}")

    def list_tasks(
        self,
        game: str | None = None,
        protocol: str | None = None,
    ) -> list[TaskConfig]:
        """List tasks, optionally filtered by game.

        Parameters
        ----------
        game : str | None
            Filter by game (``csgo``, ``gta5``, or ``gp``).
        """
        results = []
        for task in self._tasks.values():
            if game and task.game != game:
                continue
            if protocol and task.protocol != protocol:
                continue
            results.append(task)
        return sorted(results, key=lambda t: t.task_id)

    def list_task_ids(self, **filters: str) -> list[str]:
        """Return sorted list of task IDs, optionally filtered."""
        return [t.task_id for t in self.list_tasks(**filters)]

    def list_games(self) -> list[str]:
        """Return sorted list of unique game names."""
        return sorted({t.game for t in self._tasks.values()})

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return any(item_id == task_id for _game, _protocol, item_id in self._tasks)

    def __iter__(self) -> Iterator[TaskConfig]:
        return iter(sorted(self._tasks.values(), key=lambda t: t.task_id))
