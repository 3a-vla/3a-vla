"""Discovery of private/runtime and VLA-agent plugins."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

AGENT_PLUGIN_GROUP = "gameeval.agents"
ADAPTER_PLUGIN_GROUP = "gameeval.adapters"


def available_agent_plugins() -> list[str]:
    return sorted(item.name for item in entry_points(group=AGENT_PLUGIN_GROUP))


def load_agent_plugin(name: str, config: dict[str, Any] | None = None) -> Any:
    """Load a VLA agent from the ``gameeval.agents`` entry-point group.

    The entry point may expose a class, a no-argument factory, or a factory
    accepting one configuration dictionary. The result must provide
    ``act(observation)``; ``reset()`` and ``set_task_context(...)`` are optional.
    """
    matches = [
        item
        for item in entry_points(group=AGENT_PLUGIN_GROUP)
        if item.name == name
    ]
    if not matches:
        available = ", ".join(available_agent_plugins()) or "none"
        raise ValueError(
            f"No built-in or installed agent named '{name}' (plugins: {available})"
        )
    target = matches[0].load()
    if isinstance(target, type):
        try:
            agent = target(**(config or {}))
        except TypeError:
            agent = target()
    else:
        try:
            agent = target(config or {})
        except TypeError:
            agent = target()
    if not callable(getattr(agent, "act", None)):
        raise TypeError(f"Agent plugin '{name}' must provide act(observation)")
    return agent


def available_adapter_plugins() -> list[str]:
    return sorted(item.name for item in entry_points(group=ADAPTER_PLUGIN_GROUP))


def load_adapter_plugin(name: str, config: dict[str, Any] | None = None) -> Any:
    """Load a private game adapter without importing it into the OSS tree."""
    matches = [
        item
        for item in entry_points(group=ADAPTER_PLUGIN_GROUP)
        if item.name == name
    ]
    if not matches:
        available = ", ".join(available_adapter_plugins()) or "none"
        raise ValueError(f"Adapter plugin '{name}' is not installed (plugins: {available})")
    target = matches[0].load()
    if isinstance(target, type):
        adapter = target()
    else:
        try:
            adapter = target(config or {})
        except TypeError:
            adapter = target()
    required = ("connect", "close", "reset_runtime", "step_runtime", "screenshot")
    missing = [method for method in required if not callable(getattr(adapter, method, None))]
    if missing:
        raise TypeError(f"Adapter plugin '{name}' is missing: {', '.join(missing)}")
    return adapter
