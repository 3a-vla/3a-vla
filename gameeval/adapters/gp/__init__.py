"""Separate GP-State and GP-Visual runtimes."""

from gameeval.adapters.gp.adapter import GPBridgeAdapter, GPBridgeConfig
from gameeval.adapters.gp.visual import GPVisualAdapter

__all__ = ["GPBridgeAdapter", "GPBridgeConfig", "GPVisualAdapter"]
