"""Paths to resources shipped with GameEval."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"

__all__ = ["PACKAGE_ROOT", "CONFIG_ROOT"]
