"""Unified logging configuration for GameEval."""

from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str = "gameeval",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
) -> logging.Logger:
    """Create or get a logger with console handler.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    fmt : str
        Log format string.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str = "gameeval") -> logging.Logger:
    """Get an existing logger (creates one with defaults if needed)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger
