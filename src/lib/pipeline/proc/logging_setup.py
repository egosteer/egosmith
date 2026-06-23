"""Structured logging setup for HaWoR.

The codebase historically logged via ``hawor.utils.logging.vprint``, which prints
to stdout only when ``HAWOR_QUIET`` is unset. Two problems followed:

* ``HAWOR_QUIET`` was captured at import time (``from ... import QUIET_MODE``), so
  toggling it at runtime did nothing.
* Batch mode force-set ``HAWOR_QUIET=1``, which silenced *everything* -- including
  warnings and errors that the user needed to see.

This module provides a small stdlib ``logging`` setup so genuine warnings/errors
(``logger.warning`` / ``logger.error``) are visible regardless of the quiet
setting, while routine progress chatter still respects it. Entry points call
:func:`configure_logging` once; library code calls :func:`get_logger`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

LOGGER_NAME = "hawor"
_TRUTHY = {"1", "true", "yes", "y", "on"}

_configured = False


def quiet_enabled() -> bool:
    """Live read of the quiet flag (never frozen at import)."""
    return os.environ.get("HAWOR_QUIET", "").strip().lower() in _TRUTHY


# --- Back-compat shim for the historical ``hawor.utils.logging`` API --------
# Call sites used to import ``vprint`` / ``QUIET_MODE`` from the HaWoR fork's
# logging helper. That helper is first-party (not part of upstream HaWoR), so it
# lives here now. Prefer :func:`get_logger` for new code.
is_quiet = quiet_enabled  # live read; alias for the historical name

# Module-level snapshot captured at import (historical semantics; does NOT reflect
# later runtime changes -- prefer ``is_quiet()`` / ``quiet_enabled()``).
QUIET_MODE = quiet_enabled()


def vprint(*args, **kwargs) -> None:
    """Print routine progress only when not in quiet mode (live read)."""
    if not quiet_enabled():
        print(*args, **kwargs)


def _level_from_env(default: str = "INFO") -> str:
    explicit = os.environ.get("HAWOR_LOG_LEVEL", "").strip()
    if explicit:
        return explicit.upper()
    # Quiet downgrades routine INFO to WARNING; it does NOT hide warnings/errors.
    if quiet_enabled():
        return "WARNING"
    return default


def configure_logging(level: Optional[str] = None, *, quiet: Optional[bool] = None, force: bool = False) -> logging.Logger:
    """Configure the ``hawor`` logger once with a single stderr handler.

    ``level`` overrides everything; otherwise ``$HAWOR_LOG_LEVEL`` then
    ``$HAWOR_QUIET`` decide. Idempotent unless ``force=True``.
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)

    if quiet:
        level = "WARNING"
    if level is None:
        level = _level_from_env()
    logger.setLevel(level)

    if force:
        for handler in [h for h in logger.handlers if getattr(h, "_hawor_handler", False)]:
            logger.removeHandler(handler)
        _configured = False

    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        handler._hawor_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the shared ``hawor`` logger, or a ``hawor.<name>`` child."""
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
