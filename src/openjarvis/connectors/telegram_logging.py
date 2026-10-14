"""File logging for everything the Telegram user-account integration does.

``<config dir>/logs/telegram.log`` (rotating, 5 x 5 MB, mode 0600) records
incoming messages, agent runs and the tools they used, approvals, and calls.
Credentials are stripped by the CLI's :class:`SanitizingFormatter`. The very
chatty native call library (ntgcalls relay pings) is limited to errors.
"""

from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from openjarvis.core.paths import get_config_dir

_LOCK = threading.Lock()
_HANDLER: RotatingFileHandler | None = None
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
# Third-party loggers and the level worth keeping from each.
_LIBRARY_LEVELS = {
    "telethon": logging.WARNING,
    "pytgcalls": logging.WARNING,
    "ntgcalls": logging.ERROR,  # logs every relay ping as WARNING
    "httpx": logging.WARNING,
}


def log_path() -> Path:
    return get_config_dir() / "logs" / "telegram.log"


def ensure_telegram_logging(*, verbose: bool = False) -> Path:
    """Attach the rotating Telegram log file once per process."""
    global _HANDLER
    with _LOCK:
        path = log_path()
        if _HANDLER is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                str(path), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            from openjarvis.cli.log_config import SanitizingFormatter

            handler.setFormatter(SanitizingFormatter(_FORMAT))
            logging.getLogger("openjarvis").addHandler(handler)
            for name, level in _LIBRARY_LEVELS.items():
                library = logging.getLogger(name)
                library.setLevel(level)
                library.addHandler(handler)
            _HANDLER = handler
        level = logging.DEBUG if verbose else logging.INFO
        _HANDLER.setLevel(level)
        root = logging.getLogger("openjarvis")
        # Let INFO reach the file; console handlers keep their own levels.
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        return path


def preview(text: object, limit: int = 300) -> str:
    """Single-line, length-limited rendering for log messages."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


__all__ = ["ensure_telegram_logging", "log_path", "preview"]
