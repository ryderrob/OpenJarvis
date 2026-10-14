"""Cooperative cancellation for agent runs.

Python cannot kill a running thread, so a "stop" works by flagging the run:
the thread running an agent binds a :class:`threading.Event`, and the tool
executor and agent loops check :func:`is_cancelled` before every tool call or
turn. Once set, no further tools execute and the loop ends at the next check.
"""

from __future__ import annotations

import threading
from typing import Optional

_local = threading.local()

CANCELLED_MESSAGE = "Cancelled by the user."


def bind(event: Optional[threading.Event]) -> None:
    """Associate *event* with the current thread (``None`` clears it)."""
    _local.event = event


def current() -> Optional[threading.Event]:
    return getattr(_local, "event", None)


def is_cancelled() -> bool:
    event = current()
    return event is not None and event.is_set()


__all__ = ["CANCELLED_MESSAGE", "bind", "current", "is_cancelled"]
