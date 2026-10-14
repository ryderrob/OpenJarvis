"""Track running agent tasks and drop them on a "stop" command.

A stop (a message that is just "stop", "cancel", "abort", "/stop", ...):

* flags every running agent run (no further tool calls; the loop ends),
* cancels the waiting message handlers so their late replies are never sent,
* cancels in-flight Telegram operations (downloads, joins, uploads),
* denies pending approval requests, and
* stops any media playing in calls.

Calls themselves stay connected — hanging up is a separate request.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openjarvis.connectors.telegram_logging import preview

logger = logging.getLogger(__name__)

_STOP = re.compile(
    r"^\s*(?:/|jarvis[\s,!.:]*)?"
    r"(stop|cancel|abort|halt|enough|kill it|stop it|stop that|stop everything|"
    r"stop all|stop now|cancel (?:it|that|everything|all)|never ?mind|nevermind)"
    r"[\s.!]*(?:please|jarvis)?[\s.!]*$",
    re.IGNORECASE,
)
_MENTION = re.compile(r"@\w+")


def is_stop_command(text: str) -> bool:
    """True for a short message that only says stop/cancel (mentions ignored)."""
    return bool(_STOP.match(_MENTION.sub("", text or "")))


@dataclass
class _Run:
    run_id: int
    chat_id: int
    description: str
    event: threading.Event = field(default_factory=threading.Event)
    handler: Optional["asyncio.Task[Any]"] = None
    started: float = field(default_factory=time.monotonic)


class TaskRegistry:
    """Process-wide list of running agent tasks."""

    _instance: Optional["TaskRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._runs: dict[int, _Run] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "TaskRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self, chat_id: int, description: str) -> _Run:
        """Register a run; call from the handler task that awaits it."""
        try:
            handler: Optional[asyncio.Task[Any]] = asyncio.current_task()
        except RuntimeError:
            handler = None
        run = _Run(next(self._ids), chat_id, description, handler=handler)
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def finish(self, run: _Run) -> None:
        with self._lock:
            self._runs.pop(run.run_id, None)

    def running(self) -> list[_Run]:
        with self._lock:
            return list(self._runs.values())

    def cancel_runs(self, *, exclude: Optional[asyncio.Task[Any]] = None) -> list[str]:
        """Flag every run and cancel its waiting handler; returns descriptions."""
        cancelled = []
        for run in self.running():
            run.event.set()
            if run.handler is not None and run.handler is not exclude:
                run.handler.cancel()
            cancelled.append(run.description)
            self.finish(run)
        return cancelled


async def stop_everything(reason: str) -> dict[str, Any]:
    """Drop all running work (see module docstring). Runs on the Telegram loop."""
    from openjarvis.connectors.telegram_agent import ApprovalBroker
    from openjarvis.connectors.telegram_user import TelegramUserClient
    from openjarvis.connectors.telegram_voice import VoiceCallManager

    tasks = TaskRegistry.get().cancel_runs(exclude=asyncio.current_task())
    approvals = ApprovalBroker.get().deny_all()
    operations = TelegramUserClient.get().cancel_inflight()
    media = []
    manager = VoiceCallManager.get()
    for chat_id, session in list(manager.sessions.items()):
        if session.media is not None:
            title = session.media["title"]
            try:
                await manager.stop_media(chat_id, f"stopped ({reason})")
                media.append(title)
            except Exception:
                logger.exception("Could not stop media in %s", session.title)
    summary = {
        "tasks": tasks,
        "approvals_denied": approvals,
        "operations_cancelled": operations,
        "media_stopped": media,
    }
    logger.info("STOP (%s): %s", reason, summary)
    return summary


def describe_stop(summary: dict[str, Any]) -> str:
    parts = []
    if summary["tasks"]:
        names = "; ".join(preview(t, 60) for t in summary["tasks"])
        parts.append(f"{len(summary['tasks'])} task(s): {names}")
    if summary["media_stopped"]:
        parts.append("playback: " + ", ".join(summary["media_stopped"]))
    if summary["approvals_denied"]:
        parts.append(f"{summary['approvals_denied']} pending approval(s)")
    if summary["operations_cancelled"] and not summary["tasks"]:
        parts.append(f"{summary['operations_cancelled']} Telegram operation(s)")
    if not parts:
        return "Nothing was running."
    return "🛑 Stopped " + " · ".join(parts) + "."


__all__ = ["TaskRegistry", "describe_stop", "is_stop_command", "stop_everything"]
