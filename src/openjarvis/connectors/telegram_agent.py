"""Shared agent runner and Telegram approval broker.

Both the private-message responder and live voice calls run the configured
agent (``[agent] default_agent`` + ``tools``) through :class:`AgentRunner`.
Tools that need approval (``requires_confirmation`` or ``[security]
confirm_tools``) are approved over Telegram by :class:`ApprovalBroker`: it
messages the user what the agent wants to run and waits for a yes/no reply in
the same private chat. No reply within the timeout counts as "no".
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Callable, Iterable, Optional

from openjarvis.connectors.telegram_logging import preview
from openjarvis.connectors.telegram_user import TelegramUserClient

logger = logging.getLogger(__name__)

APPROVAL_TIMEOUT_S = 300.0
_YES = re.compile(r"^\s*(y|yes|ok|okay|approve|approved|allow|go|do it|sure)\b", re.I)
_NO = re.compile(r"^\s*(n|no|nope|deny|denied|stop|cancel|don'?t)\b", re.I)
_PROMPT = re.compile(
    r"Allow execution of tool '(?P<tool>[^']+)' with args (?P<args>.*)\?$", re.S
)
_MAX_ARGS_CHARS = 1500


def format_approval_request(prompt: str) -> str:
    """Turn the executor's confirmation prompt into a readable message."""
    match = _PROMPT.match(prompt)
    if match is None:
        body = prompt
    else:
        args = match.group("args")
        if len(args) > _MAX_ARGS_CHARS:
            args = args[:_MAX_ARGS_CHARS] + " …"
        body = f"Tool: {match.group('tool')}\nArguments: {args}"
    return (
        "🔐 Jarvis needs your approval:\n\n"
        f"{body}\n\n"
        f"Reply yes to allow or no to deny ({int(APPROVAL_TIMEOUT_S // 60)} min)."
    )


class ApprovalBroker:
    """Ask a Telegram user to approve tool calls and wait for their answer."""

    _instance: Optional["ApprovalBroker"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._pending: dict[int, asyncio.Future[bool]] = {}
        self._installed = False

    @classmethod
    def get(cls) -> "ApprovalBroker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def install(self, client: Any) -> None:
        """Register the reply handler (before any other message handler)."""
        if self._installed:
            return
        from telethon import events

        client.add_event_handler(
            self._on_message,
            events.NewMessage(incoming=True, func=lambda e: e.is_private),
        )
        self._installed = True

    def has_pending(self, chat_id: int) -> bool:
        return chat_id in self._pending

    def deny_all(self) -> int:
        """Answer every open approval with "no" (call on the Telegram loop)."""
        denied = 0
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(False)
                denied += 1
        return denied

    async def _on_message(self, event: Any) -> None:
        from telethon import events

        future = self._pending.get(event.chat_id)
        if future is None or future.done():
            return
        text = event.raw_text or ""
        if _YES.match(text):
            future.set_result(True)
        elif _NO.match(text):
            future.set_result(False)
        else:
            await event.reply("Please answer yes or no.")
        raise events.StopPropagation

    async def _ask(self, chat_id: int, prompt: str, timeout: float) -> bool:
        client = await TelegramUserClient.get().client()
        self.install(client)
        previous = self._pending.get(chat_id)
        if previous is not None and not previous.done():
            previous.set_result(False)  # one outstanding approval per chat
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[chat_id] = future
        logger.info("Approval requested in chat %s: %s", chat_id, preview(prompt))
        try:
            await client.send_message(chat_id, format_approval_request(prompt))
            approved = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            approved = False
            await client.send_message(chat_id, "⌛ No answer — denied.")
        finally:
            if self._pending.get(chat_id) is future:
                del self._pending[chat_id]
        logger.info(
            "Approval for chat %s: %s", chat_id, "granted" if approved else "denied"
        )
        return approved

    def callback_for(
        self, chat_id: int, *, timeout: float = APPROVAL_TIMEOUT_S
    ) -> Callable[[str], bool]:
        """A blocking ``confirm_callback`` that asks *chat_id* over Telegram.

        Must be called from a worker thread (agents run in executors), never
        from the Telegram event loop itself.
        """

        def _confirm(prompt: str) -> bool:
            try:
                return (
                    TelegramUserClient.get()
                    .submit(self._ask(chat_id, prompt, timeout))
                    .result(timeout + 30)
                )
            except Exception:
                logger.exception("Telegram approval request failed")
                return False

        return _confirm


class AgentRunner:
    """One lazily-created SDK instance shared by messages and calls."""

    _instance: Optional["AgentRunner"] = None
    _instance_lock = threading.Lock()

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._jarvis: Any = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls, config: Any = None) -> "AgentRunner":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    def _sdk(self) -> Any:
        if self._jarvis is None:
            from openjarvis.sdk import Jarvis

            self._jarvis = Jarvis(config=self._config) if self._config else Jarvis()
        return self._jarvis

    def tools(self, exclude: Iterable[str] = ()) -> list[str]:
        with self._lock:
            config = self._sdk()._config
        skip = set(exclude)
        names = [t.strip() for t in config.agent.tools.split(",") if t.strip()]
        return [
            t
            for t in names
            if t not in skip
            and not any(
                t.startswith(prefix.rstrip("*"))
                for prefix in skip
                if prefix.endswith("*")
            )
        ]

    def run(
        self,
        prompt: str,
        *,
        confirm: Optional[Callable[[str], bool]] = None,
        exclude_tools: Iterable[str] = (),
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run the configured agent on *prompt* (blocking).

        Runs may overlap (only SDK creation is locked) so that a stopped run
        that is still winding down never blocks the next request. Setting
        *cancel_event* stops further tool calls and ends the agent loop.
        """
        from openjarvis.core import cancellation

        tools = self.tools(exclude_tools)
        with self._lock:
            jarvis = self._sdk()
        agent = jarvis._config.agent.default_agent or "orchestrator"
        started = time.monotonic()
        logger.info(
            "Agent %s run started (%d tools, approvals %s): %s",
            agent,
            len(tools),
            "via Telegram" if confirm else "off",
            preview(prompt[-400:]),
        )
        cancellation.bind(cancel_event)
        try:
            result = jarvis.ask_full(
                prompt, agent=agent, tools=tools, confirm_callback=confirm
            )
        except Exception:
            if cancellation.is_cancelled():
                logger.info("Agent %s run stopped by the user", agent)
                return cancellation.CANCELLED_MESSAGE
            logger.exception("Agent %s run failed", agent)
            raise
        finally:
            cancellation.bind(None)
        for used in result.get("tool_results", []):
            logger.info(
                "  tool %s -> %s: %s",
                used.get("tool_name"),
                "ok" if used.get("success") else "FAILED",
                preview(used.get("content", ""), 200),
            )
        content = str(result.get("content", ""))
        if cancel_event is not None and cancel_event.is_set():
            logger.info(
                "Agent %s run stopped by the user after %.1fs",
                agent,
                time.monotonic() - started,
            )
            return cancellation.CANCELLED_MESSAGE
        logger.info(
            "Agent %s finished in %.1fs (%d turns, model %s): %s",
            agent,
            time.monotonic() - started,
            result.get("turns", 0),
            result.get("model"),
            preview(content),
        )
        return content


__all__ = [
    "APPROVAL_TIMEOUT_S",
    "AgentRunner",
    "ApprovalBroker",
    "format_approval_request",
]
