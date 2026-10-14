"""Telegram *user account* client (MTProto via Telethon).

Unlike :mod:`openjarvis.channels.telegram`, which drives a Bot API token, this
module logs in as a real Telegram user so agents can do everything that
account can: message anyone, edit the profile, manage groups, and place or
join calls (see :mod:`openjarvis.connectors.telegram_voice`).

Telethon is asyncio-native while OpenJarvis tools are synchronous and run on
worker threads, so one process-wide :class:`TelegramUserClient` owns a private
event loop on a daemon thread and tools submit coroutines to it via
:meth:`TelegramUserClient.run`.

Credentials (``TELEGRAM_API_ID`` / ``TELEGRAM_API_HASH`` from
https://my.telegram.org) live in ``credentials.toml`` under ``telegram_user``;
the authorized session lives in ``<config dir>/telegram/user.session``. Run
``jarvis telegram login`` once to create it.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import re
import threading
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Optional, TypeVar

from openjarvis.core.paths import get_config_dir

logger = logging.getLogger(__name__)

CREDENTIAL_TOOL = "telegram_user"
API_ID_ENV = "TELEGRAM_API_ID"
API_HASH_ENV = "TELEGRAM_API_HASH"

_T = TypeVar("_T")
_INVITE_LINK = re.compile(r"(?:t\.me|telegram\.me)/(?:\+|joinchat/)([\w-]+)")
_PUBLIC_LINK = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9_]{4,})/?$")


_PEER_REF = re.compile(r"([un])(\d+)")


def peer_ref(peer_id: int) -> str:
    """Opaque chat/user reference that survives the outbound PII filter.

    Bare Telegram ids (10 digits for newer users, ``-4…`` for basic groups)
    match the boundary guard's phone-number pattern and get redacted from
    tool arguments. A letter prefix glued to the digits has no word boundary,
    so ``u8123456789`` / ``n1001234567890`` pass through untouched.
    """
    return f"u{peer_id}" if peer_id > 0 else f"n{-peer_id}"


def parse_peer_ref(text: str) -> Optional[int]:
    match = _PEER_REF.fullmatch(text)
    if match is None:
        return None
    value = int(match.group(2))
    return value if match.group(1) == "u" else -value


class TelegramUserError(RuntimeError):
    """User-facing failure (missing setup, session in use, bad reference)."""


def telegram_dir() -> Path:
    """Directory holding the session file and its lock (created 0700)."""
    path = get_config_dir() / "telegram"
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def session_path() -> Path:
    """Telethon session file path (Telethon appends ``.session`` itself)."""
    return telegram_dir() / "user"


def is_logged_in_on_disk() -> bool:
    return session_path().with_suffix(".session").exists()


def load_api_credentials() -> tuple[int, str]:
    """Return ``(api_id, api_hash)`` or raise a setup hint."""
    from openjarvis.core.credentials import get_tool_credential

    api_id = get_tool_credential(CREDENTIAL_TOOL, API_ID_ENV)
    api_hash = get_tool_credential(CREDENTIAL_TOOL, API_HASH_ENV)
    if not api_id or not api_hash:
        raise TelegramUserError(
            "Telegram API credentials are missing. Run `jarvis telegram login` "
            "(get api_id/api_hash at https://my.telegram.org)."
        )
    try:
        return int(api_id), api_hash
    except ValueError as exc:
        raise TelegramUserError("TELEGRAM_API_ID must be an integer.") from exc


def import_telethon() -> Any:
    try:
        import telethon
    except ImportError as exc:
        raise TelegramUserError(
            "Telethon is not installed. Install with: uv sync --extra telegram-user"
        ) from exc
    return telethon


class _SessionLock:
    """Exclusive advisory lock so two processes never share one auth key.

    Telegram invalidates (``AUTH_KEY_DUPLICATED``) sessions used concurrently
    from two connections, and Telethon's SQLite session would also hit
    "database is locked".
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any = None

    def acquire(self) -> None:
        fh = open(self._path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise TelegramUserError(
                "The Telegram session is in use by another OpenJarvis process "
                "(for example `jarvis telegram listen`). Stop it or run this "
                "task through that process."
            ) from None
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


class TelegramUserClient:
    """Process-wide Telethon client running on a private event loop."""

    _instance: Optional["TelegramUserClient"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        from openjarvis.connectors.telegram_logging import ensure_telegram_logging

        ensure_telegram_logging()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._calls: Any = None
        self._lock = _SessionLock(telegram_dir() / "session.lock")
        self._connect_lock: Optional[asyncio.Lock] = None
        self._start_lock = threading.Lock()
        # Work submitted from other threads (tools); a "stop" cancels it.
        self._inflight: set[Future[Any]] = set()
        self._inflight_lock = threading.Lock()

    @classmethod
    def get(cls) -> "TelegramUserClient":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # -- loop plumbing -------------------------------------------------

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="telegram-user-loop", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(
        self,
        func: Callable[[Any], Awaitable[_T]],
        *,
        timeout: float = 60.0,
    ) -> _T:
        """Run ``func(client)`` on the Telegram loop and return its result."""

        async def _call() -> _T:
            client = await self.client()
            return await func(client)

        future = self._track(asyncio.run_coroutine_threadsafe(_call(), self.loop))
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TelegramUserError(f"Telegram request timed out after {timeout:.0f}s")
        except CancelledError:
            raise TelegramUserError("Stopped by the user.") from None

    def submit(self, coro: Coroutine[Any, Any, _T]) -> "Future[_T]":
        """Schedule a coroutine on the Telegram loop without waiting."""
        return self._track(asyncio.run_coroutine_threadsafe(coro, self.loop))

    def _track(self, future: "Future[_T]") -> "Future[_T]":
        with self._inflight_lock:
            self._inflight.add(future)
        future.add_done_callback(self._untrack)
        return future

    def _untrack(self, future: "Future[Any]") -> None:
        with self._inflight_lock:
            self._inflight.discard(future)

    def cancel_inflight(self) -> int:
        """Cancel every pending Telegram operation (downloads, joins, ...)."""
        with self._inflight_lock:
            pending = [f for f in self._inflight if not f.done()]
        return sum(1 for f in pending if f.cancel())

    # -- client lifecycle ---------------------------------------------

    async def client(self) -> Any:
        """Return the connected, authorized Telethon client (lazy)."""
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self._client is not None and self._client.is_connected():
                return self._client
            telethon = import_telethon()
            api_id, api_hash = load_api_credentials()
            if not is_logged_in_on_disk():
                raise TelegramUserError(
                    "No Telegram session. Run `jarvis telegram login` first."
                )
            if self._client is None:
                self._lock.acquire()
                self._client = telethon.TelegramClient(
                    str(session_path()), api_id, api_hash
                )
            await self._client.connect()
            if not await self._client.is_user_authorized():
                await self._client.disconnect()
                raise TelegramUserError(
                    "The Telegram session is no longer authorized. "
                    "Run `jarvis telegram login` again."
                )
            return self._client

    async def calls(self) -> Any:
        """Return the started PyTgCalls instance bound to this client."""
        if self._calls is not None:
            return self._calls
        client = await self.client()
        try:
            from pytgcalls import PyTgCalls
        except ImportError as exc:
            raise TelegramUserError(
                "py-tgcalls is not installed. Install with: "
                "uv sync --extra telegram-user"
            ) from exc
        calls = PyTgCalls(client)
        await calls.start()
        self._calls = calls
        return calls

    def shutdown(self) -> None:
        if self._loop is None:
            return

        async def _close() -> None:
            if self._client is not None:
                await self._client.disconnect()

        try:
            asyncio.run_coroutine_threadsafe(_close(), self._loop).result(10)
        except Exception:
            logger.debug("Telegram disconnect failed", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._lock.release()


# ---------------------------------------------------------------------------
# Entity / message helpers shared by tools and the voice engine
# ---------------------------------------------------------------------------


async def resolve_entity(client: Any, ref: Any) -> Any:
    """Resolve ``me``, numeric ids, ``@user``, phones, and t.me links."""
    if ref is None or (isinstance(ref, str) and not ref.strip()):
        raise TelegramUserError("A chat/user reference is required.")
    if isinstance(ref, int):
        return await client.get_entity(ref)
    text = str(ref).strip()
    if text.lower() in {"me", "self", "saved", "saved messages"}:
        return await client.get_me()
    if "[REDACTED" in text:
        raise TelegramUserError(
            "The chat reference was redacted by the security filter (it looked "
            "like a phone number). Use the @username, or the 'id' value exactly "
            "as a telegram_* tool returned it (e.g. u8123456789)."
        )
    peer_id = parse_peer_ref(text)
    if peer_id is not None:
        return await client.get_entity(peer_id)
    if re.fullmatch(r"-?\d+", text):
        return await client.get_entity(int(text))
    match = _PUBLIC_LINK.match(text)
    if match:
        text = match.group(1)
    try:
        return await client.get_entity(text)
    except ValueError:
        pass
    # Fall back to a dialog title match ("Mom", "Work group").
    lowered = text.lstrip("@").lower()
    async for dialog in client.iter_dialogs(limit=300):
        if (dialog.name or "").lower() == lowered:
            return dialog.entity
    async for dialog in client.iter_dialogs(limit=300):
        if lowered in (dialog.name or "").lower():
            return dialog.entity
    raise TelegramUserError(f"Could not find a Telegram chat or user for {ref!r}.")


def invite_hash(link: str) -> Optional[str]:
    match = _INVITE_LINK.search(link or "")
    return match.group(1) if match else None


def entity_summary(entity: Any) -> dict[str, Any]:
    """Compact, JSON-safe description of a user, group, or channel."""
    from telethon import utils

    kind = type(entity).__name__.lower()
    if kind == "user":
        kind = "bot" if getattr(entity, "bot", False) else "user"
    elif kind == "channel":
        kind = "supergroup" if getattr(entity, "megagroup", False) else "channel"
    elif kind == "chat":
        kind = "group"
    data: dict[str, Any] = {
        "id": peer_ref(utils.get_peer_id(entity)),
        "type": kind,
        "name": utils.get_display_name(entity),
    }
    for attr in ("username", "phone"):
        value = getattr(entity, attr, None)
        if value:
            data[attr] = value
    if getattr(entity, "participants_count", None):
        data["members"] = entity.participants_count
    return data


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def message_summary(message: Any, *, max_text: int = 4000) -> dict[str, Any]:
    """Compact, JSON-safe description of one message."""
    from telethon import utils

    sender = getattr(message, "sender", None)
    text = message.message or ""
    if len(text) > max_text:
        text = text[:max_text] + "…"
    data: dict[str, Any] = {
        "id": message.id,
        "date": _iso(message.date),
        "out": bool(message.out),
        "sender_id": peer_ref(message.sender_id) if message.sender_id else None,
        "sender": utils.get_display_name(sender) if sender else None,
        "text": text,
    }
    if message.media is not None:
        data["media"] = type(message.media).__name__.replace("MessageMedia", "")
        file = getattr(message, "file", None)
        if file is not None and file.name:
            data["file_name"] = file.name
    if message.reply_to_msg_id:
        data["reply_to"] = message.reply_to_msg_id
    if message.edit_date:
        data["edited"] = _iso(message.edit_date)
    if getattr(message, "action", None) is not None:
        data["action"] = type(message.action).__name__.replace("MessageAction", "")
    return data


def safe_media_path(raw: str) -> Path:
    """Resolve a local file the user referenced, rejecting missing paths."""
    path = Path(os.path.expanduser(raw)).resolve()
    if not path.is_file():
        raise TelegramUserError(f"File not found: {raw}")
    return path


__all__ = [
    "parse_peer_ref",
    "peer_ref",
    "API_HASH_ENV",
    "API_ID_ENV",
    "CREDENTIAL_TOOL",
    "TelegramUserClient",
    "TelegramUserError",
    "entity_summary",
    "import_telethon",
    "invite_hash",
    "is_logged_in_on_disk",
    "load_api_credentials",
    "message_summary",
    "resolve_entity",
    "safe_media_path",
    "session_path",
    "telegram_dir",
]
