"""Telegram user-account tools: messaging, profile, groups, admin, and calls.

These act as the logged-in Telegram *user* (not a bot); run
``jarvis telegram login`` once first. Destructive or identity-changing
actions (deleting, leaving, blocking, moderation, profile edits, raw API
calls) set ``requires_confirmation`` so the agent must get the user's OK.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openjarvis.connectors.telegram_logging import preview
from openjarvis.connectors.telegram_user import (
    TelegramUserClient,
    TelegramUserError,
    entity_summary,
    invite_hash,
    message_summary,
    peer_ref,
    resolve_entity,
    safe_media_path,
)
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_CHAT = {
    "type": "string",
    "description": (
        "Chat or user: @username, an 'id' exactly as returned by a telegram_* "
        "tool (e.g. u8123456789), phone, t.me link, an exact chat title, or "
        "'me' for Saved Messages. Prefer @username when known."
    ),
}
_MESSAGE_IDS = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
_MAX_RESULT_CHARS = 20000
logger = logging.getLogger(__name__)
_CALL_WAIT_LIMIT_MIN = 60


class _TelegramTool(BaseTool):
    """Shared plumbing: run on the Telegram loop, JSON results, error mapping."""

    is_local = False
    name = ""
    description = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    confirm = False
    capabilities: tuple[str, ...] = ("network:fetch",)
    timeout = 60.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.tool_id = cls.name

    def __init__(self, **_: Any) -> None:
        pass

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            category="telegram",
            requires_confirmation=self.confirm,
            timeout_seconds=self.timeout,
            required_capabilities=list(self.capabilities),
            metadata={"setup": "jarvis telegram login"},
        )

    def execute(self, **params: Any) -> ToolResult:
        started = time.monotonic()
        logger.info(
            "%s called with %s", self.name, preview(json.dumps(params, default=str))
        )
        try:
            data = self._execute(params)
        except TelegramUserError as exc:
            logger.warning("%s failed: %s", self.name, exc)
            return self._result(str(exc), success=False)
        except Exception as exc:  # Telethon RPC errors carry useful names
            logger.exception("%s raised %s", self.name, type(exc).__name__)
            return self._result(f"{type(exc).__name__}: {exc}", success=False)
        content = json.dumps(data, ensure_ascii=False, default=str)
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS] + "… (truncated)"
        logger.info(
            "%s ok in %.1fs: %s",
            self.name,
            time.monotonic() - started,
            preview(content, 200),
        )
        return self._result(content)

    def _execute(self, params: dict[str, Any]) -> Any:
        return TelegramUserClient.get().run(
            lambda client: self._run(client, params), timeout=self.timeout - 5
        )

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _result(self, content: str, *, success: bool = True) -> ToolResult:
        return ToolResult(tool_name=self.name, content=content, success=success)


def _require(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise TelegramUserError(f"'{key}' is required.")
    return value


# ---------------------------------------------------------------------------
# Account and lookup
# ---------------------------------------------------------------------------


@ToolRegistry.register("telegram_me")
class TelegramMeTool(_TelegramTool):
    name = "telegram_me"
    capabilities = ("network:fetch",)
    description = "Show the logged-in Telegram account: name, username, phone, bio."

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl.functions.users import GetFullUserRequest

        me = await client.get_me()
        full = await client(GetFullUserRequest(me))
        return {**entity_summary(me), "bio": full.full_user.about or ""}


@ToolRegistry.register("telegram_list_chats")
class TelegramListChatsTool(_TelegramTool):
    name = "telegram_list_chats"
    capabilities = ("network:fetch",)
    description = (
        "List the account's chats (users, groups, channels) with unread counts "
        "and the latest message. Optionally filter by name or unread only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "query": {"type": "string", "description": "Substring of chat name."},
            "unread_only": {"type": "boolean"},
            "archived": {"type": "boolean", "description": "List archived chats."},
        },
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        limit = int(params.get("limit") or 30)
        query = str(params.get("query") or "").lower()
        chats = []
        async for dialog in client.iter_dialogs(archived=bool(params.get("archived"))):
            if query and query not in (dialog.name or "").lower():
                continue
            if params.get("unread_only") and not dialog.unread_count:
                continue
            item = entity_summary(dialog.entity)
            item["unread"] = dialog.unread_count
            if dialog.pinned:
                item["pinned"] = True
            if dialog.message is not None:
                item["last_message"] = message_summary(dialog.message, max_text=200)
            chats.append(item)
            if len(chats) >= limit:
                break
        return chats


@ToolRegistry.register("telegram_get_entity")
class TelegramGetEntityTool(_TelegramTool):
    name = "telegram_get_entity"
    capabilities = ("network:fetch",)
    description = (
        "Look up a Telegram user, group, or channel: id, type, username, bio or "
        "description, member count, and whether a voice chat is active."
    )
    parameters = {
        "type": "object",
        "properties": {"chat": _CHAT},
        "required": ["chat"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        entity = await resolve_entity(client, _require(params, "chat"))
        data = entity_summary(entity)
        if isinstance(entity, types.User):
            full = await client(functions.users.GetFullUserRequest(entity))
            data["bio"] = full.full_user.about or ""
            data["mutual_contact"] = bool(entity.mutual_contact)
            if entity.status is not None:
                data["status"] = type(entity.status).__name__.replace("UserStatus", "")
        elif isinstance(entity, types.Channel):
            full = await client(functions.channels.GetFullChannelRequest(entity))
            data["about"] = full.full_chat.about or ""
            data["members"] = full.full_chat.participants_count
            data["voice_chat_active"] = full.full_chat.call is not None
        else:
            full = await client(functions.messages.GetFullChatRequest(entity.id))
            data["about"] = full.full_chat.about or ""
            data["voice_chat_active"] = full.full_chat.call is not None
        return data


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@ToolRegistry.register("telegram_read_messages")
class TelegramReadMessagesTool(_TelegramTool):
    name = "telegram_read_messages"
    capabilities = ("network:fetch",)
    description = (
        "Read recent messages from a chat (newest first), or search messages. "
        "Omit 'chat' with 'search' to search across all chats."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "search": {"type": "string", "description": "Text to search for."},
            "from_user": {"type": "string", "description": "Only this sender."},
            "before_id": {
                "type": "integer",
                "description": "Page backwards: only messages older than this id.",
            },
        },
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        chat = params.get("chat")
        search = params.get("search") or None
        if not chat and not search:
            raise TelegramUserError("Provide 'chat', 'search', or both.")
        entity = await resolve_entity(client, chat) if chat else None
        from_user = (
            await resolve_entity(client, params["from_user"])
            if params.get("from_user")
            else None
        )
        messages = []
        async for message in client.iter_messages(
            entity,
            limit=int(params.get("limit") or 20),
            search=search,
            from_user=from_user,
            offset_id=int(params.get("before_id") or 0),
        ):
            item = message_summary(message)
            if entity is None and message.chat is not None:
                item["chat"] = entity_summary(message.chat)
            messages.append(item)
        return messages


@ToolRegistry.register("telegram_send_message")
class TelegramSendMessageTool(_TelegramTool):
    name = "telegram_send_message"
    capabilities = ("channel:send", "file:read")
    description = (
        "Send a message as the user: text (Markdown), optionally with files "
        "(photos, videos, documents, or an audio file as a voice note), as a "
        "reply, silently, or scheduled for later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "text": {"type": "string", "description": "Message text or caption."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Local file paths to attach (several = album).",
            },
            "voice_note": {
                "type": "boolean",
                "description": "Send the single audio file as a voice message.",
            },
            "reply_to": {"type": "integer", "description": "Message id to reply to."},
            "silent": {"type": "boolean", "description": "No notification sound."},
            "schedule_minutes": {
                "type": "integer",
                "minimum": 1,
                "description": "Schedule delivery this many minutes from now.",
            },
        },
        "required": ["chat"],
    }
    timeout = 300.0

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        entity = await resolve_entity(client, _require(params, "chat"))
        text = str(params.get("text") or "")
        files = [str(safe_media_path(f)) for f in params.get("files") or []]
        if not text and not files:
            raise TelegramUserError("Provide 'text' and/or 'files'.")
        schedule = None
        if params.get("schedule_minutes"):
            schedule = datetime.now(timezone.utc) + timedelta(
                minutes=int(params["schedule_minutes"])
            )
        common = {
            "reply_to": params.get("reply_to"),
            "silent": bool(params.get("silent")),
            "schedule": schedule,
            "parse_mode": "md",
        }
        if files:
            sent = await client.send_file(
                entity,
                files if len(files) > 1 else files[0],
                caption=text or None,
                voice_note=bool(params.get("voice_note")),
                **common,
            )
        else:
            sent = await client.send_message(entity, text, **common)
        sent_list = sent if isinstance(sent, list) else [sent]
        return {
            "chat": entity_summary(entity),
            "message_ids": [m.id for m in sent_list],
            "scheduled_for": schedule,
        }


@ToolRegistry.register("telegram_edit_message")
class TelegramEditMessageTool(_TelegramTool):
    name = "telegram_edit_message"
    capabilities = ("channel:send",)
    description = "Edit the text of a message the user sent."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "message_id": {"type": "integer"},
            "text": {"type": "string"},
        },
        "required": ["chat", "message_id", "text"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        entity = await resolve_entity(client, _require(params, "chat"))
        message = await client.edit_message(
            entity,
            int(params["message_id"]),
            str(_require(params, "text")),
            parse_mode="md",
        )
        return message_summary(message)


@ToolRegistry.register("telegram_forward_messages")
class TelegramForwardMessagesTool(_TelegramTool):
    name = "telegram_forward_messages"
    capabilities = ("channel:send",)
    description = "Forward messages from one chat to another."
    parameters = {
        "type": "object",
        "properties": {
            "from_chat": _CHAT,
            "message_ids": _MESSAGE_IDS,
            "to_chat": _CHAT,
            "hide_author": {"type": "boolean", "description": "Send as a copy."},
        },
        "required": ["from_chat", "message_ids", "to_chat"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        source = await resolve_entity(client, _require(params, "from_chat"))
        target = await resolve_entity(client, _require(params, "to_chat"))
        sent = await client.forward_messages(
            target,
            [int(i) for i in _require(params, "message_ids")],
            source,
            drop_author=bool(params.get("hide_author")) or None,
        )
        return {"to": entity_summary(target), "message_ids": [m.id for m in sent if m]}


@ToolRegistry.register("telegram_message_action")
class TelegramMessageActionTool(_TelegramTool):
    name = "telegram_message_action"
    capabilities = ("channel:send",)
    description = (
        "React to, pin, or unpin a message, or mark a chat as read. "
        "'react' needs an emoji (empty string removes the reaction)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "action": {
                "type": "string",
                "enum": ["react", "pin", "unpin", "mark_read"],
            },
            "message_id": {"type": "integer"},
            "emoji": {"type": "string"},
            "notify": {"type": "boolean", "description": "Notify members on pin."},
        },
        "required": ["chat", "action"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        entity = await resolve_entity(client, _require(params, "chat"))
        action = params["action"]
        message_id = params.get("message_id")
        if action == "mark_read":
            await client.send_read_acknowledge(entity, clear_mentions=True)
        elif message_id is None and action != "unpin":
            raise TelegramUserError("'message_id' is required for this action.")
        elif action == "react":
            emoji = str(params.get("emoji") or "")
            reaction = [types.ReactionEmoji(emoticon=emoji)] if emoji else []
            await client(
                functions.messages.SendReactionRequest(
                    peer=entity, msg_id=int(message_id), reaction=reaction
                )
            )
        elif action == "pin":
            await client.pin_message(
                entity, int(message_id), notify=bool(params.get("notify"))
            )
        else:
            await client.unpin_message(entity, message_id)
        return {"ok": True, "action": action, "chat": entity_summary(entity)}


@ToolRegistry.register("telegram_download_media")
class TelegramDownloadMediaTool(_TelegramTool):
    name = "telegram_download_media"
    capabilities = ("network:fetch", "file:write")
    description = (
        "Download the photo, video, voice note, or document attached to a "
        "message and return the local file path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "message_id": {"type": "integer"},
            "directory": {
                "type": "string",
                "description": "Target folder (default ~/Downloads/telegram).",
            },
        },
        "required": ["chat", "message_id"],
    }
    timeout = 600.0

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        entity = await resolve_entity(client, _require(params, "chat"))
        message = await client.get_messages(entity, ids=int(params["message_id"]))
        if message is None or message.media is None:
            raise TelegramUserError("That message has no downloadable media.")
        directory = Path(
            os.path.expanduser(params.get("directory") or "~/Downloads/telegram")
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = await client.download_media(message, file=str(directory) + os.sep)
        return {"path": path, "message": message_summary(message)}


# ---------------------------------------------------------------------------
# Contacts, groups, channels
# ---------------------------------------------------------------------------


@ToolRegistry.register("telegram_contacts")
class TelegramContactsTool(_TelegramTool):
    name = "telegram_contacts"
    capabilities = ("network:fetch", "channel:send")
    description = "List contacts, or add a contact by phone number."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "add"]},
            "query": {"type": "string", "description": "Filter for 'list'."},
            "phone": {"type": "string"},
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
        },
        "required": ["action"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        if params["action"] == "add":
            result = await client(
                functions.contacts.ImportContactsRequest(
                    [
                        types.InputPhoneContact(
                            client_id=int(time.time()),
                            phone=str(_require(params, "phone")),
                            first_name=str(_require(params, "first_name")),
                            last_name=str(params.get("last_name") or ""),
                        )
                    ]
                )
            )
            if not result.users:
                raise TelegramUserError("That phone number is not on Telegram.")
            return entity_summary(result.users[0])
        contacts = await client(functions.contacts.GetContactsRequest(hash=0))
        query = str(params.get("query") or "").lower()
        users = [entity_summary(u) for u in contacts.users]
        return [u for u in users if query in json.dumps(u).lower()] if query else users


@ToolRegistry.register("telegram_create_chat")
class TelegramCreateChatTool(_TelegramTool):
    name = "telegram_create_chat"
    capabilities = ("channel:send",)
    description = "Create a group, supergroup, or broadcast channel."
    parameters = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["group", "supergroup", "channel"]},
            "title": {"type": "string"},
            "about": {"type": "string"},
            "users": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Users to add (a basic group needs at least one).",
            },
        },
        "required": ["type", "title"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions

        users = [await resolve_entity(client, u) for u in params.get("users") or []]
        title = str(_require(params, "title"))
        if params["type"] == "group":
            if not users:
                raise TelegramUserError("A basic group needs at least one user.")
            result = await client(functions.messages.CreateChatRequest(users, title))
            updates = getattr(result, "updates", result)
            chat = updates.chats[0]
        else:
            result = await client(
                functions.channels.CreateChannelRequest(
                    title=title,
                    about=str(params.get("about") or ""),
                    broadcast=params["type"] == "channel",
                    megagroup=params["type"] == "supergroup",
                )
            )
            chat = result.chats[0]
            if users:
                await client(functions.channels.InviteToChannelRequest(chat, users))
        return entity_summary(chat)


@ToolRegistry.register("telegram_join_chat")
class TelegramJoinChatTool(_TelegramTool):
    name = "telegram_join_chat"
    capabilities = ("channel:send",)
    description = (
        "Join a public group/channel by @username or link, or a private invite link."
    )
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions

        target = str(_require(params, "target"))
        hash_ = invite_hash(target)
        if hash_:
            result = await client(functions.messages.ImportChatInviteRequest(hash_))
            return entity_summary(result.chats[0])
        entity = await resolve_entity(client, target)
        await client(functions.channels.JoinChannelRequest(entity))
        return entity_summary(entity)


@ToolRegistry.register("telegram_invite_users")
class TelegramInviteUsersTool(_TelegramTool):
    name = "telegram_invite_users"
    capabilities = ("channel:send",)
    description = "Add users to a group or channel, or create an invite link."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "users": {"type": "array", "items": {"type": "string"}},
            "create_link": {"type": "boolean"},
        },
        "required": ["chat"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        chat = await resolve_entity(client, _require(params, "chat"))
        users = [await resolve_entity(client, u) for u in params.get("users") or []]
        for user in users:
            if isinstance(chat, types.Channel):
                await client(functions.channels.InviteToChannelRequest(chat, [user]))
            else:
                await client(
                    functions.messages.AddChatUserRequest(chat.id, user, fwd_limit=50)
                )
        data: dict[str, Any] = {"added": [entity_summary(u) for u in users]}
        if params.get("create_link"):
            invite = await client(functions.messages.ExportChatInviteRequest(chat))
            data["invite_link"] = invite.link
        return data


@ToolRegistry.register("telegram_edit_chat")
class TelegramEditChatTool(_TelegramTool):
    name = "telegram_edit_chat"
    capabilities = ("channel:send", "file:read")
    description = "Change a group or channel's title, description, or photo."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "title": {"type": "string"},
            "about": {"type": "string"},
            "photo_path": {"type": "string"},
        },
        "required": ["chat"],
    }
    timeout = 180.0

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        chat = await resolve_entity(client, _require(params, "chat"))
        is_channel = isinstance(chat, types.Channel)
        changed = []
        if params.get("title"):
            title = str(params["title"])
            if is_channel:
                await client(functions.channels.EditTitleRequest(chat, title))
            else:
                await client(functions.messages.EditChatTitleRequest(chat.id, title))
            changed.append("title")
        if params.get("about") is not None:
            await client(
                functions.messages.EditChatAboutRequest(chat, str(params["about"]))
            )
            changed.append("about")
        if params.get("photo_path"):
            upload = await client.upload_file(
                str(safe_media_path(params["photo_path"]))
            )
            photo = types.InputChatUploadedPhoto(file=upload)
            if is_channel:
                await client(functions.channels.EditPhotoRequest(chat, photo))
            else:
                await client(functions.messages.EditChatPhotoRequest(chat.id, photo))
            changed.append("photo")
        return {"chat": entity_summary(chat), "changed": changed}


@ToolRegistry.register("telegram_participants")
class TelegramParticipantsTool(_TelegramTool):
    name = "telegram_participants"
    capabilities = ("network:fetch",)
    description = "List members of a group or channel (admins only if requested)."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "search": {"type": "string"},
            "admins_only": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "required": ["chat"],
    }

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import types

        chat = await resolve_entity(client, _require(params, "chat"))
        members = await client.get_participants(
            chat,
            limit=int(params.get("limit") or 100),
            search=str(params.get("search") or ""),
            filter=types.ChannelParticipantsAdmins
            if params.get("admins_only")
            else None,
        )
        return [entity_summary(m) for m in members]


# ---------------------------------------------------------------------------
# Destructive / identity-changing (require confirmation)
# ---------------------------------------------------------------------------


@ToolRegistry.register("telegram_update_profile")
class TelegramUpdateProfileTool(_TelegramTool):
    name = "telegram_update_profile"
    capabilities = ("channel:send", "file:read")
    description = (
        "Change the user's own profile: first/last name, bio, username, or "
        "profile photo (set a new one or delete the current one)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "bio": {"type": "string"},
            "username": {"type": "string", "description": "Empty string removes it."},
            "photo_path": {"type": "string"},
            "delete_current_photo": {"type": "boolean"},
        },
    }
    confirm = True
    timeout = 180.0

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions, types

        changed = []
        names = {k: params.get(k) for k in ("first_name", "last_name", "bio")}
        if any(v is not None for v in names.values()):
            await client(
                functions.account.UpdateProfileRequest(
                    first_name=names["first_name"],
                    last_name=names["last_name"],
                    about=names["bio"],
                )
            )
            changed += [k for k, v in names.items() if v is not None]
        if params.get("username") is not None:
            await client(
                functions.account.UpdateUsernameRequest(
                    str(params["username"]).lstrip("@")
                )
            )
            changed.append("username")
        if params.get("delete_current_photo"):
            photos = await client.get_profile_photos("me", limit=1)
            if photos:
                await client(
                    functions.photos.DeletePhotosRequest(
                        [
                            types.InputPhoto(
                                id=photos[0].id,
                                access_hash=photos[0].access_hash,
                                file_reference=photos[0].file_reference,
                            )
                        ]
                    )
                )
                changed.append("photo deleted")
        if params.get("photo_path"):
            upload = await client.upload_file(
                str(safe_media_path(params["photo_path"]))
            )
            await client(functions.photos.UploadProfilePhotoRequest(file=upload))
            changed.append("photo")
        if not changed:
            raise TelegramUserError("Nothing to change.")
        return {"changed": changed, "profile": entity_summary(await client.get_me())}


@ToolRegistry.register("telegram_delete_messages")
class TelegramDeleteMessagesTool(_TelegramTool):
    name = "telegram_delete_messages"
    capabilities = ("channel:send",)
    description = "Delete messages (for everyone by default)."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "message_ids": _MESSAGE_IDS,
            "for_everyone": {"type": "boolean"},
        },
        "required": ["chat", "message_ids"],
    }
    confirm = True

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        entity = await resolve_entity(client, _require(params, "chat"))
        ids = [int(i) for i in _require(params, "message_ids")]
        await client.delete_messages(
            entity, ids, revoke=params.get("for_everyone") is not False
        )
        return {"deleted": ids, "chat": entity_summary(entity)}


@ToolRegistry.register("telegram_leave_chat")
class TelegramLeaveChatTool(_TelegramTool):
    name = "telegram_leave_chat"
    capabilities = ("channel:send",)
    description = "Leave a group/channel, or delete a private chat's history."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "delete_for_everyone": {
                "type": "boolean",
                "description": "Private chats: also delete for the other side.",
            },
        },
        "required": ["chat"],
    }
    confirm = True

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        entity = await resolve_entity(client, _require(params, "chat"))
        await client.delete_dialog(
            entity, revoke=bool(params.get("delete_for_everyone"))
        )
        return {"left": entity_summary(entity)}


@ToolRegistry.register("telegram_moderate")
class TelegramModerateTool(_TelegramTool):
    name = "telegram_moderate"
    capabilities = ("channel:send",)
    description = (
        "Moderate a group/channel member: ban, unban, kick, mute, unmute, "
        "promote to admin (optional title), or demote."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "user": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["ban", "unban", "kick", "mute", "unmute", "promote", "demote"],
            },
            "admin_title": {"type": "string"},
        },
        "required": ["chat", "user", "action"],
    }
    confirm = True

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        chat = await resolve_entity(client, _require(params, "chat"))
        user = await resolve_entity(client, _require(params, "user"))
        action = params["action"]
        if action == "ban":
            await client.edit_permissions(chat, user, view_messages=False)
        elif action == "unban":
            await client.edit_permissions(chat, user)
        elif action == "kick":
            await client.kick_participant(chat, user)
        elif action == "mute":
            await client.edit_permissions(chat, user, send_messages=False)
        elif action == "unmute":
            await client.edit_permissions(chat, user, send_messages=True)
        elif action == "promote":
            await client.edit_admin(
                chat,
                user,
                is_admin=True,
                title=params.get("admin_title") or None,
            )
        else:
            await client.edit_admin(chat, user, is_admin=False)
        return {
            "action": action,
            "chat": entity_summary(chat),
            "user": entity_summary(user),
        }


@ToolRegistry.register("telegram_block_user")
class TelegramBlockUserTool(_TelegramTool):
    name = "telegram_block_user"
    capabilities = ("channel:send",)
    description = "Block or unblock a user, or delete them from contacts."
    parameters = {
        "type": "object",
        "properties": {
            "user": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["block", "unblock", "delete_contact"],
            },
        },
        "required": ["user", "action"],
    }
    confirm = True

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions

        user = await resolve_entity(client, _require(params, "user"))
        action = params["action"]
        if action == "block":
            await client(functions.contacts.BlockRequest(user))
        elif action == "unblock":
            await client(functions.contacts.UnblockRequest(user))
        else:
            await client(functions.contacts.DeleteContactsRequest([user]))
        return {"action": action, "user": entity_summary(user)}


_PEER_PARAMS = {"peer", "user_id", "channel", "bot", "from_peer", "to_peer", "send_as"}


async def _build_tl_value(client: Any, key: str, value: Any) -> Any:
    """Turn JSON into TL objects: peers by reference, ``{"_": "Type"}`` dicts."""
    from telethon.tl import types

    if isinstance(value, dict) and "_" in value:
        cls = getattr(types, str(value["_"]), None)
        if cls is None:
            raise TelegramUserError(f"Unknown TL type {value['_']!r}.")
        fields = {
            k: await _build_tl_value(client, k, v) for k, v in value.items() if k != "_"
        }
        return cls(**fields)
    if isinstance(value, list):
        return [await _build_tl_value(client, key, v) for v in value]
    is_peer = key in _PEER_PARAMS or (key in {"users", "id"} and isinstance(value, str))
    if is_peer and isinstance(value, str):
        return await client.get_input_entity(await resolve_entity(client, value))
    if is_peer and isinstance(value, int) and not isinstance(value, bool):
        return await client.get_input_entity(value)
    return value


@ToolRegistry.register("telegram_api_call")
class TelegramApiCallTool(_TelegramTool):
    name = "telegram_api_call"
    capabilities = ("channel:send", "system:admin")
    description = (
        "Advanced escape hatch: call any raw Telegram MTProto method by name, "
        "e.g. 'account.UpdateStatusRequest' with {\"offline\": false}, "
        "'account.GetAuthorizationsRequest', or 'account.SetPrivacyRequest'. "
        "Peer fields (peer, user_id, channel, users) accept @usernames/ids; "
        'nested objects are {"_": "TypeName", ...}. Use only when no '
        "dedicated telegram_* tool fits."
    )
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "e.g. 'messages.GetHistoryRequest'",
            },
            "params": {"type": "object"},
        },
        "required": ["method"],
    }
    confirm = True
    timeout = 120.0

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from telethon.tl import functions

        target: Any = functions
        for part in str(_require(params, "method")).split("."):
            target = getattr(target, part, None)
            if target is None:
                raise TelegramUserError(
                    f"Unknown Telegram method {params['method']!r}."
                )
        kwargs = {
            k: await _build_tl_value(client, k, v)
            for k, v in (params.get("params") or {}).items()
        }
        result = await client(target(**kwargs))
        return result.to_dict() if hasattr(result, "to_dict") else result


# ---------------------------------------------------------------------------
# Calls and voice chats
# ---------------------------------------------------------------------------


@ToolRegistry.register("telegram_call_start")
class TelegramCallStartTool(_TelegramTool):
    name = "telegram_call_start"
    capabilities = ("channel:send",)
    description = (
        "Call a user (private voice call) or join/start a group's voice chat. "
        "In 'conversation' mode the assistant listens, transcribes, and talks "
        "back in real time following 'instructions'; 'silent' mode only joins "
        "so you can play audio with telegram_call_speak. By default this "
        "waits and returns the full transcript once the call ends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "mode": {"type": "string", "enum": ["conversation", "silent"]},
            "instructions": {
                "type": "string",
                "description": (
                    "Goal/persona for the call, e.g. 'Book a table for 2 at 8pm'."
                ),
            },
            "greeting": {"type": "string", "description": "First thing to say."},
            "max_minutes": {"type": "number", "minimum": 0.5, "maximum": 120},
            "wait": {
                "type": "boolean",
                "description": (
                    "Block until the call ends and return the transcript. "
                    "Defaults to true except under `jarvis telegram listen`."
                ),
            },
            "invite_link": {
                "type": "string",
                "description": "Voice chat invite link (for voice chats needing one).",
            },
        },
        "required": ["chat"],
    }
    timeout = _CALL_WAIT_LIMIT_MIN * 60 + 120.0

    def _execute(self, params: dict[str, Any]) -> Any:
        from openjarvis.connectors.telegram_voice import VoiceCallManager

        tg = TelegramUserClient.get()
        manager = VoiceCallManager.get()
        max_minutes = min(float(params.get("max_minutes") or 30), _CALL_WAIT_LIMIT_MIN)
        session = tg.submit(
            manager.join(
                _require(params, "chat"),
                conversational=params.get("mode", "conversation") != "silent",
                instructions=str(params.get("instructions") or ""),
                greeting=str(params.get("greeting") or ""),
                max_minutes=max_minutes,
                invite_link=str(params.get("invite_link") or ""),
            )
        ).result(timeout=120)
        wait = params.get("wait")
        if wait is None:
            # One-shot runs (`jarvis ask`/`chat`) exit after the reply, which
            # would hang up immediately; only the long-lived listener can leave
            # a call running in the background.
            wait = not manager.persistent
        if wait:
            session.ended.wait(timeout=max_minutes * 60 + 30)
        return session.snapshot()


@ToolRegistry.register("telegram_call_speak")
class TelegramCallSpeakTool(_TelegramTool):
    name = "telegram_call_speak"
    capabilities = ("channel:send", "file:read")
    description = "In an active call, say text out loud (TTS) or play an audio file."
    parameters = {
        "type": "object",
        "properties": {
            "chat": _CHAT,
            "text": {"type": "string"},
            "audio_path": {"type": "string"},
            "interrupt": {
                "type": "boolean",
                "description": "Drop queued speech first.",
            },
        },
        "required": ["chat"],
    }
    timeout = 300.0

    def _execute(self, params: dict[str, Any]) -> Any:
        from openjarvis.connectors.telegram_voice import (
            VoiceCallManager,
            decode_to_pcm,
            resolve_chat_id,
        )

        tg = TelegramUserClient.get()

        async def _speak() -> dict[str, Any]:
            session = VoiceCallManager.get().session(
                await resolve_chat_id(_require(params, "chat"))
            )
            if params.get("interrupt"):
                session.drop_queued_audio()
            if params.get("audio_path"):
                path = str(safe_media_path(params["audio_path"]))
                session.enqueue_pcm(decode_to_pcm(path=path))
            elif params.get("text"):
                await session.say(str(params["text"]))
                session.transcript.append(
                    {
                        "role": "assistant",
                        "speaker": "assistant",
                        "content": params["text"],
                    }
                )
            else:
                raise TelegramUserError("Provide 'text' or 'audio_path'.")
            return {"ok": True, "chat": peer_ref(session.chat_id)}

        return tg.submit(_speak()).result(timeout=self.timeout - 5)


@ToolRegistry.register("telegram_call_control")
class TelegramCallControlTool(_TelegramTool):
    name = "telegram_call_control"
    capabilities = ("channel:send",)
    description = (
        "Manage calls: 'status' (active calls), 'transcript' of a call, "
        "'mute'/'unmute' the assistant's microphone, or 'leave' (hang up)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "transcript", "mute", "unmute", "leave"],
            },
            "chat": _CHAT,
        },
        "required": ["action"],
    }

    def _execute(self, params: dict[str, Any]) -> Any:
        from openjarvis.connectors.telegram_voice import (
            VoiceCallManager,
            resolve_chat_id,
        )

        tg = TelegramUserClient.get()
        manager = VoiceCallManager.get()
        action = params["action"]
        if action == "status":
            return {
                "active": [
                    {k: v for k, v in s.snapshot().items() if k != "transcript"}
                    for s in manager.sessions.values()
                ],
                "recent": [
                    {k: v for k, v in s.items() if k != "transcript"}
                    for s in manager.history
                ],
            }

        async def _act() -> Any:
            chat_id = await resolve_chat_id(_require(params, "chat"))
            if action == "leave":
                return await manager.leave(chat_id)
            if action == "transcript":
                if chat_id in manager.sessions:
                    return manager.sessions[chat_id].snapshot()
                for past in reversed(manager.history):
                    if past["chat"] == peer_ref(chat_id):
                        return past
                raise TelegramUserError("No call with that chat in this session.")
            await manager.set_muted(chat_id, action == "mute")
            return {"ok": True, "muted": action == "mute"}

        return tg.submit(_act()).result(timeout=self.timeout - 5)


def _media_dirs() -> list[str]:
    from openjarvis.connectors.telegram_responder import parse_allowed
    from openjarvis.core.config import load_config

    return parse_allowed(load_config().tools.telegram_user.media_dirs)


def _resolve_media(query: str, source: str) -> Any:
    """Local folders first (unless source='youtube'), then YouTube."""
    from openjarvis.connectors.telegram_media import (
        find_local_media,
        resolve_youtube,
    )

    if source != "youtube":
        dirs = _media_dirs()
        local = find_local_media(query, dirs) if dirs else None
        if local is not None:
            return local
        if source == "local":
            raise TelegramUserError(
                f"No local media matches {query!r}. Use action 'list' to see files."
            )
    return resolve_youtube(query)


@ToolRegistry.register("telegram_call_media")
class TelegramCallMediaTool(_TelegramTool):
    name = "telegram_call_media"
    description = (
        "Stream music or video into a private call or a group voice chat "
        "(joins it first if needed), from the local media folders or YouTube. "
        "action 'list' shows the local media files. action 'play' takes 'query' "
        "(song/video name or YouTube link); set video=true only when the user "
        "asks for video, otherwise only the audio is streamed. 'seek' jumps "
        "with 'by' (seconds, negative rewinds) or 'to' ('1:23'). Also 'pause', "
        "'resume' (also 'play' without a query), 'stop' (back to your voice) "
        "and 'status' (title and position). In a call these also work by "
        "voice: 'forward 30 seconds', 'rewind 10', 'go to 1:23', 'pause'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "resume", "seek", "stop", "status", "list"],
            },
            "chat": _CHAT,
            "query": {
                "type": "string",
                "description": (
                    "What to play: a song/video/episode name, a YouTube link, or "
                    "(list) a filter for the local media folders."
                ),
            },
            "source": {
                "type": "string",
                "enum": ["auto", "local", "youtube"],
                "description": (
                    "auto (default): local media folders first, then YouTube. "
                    "Use 'local'/'youtube' when the user names the source."
                ),
            },
            "video": {
                "type": "boolean",
                "description": "Stream the video too (default false = audio only).",
            },
            "to": {
                "type": "string",
                "description": "seek: absolute position, e.g. '1:23' or '90'.",
            },
            "by": {
                "type": "number",
                "description": (
                    "seek: seconds to jump; positive = forward, negative = rewind."
                ),
            },
            "invite_link": {"type": "string"},
            "wait": {
                "type": "boolean",
                "description": (
                    "Block until the track ends. Defaults to true except under "
                    "`jarvis telegram listen`."
                ),
            },
        },
        "required": ["action", "chat"],
    }
    capabilities = ("channel:send", "network:fetch")
    timeout = 4 * 3600.0

    def _execute(self, params: dict[str, Any]) -> Any:
        from openjarvis.connectors.telegram_media import (
            describe_local_media,
            parse_timestamp,
        )
        from openjarvis.connectors.telegram_voice import (
            VoiceCallManager,
            media_status,
            resolve_chat_id,
        )

        tg = TelegramUserClient.get()
        manager = VoiceCallManager.get()
        action = params["action"]
        chat = _require(params, "chat")

        if action == "play" and not str(params.get("query") or "").strip():
            action = "resume"  # "play" on a paused track
        if action == "list":
            dirs = _media_dirs()
            if not dirs:
                raise TelegramUserError(
                    "No local media folders configured ([tools.telegram_user] "
                    "media_dirs)."
                )
            return {
                "folders": dirs,
                "files": describe_local_media(dirs, str(params.get("query") or "")),
            }
        if action == "play":
            info = _resolve_media(
                str(_require(params, "query")), str(params.get("source") or "auto")
            )
            video = bool(params.get("video"))
            session = tg.submit(
                manager.play_media(
                    chat,
                    info,
                    video=video,
                    invite_link=str(params.get("invite_link") or ""),
                )
            ).result(timeout=180)
            wait = params.get("wait")
            if wait is None:
                wait = not manager.persistent
            if wait:
                limit = (info.duration_s or 3600) + 120
                deadline = time.monotonic() + limit
                while session.media is not None and not session.ended.is_set():
                    if time.monotonic() > deadline:
                        break
                    time.sleep(1.0)
            return {
                "playing": info.as_dict(),
                "video": video,
                "chat": peer_ref(session.chat_id),
                "finished": session.media is None,
            }

        async def _control() -> Any:
            chat_id = await resolve_chat_id(chat)
            if action == "stop":
                return await manager.stop_media(chat_id, "stopped by request")
            if action == "pause":
                return await manager.pause_media(chat_id)
            if action == "resume":
                return await manager.resume_media(chat_id)
            if action == "seek":
                if params.get("to") not in (None, ""):
                    target = parse_timestamp(str(params["to"]))
                    if target is None:
                        raise TelegramUserError(
                            f"Can't read position {params['to']!r}."
                        )
                    return await manager.seek_media(chat_id, to=target)
                if params.get("by") is None:
                    raise TelegramUserError("seek needs 'by' (seconds) or 'to'.")
                return await manager.seek_media(chat_id, by=float(params["by"]))
            session = manager.sessions.get(chat_id)
            return {
                "in_call": session is not None,
                "media": media_status(session.media)
                if session and session.media
                else None,
            }

        return tg.submit(_control()).result(timeout=60)


@ToolRegistry.register("telegram_media_library")
class TelegramMediaLibraryTool(_TelegramTool):
    name = "telegram_media_library"
    description = (
        "Manage the local media folders that telegram_call_media streams from. "
        "'add' saves the audio/video file of a Telegram message (give 'chat' "
        "and 'message_id' — the message the user attached or replied to) into "
        "a media folder, optionally renamed with 'name'. 'remove' deletes the "
        "file best matching 'query'. 'list' shows the files. No approval needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "remove", "list"]},
            "chat": _CHAT,
            "message_id": {"type": "integer"},
            "name": {
                "type": "string",
                "description": "add: file name to save as (extension kept).",
            },
            "folder": {
                "type": "string",
                "description": "add: target media folder (default: the first one).",
            },
            "query": {"type": "string", "description": "remove/list: file to match."},
        },
        "required": ["action"],
    }
    capabilities = ("network:fetch", "file:write")
    timeout = 1800.0

    def _execute(self, params: dict[str, Any]) -> Any:
        from openjarvis.connectors import telegram_media as tm

        dirs = _media_dirs()
        if not dirs:
            raise TelegramUserError(
                "No local media folders configured ([tools.telegram_user] media_dirs)."
            )
        action = params["action"]
        if action == "list":
            return {
                "folders": dirs,
                "files": tm.describe_local_media(dirs, str(params.get("query") or "")),
            }
        if action == "remove":
            target = tm.match_local_media(str(_require(params, "query")), dirs)
            if target is None:
                raise TelegramUserError(
                    f"No media file clearly matches {params['query']!r}; use 'list'."
                )
            tm.local_media_info(target, dirs)  # enforces the folder boundary
            target.unlink()
            logger.info("Removed media file %s", target)
            return {"removed": target.name, "folder": str(target.parent)}
        # add
        folder = str(params.get("folder") or dirs[0])
        root = Path(os.path.expanduser(folder)).resolve()
        if not any(
            root == Path(os.path.expanduser(d)).resolve()
            or root.is_relative_to(Path(os.path.expanduser(d)).resolve())
            for d in dirs
        ):
            raise TelegramUserError(f"{folder} is not a configured media folder.")
        return super()._execute({**params, "_root": str(root)})

    async def _run(self, client: Any, params: dict[str, Any]) -> Any:
        from openjarvis.connectors import telegram_media as tm

        entity = await resolve_entity(client, _require(params, "chat"))
        message = await client.get_messages(
            entity, ids=int(_require(params, "message_id"))
        )
        if message is None or message.media is None or message.file is None:
            raise TelegramUserError("That message has no file to add.")
        ext = (message.file.ext or "").lower()
        if ext not in tm.MEDIA_EXTENSIONS:
            raise TelegramUserError(
                f"Only audio/video files can be added (got {ext or 'unknown type'})."
            )
        original = message.file.name or f"telegram_{message.id}{ext}"
        stem = tm.safe_media_stem(str(params.get("name") or Path(original).stem))
        root = Path(params["_root"])
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{stem}{ext}"
        counter = 2
        while target.exists():
            target = root / f"{stem} ({counter}){ext}"
            counter += 1
        path = await client.download_media(message, file=str(target))
        logger.info(
            "Added media file %s (%.1f MB)", path, (message.file.size or 0) / 1e6
        )
        return {
            "added": Path(path).name,
            "folder": str(root),
            "size_mb": round((message.file.size or 0) / 1_048_576, 1),
        }


TELEGRAM_USER_TOOLS = sorted(cls.name for cls in _TelegramTool.__subclasses__())

__all__ = ["TELEGRAM_USER_TOOLS"]
