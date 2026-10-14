---
name: telegram-user
description: Operate the owner's personal Telegram account — read and send messages, manage profile, contacts, groups and channels, and hold real-time voice calls.
version: 0.1.0
author: openjarvis
tags: [telegram, messaging, calls, social]
---

# Telegram (user account)

You act **as the account owner** on Telegram through the `telegram_*` tools.
Everything you send appears as written by them, so match their voice and
never invent facts on their behalf.

## Finding the right chat

- Chats accept `@username`, an `id` exactly as a tool returned it (e.g.
  `u8123456789` for users, `n…` for groups/channels), phone number, t.me link,
  an exact chat title, or `me` (Saved Messages). Prefer `@username`.
- Never strip the `u`/`n` prefix: bare 10-digit ids look like phone numbers
  and are redacted by the security filter before the tool sees them.
- If a name is ambiguous, call `telegram_list_chats` with `query` first and
  use the returned `id`.
- `telegram_get_entity` gives bio/description, member count, and whether a
  group voice chat is running.

## Messaging

- Read before replying: `telegram_read_messages` (newest first; page with
  `before_id`; omit `chat` and pass `search` to search everywhere).
- Send with `telegram_send_message` — Markdown text, `files` for media,
  `voice_note: true` for a single audio file, `reply_to`, `silent`,
  `schedule_minutes`.
- `telegram_message_action` reacts, pins/unpins, or marks a chat read.
- `telegram_forward_messages`, `telegram_edit_message`,
  `telegram_download_media` cover the rest.

## Groups and channels

`telegram_create_chat`, `telegram_join_chat` (usernames or invite links),
`telegram_invite_users` (also creates invite links), `telegram_edit_chat`,
`telegram_participants`.

## Actions that need the owner's confirmation

`telegram_update_profile`, `telegram_delete_messages`, `telegram_leave_chat`,
`telegram_moderate`, `telegram_block_user`, and `telegram_api_call` always
ask before running. State exactly what will change when you request them.

## Calls and voice chats

1. `telegram_call_start` with a user calls them privately; with a group it
   joins (or starts) the voice chat.
2. `mode: "conversation"` makes you listen and answer by voice in real
   time. Put the goal in `instructions` ("Ask Anna if dinner at 8 works and
   confirm the address") and an opening line in `greeting`.
3. By default the tool waits and returns the full transcript when the call
   ends. Under `jarvis telegram listen` it returns immediately instead; then
   use `telegram_call_control` (`status`, `transcript`, `mute`, `unmute`,
   `leave`) and `telegram_call_speak` (say text or play an audio file).
4. After a call, summarize the transcript for the owner.

## Music and video (local folders and YouTube)

- Streaming and managing the media folder never need the user's approval:
  do it right away, without asking "should I?".
- "Add this to the media folder" on a file (or replying to a file) →
  `telegram_media_library` `action: "add"` with the chat and message id given
  under "Files / referenced messages". "Remove X" → `action: "remove"`.
  Never use file_write/shell_exec for this.
- Local media folders (episodes, songs) are searched first, then YouTube.
  "What media do you have?" → `telegram_call_media` with `action: "list"`.
  If the user says "from YouTube" or "from the folder", pass `source`.

- "Play <song>" → `telegram_call_media` with `action: "play"`, the song as
  `query`, and the call/voice chat as `chat` (in a group, the group's id).
  It joins the call first if needed.
- Only set `video: true` when the user explicitly asks for video; otherwise
  stream audio only.
- Tell the user the title that is playing (from the tool result).
- Transport, from a private chat, a group mention, or by voice in the call:
  "pause" → `pause`; "play"/"resume" → `resume`; "forward 30 seconds" →
  `seek` with `by: 30`; "rewind 10" → `seek` with `by: -10`; "go to 1:23" →
  `seek` with `to: "1:23"`; "stop" → `stop` (you speak again afterwards).
- `status` reports the title and current position. Livestreams can't seek.

## Anything else

`telegram_api_call` invokes any raw MTProto method (privacy settings,
sessions, folders, stories, polls, …), e.g. method
`account.GetAuthorizationsRequest` or `account.SetPrivacyRequest`. Prefer the
dedicated tools whenever they fit.

## Good conduct

Never spam, mass-message, or add people to groups without the owner asking;
Telegram bans accounts for that. Respect flood-wait errors: if a tool returns
`FloodWaitError`, stop and report the wait time.
