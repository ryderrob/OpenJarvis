"""Tests for the Telegram user-account tools and voice-call plumbing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("telethon")


from telethon.tl import functions, types  # noqa: E402

from openjarvis.connectors import telegram_user as tu  # noqa: E402
from openjarvis.connectors import telegram_voice as tv  # noqa: E402
from openjarvis.security.data_boundary_audit import TELEGRAM_USER_TOOLS  # noqa: E402
from openjarvis.tools import telegram_user_tools as tt  # noqa: E402

ALICE = types.User(id=42, first_name="Alice", username="alice", access_hash=1)


class FakeClient:
    """Minimal async stand-in for ``telethon.TelegramClient``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def get_entity(self, ref: Any) -> Any:
        if ref in ("alice", "@alice", 42):
            return ALICE
        raise ValueError(ref)

    async def get_input_entity(self, entity: Any) -> Any:
        return types.InputPeerUser(user_id=entity.id, access_hash=1)

    async def get_me(self) -> Any:
        return ALICE

    def iter_dialogs(self, **_: Any) -> Any:
        async def _gen():
            if False:
                yield None

        return _gen()

    async def send_message(self, entity: Any, text: str, **kwargs: Any) -> Any:
        self.calls.append(("send_message", (entity, text, kwargs)))
        return SimpleNamespace(id=7)

    async def send_file(self, entity: Any, file: Any, **kwargs: Any) -> Any:
        self.calls.append(("send_file", (entity, file, kwargs)))
        return SimpleNamespace(id=8)

    async def __call__(self, request: Any) -> Any:
        self.calls.append(("request", request))
        return types.Updates(updates=[], users=[], chats=[], date=None, seq=0)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()

    class _Runner:
        def run(self, func: Any, *, timeout: float = 60.0) -> Any:
            return asyncio.run(func(client))

    monkeypatch.setattr(
        tu.TelegramUserClient, "get", classmethod(lambda cls: _Runner())
    )
    monkeypatch.setattr(
        tt.TelegramUserClient, "get", classmethod(lambda cls: _Runner())
    )
    return client


def test_tool_names_match_audit_classification() -> None:
    assert set(tt.TELEGRAM_USER_TOOLS) == TELEGRAM_USER_TOOLS


def test_destructive_tools_require_confirmation() -> None:
    confirmed = {
        cls.name
        for cls in tt._TelegramTool.__subclasses__()
        if cls().spec.requires_confirmation
    }
    assert confirmed == {
        "telegram_api_call",
        "telegram_block_user",
        "telegram_delete_messages",
        "telegram_leave_chat",
        "telegram_moderate",
        "telegram_update_profile",
    }


def test_specs_are_valid_function_schemas() -> None:
    for cls in tt._TelegramTool.__subclasses__():
        tool = cls()
        fn = tool.to_openai_function()["function"]
        assert fn["name"] == tool.tool_id == cls.name
        assert fn["parameters"]["type"] == "object"
        for required in fn["parameters"].get("required", []):
            assert required in fn["parameters"]["properties"]


def test_send_message_resolves_username(fake_client: FakeClient) -> None:
    result = tt.TelegramSendMessageTool().execute(chat="@alice", text="hi", silent=True)
    assert result.success, result.content
    name, (entity, text, kwargs) = fake_client.calls[0]
    assert (name, entity, text, kwargs["silent"]) == ("send_message", ALICE, "hi", True)
    assert json.loads(result.content)["message_ids"] == [7]


def test_send_message_rejects_missing_file(fake_client: FakeClient) -> None:
    result = tt.TelegramSendMessageTool().execute(chat="alice", files=["/nope/x.jpg"])
    assert not result.success
    assert "File not found" in result.content
    assert fake_client.calls == []


def test_send_message_voice_note(fake_client: FakeClient, tmp_path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"OggS")
    result = tt.TelegramSendMessageTool().execute(
        chat="alice", files=[str(audio)], voice_note=True
    )
    assert result.success, result.content
    name, (_, file, kwargs) = fake_client.calls[0]
    assert name == "send_file" and file == str(audio) and kwargs["voice_note"]


def test_unknown_chat_is_reported(fake_client: FakeClient) -> None:
    result = tt.TelegramSendMessageTool().execute(chat="@nobody", text="x")
    assert not result.success
    assert "Could not find" in result.content


def test_reaction_builds_send_reaction_request(fake_client: FakeClient) -> None:
    result = tt.TelegramMessageActionTool().execute(
        chat="alice", action="react", message_id=5, emoji="👍"
    )
    assert result.success, result.content
    request = fake_client.calls[0][1]
    assert isinstance(request, functions.messages.SendReactionRequest)
    assert request.msg_id == 5
    assert request.reaction[0].emoticon == "👍"


def test_api_call_resolves_method_peers_and_nested_types(
    fake_client: FakeClient,
) -> None:
    result = tt.TelegramApiCallTool().execute(
        method="contacts.BlockRequest", params={"id": "@alice"}
    )
    assert result.success, result.content
    request = fake_client.calls[0][1]
    assert isinstance(request, functions.contacts.BlockRequest)
    assert isinstance(request.id, types.InputPeerUser)

    nested = asyncio.run(
        tt._build_tl_value(
            fake_client,
            "contacts",
            [
                {
                    "_": "InputPhoneContact",
                    "client_id": 1,
                    "phone": "+1",
                    "first_name": "A",
                    "last_name": "",
                }
            ],
        )
    )
    assert isinstance(nested[0], types.InputPhoneContact)


def test_api_call_unknown_method(fake_client: FakeClient) -> None:
    result = tt.TelegramApiCallTool().execute(method="nope.MissingRequest")
    assert not result.success
    assert "Unknown Telegram method" in result.content


def test_missing_credentials_message(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.setattr(
        "openjarvis.core.credentials._default_path",
        lambda: tmp_path / "credentials.toml",
    )
    with pytest.raises(tu.TelegramUserError, match="jarvis telegram login"):
        tu.load_api_credentials()


def test_invite_hash_parsing() -> None:
    assert tu.invite_hash("https://t.me/+AbC_12") == "AbC_12"
    assert tu.invite_hash("https://t.me/joinchat/XyZ") == "XyZ"
    assert tu.invite_hash("https://t.me/somegroup") is None


# ---------------------------------------------------------------------------
# Voice session: VAD segmentation and barge-in (no Telegram needed)
# ---------------------------------------------------------------------------


def _frame(loud: bool, ssrc: int = 1) -> Any:
    import numpy as np

    level = 3000 if loud else 0
    pcm = (np.full(tv.FRAME_BYTES // 2, level, dtype=np.int16)).tobytes()
    return SimpleNamespace(ssrc=ssrc, frame=pcm)


def _session() -> tv.VoiceSession:
    return tv.VoiceSession(
        42,
        "Alice",
        conversational=True,
        instructions="",
        max_minutes=1,
        pipeline=None,
    )


def test_vad_emits_utterance_after_silence() -> None:
    async def _run() -> list[tuple[int, bytes]]:
        session = _session()
        session.on_frames([_frame(True)] * 50)  # 0.5 s speech
        assert session._utterances.empty()
        session.on_frames([_frame(False)] * 85)  # 0.85 s silence
        out = []
        while not session._utterances.empty():
            out.append(session._utterances.get_nowait())
        return out

    utterances = asyncio.run(_run())
    assert len(utterances) == 1
    ssrc, pcm = utterances[0]
    assert ssrc == 1
    assert len(pcm) >= 50 * tv.FRAME_BYTES


def test_vad_ignores_short_noise() -> None:
    async def _run() -> bool:
        session = _session()
        session.on_frames([_frame(True)] * 10 + [_frame(False)] * 100)
        return session._utterances.empty()

    assert asyncio.run(_run())


def test_barge_in_drops_queued_speech() -> None:
    async def _run() -> bool:
        session = _session()
        session.enqueue_pcm(b"\x01" * tv.FRAME_BYTES * 20)
        assert session.speaking
        session.on_frames([_frame(True)])
        return session.speaking

    assert asyncio.run(_run()) is False


def test_enqueue_pads_to_whole_frames() -> None:
    async def _run() -> list[bytes]:
        session = _session()
        session.enqueue_pcm(b"\x01" * (tv.FRAME_BYTES + 10))
        return [session._out.get_nowait() for _ in range(session._out.qsize())]

    chunks = asyncio.run(_run())
    assert [len(c) for c in chunks] == [tv.FRAME_BYTES, tv.FRAME_BYTES]


def test_pcm_to_wav_header() -> None:
    wav = tv.pcm_to_wav(b"\x00\x00" * 480)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_call_transcription_falls_back_to_local_whisper() -> None:
    pipeline = tv._SpeechPipeline.__new__(tv._SpeechPipeline)
    pipeline.config = SimpleNamespace(speech=SimpleNamespace(language=""))
    pipeline._lock = __import__("threading").Lock()

    class _Remote:
        def transcribe(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("gateway 502")

    class _Local:
        def transcribe(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace(text=" hello ")

    pipeline._stt = _Remote()
    pipeline._local_stt = _Local()
    assert pipeline.transcribe(b"\x00\x00" * 480) == "hello"


# ---------------------------------------------------------------------------
# Allow-listed responder and call gating
# ---------------------------------------------------------------------------

from openjarvis.connectors import telegram_responder as tr  # noqa: E402


class _Typing:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> None:
        return None


class _ChatClient:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    def iter_messages(self, *_: Any, **__: Any) -> Any:
        async def _gen():
            yield SimpleNamespace(message="earlier question", out=False)

        return _gen()

    def action(self, *_: Any) -> _Typing:
        return _Typing()

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


class _Event:
    def __init__(self, sender_id: int, client: _ChatClient) -> None:
        self.sender_id = sender_id
        self.chat_id = sender_id
        self.id = 99
        self.raw_text = "what's up?"
        self.client = client
        self.replies: list[str] = []
        self.read = False

    async def mark_read(self) -> None:
        self.read = True

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def _responder(answer: str = "all good") -> tr.TelegramResponder:
    responder = tr.TelegramResponder({7: "Crisis"})
    responder.answer = lambda prompt, chat_id, cancel_event=None: (  # type: ignore[method-assign]
        responder.prompts.append(prompt),  # type: ignore[attr-defined]
        answer,
    )[1]
    responder.prompts = []  # type: ignore[attr-defined]
    return responder


def test_responder_ignores_non_allowed_users() -> None:
    responder = _responder()
    event = _Event(sender_id=8, client=_ChatClient())
    asyncio.run(responder._on_message(event))
    assert event.replies == [] and not event.read
    assert responder.prompts == []  # type: ignore[attr-defined]


def test_responder_answers_allowed_user_with_history() -> None:
    responder = _responder()
    event = _Event(sender_id=7, client=_ChatClient())
    asyncio.run(responder._on_message(event))
    assert event.read and event.replies == ["all good"]
    prompt = responder.prompts[0]  # type: ignore[attr-defined]
    assert "Crisis: earlier question" in prompt and "what's up?" in prompt


def test_responder_splits_long_replies() -> None:
    client = _ChatClient()
    responder = _responder("word " * 2000)
    event = _Event(sender_id=7, client=client)
    asyncio.run(responder._on_message(event))
    assert len(event.replies) == 1 and len(client.sent) >= 1
    assert all(len(t) <= 4096 for t in event.replies + [s for _, s in client.sent])


def test_parse_allowed() -> None:
    assert tr.parse_allowed(" @a, 12 ,,") == ["@a", "12"]
    assert tr.parse_allowed("") == []


def test_incoming_call_from_stranger_is_not_answered() -> None:
    manager = tv.VoiceCallManager()
    manager.auto_answer = {"conversational": True}
    manager.allowed_callers = frozenset({7})
    joined: list[int] = []

    async def _join(chat_id: int, **_: Any) -> None:
        joined.append(chat_id)

    manager.join = _join  # type: ignore[method-assign]
    handlers: list[Any] = []

    class _Calls:
        def on_update(self, _filter: Any) -> Any:
            def _decorator(func: Any) -> Any:
                handlers.append(func)
                return func

            return _decorator

    asyncio.run(manager.install_handlers(_Calls()))
    incoming = handlers[-1]
    asyncio.run(incoming(None, SimpleNamespace(chat_id=8)))
    asyncio.run(incoming(None, SimpleNamespace(chat_id=7)))
    assert joined == [7]


# ---------------------------------------------------------------------------
# Peer references must survive the outbound PII (phone-number) filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("peer_id", [8123456789, -4123456789, -1001234567890, 42])
def test_peer_ref_round_trip(peer_id: int) -> None:
    assert tu.parse_peer_ref(tu.peer_ref(peer_id)) == peer_id


def test_peer_refs_are_not_redacted_by_boundary_scanner() -> None:
    from openjarvis.security.scanner import PIIScanner

    args = json.dumps(
        {"chat": tu.peer_ref(8123456789), "users": [tu.peer_ref(-4123456789)]}
    )
    assert PIIScanner().redact(args) == args
    # The raw id is what the guard mangles — the reason refs exist.
    assert "REDACTED" in PIIScanner().redact(json.dumps({"chat": "8123456789"}))


def test_entity_summary_uses_peer_ref() -> None:
    assert tt.entity_summary(ALICE)["id"] == "u42"


def test_redacted_reference_gives_actionable_error(fake_client: FakeClient) -> None:
    result = tt.TelegramSendMessageTool().execute(chat="[REDACTED:us_phone]", text="x")
    assert not result.success
    assert "@username" in result.content and fake_client.calls == []


def test_send_message_accepts_peer_ref(fake_client: FakeClient) -> None:
    result = tt.TelegramSendMessageTool().execute(chat="u42", text="hi")
    assert result.success, result.content
    assert fake_client.calls[0][1][0] is ALICE


# ---------------------------------------------------------------------------
# Approvals over Telegram
# ---------------------------------------------------------------------------

from openjarvis.connectors import telegram_agent as ta  # noqa: E402


def test_format_approval_request_is_readable() -> None:
    text = ta.format_approval_request(
        "Allow execution of tool 'shell_exec' with args {'command': 'ls ~'}?"
    )
    assert "Tool: shell_exec" in text and "'command': 'ls ~'" in text
    assert "yes" in text and "no" in text


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("yes", True), ("Yes please", True), ("no", False), ("deny", False)],
)
def test_broker_resolves_yes_no_and_stops_propagation(
    reply: str, expected: bool
) -> None:
    from telethon import events

    async def _run() -> bool:
        broker = ta.ApprovalBroker()
        future = asyncio.get_running_loop().create_future()
        broker._pending[7] = future
        event = SimpleNamespace(chat_id=7, raw_text=reply)
        with pytest.raises(events.StopPropagation):
            await broker._on_message(event)
        return future.result()

    assert asyncio.run(_run()) is expected


def test_broker_ignores_chats_without_pending_approval() -> None:
    async def _run() -> None:
        await ta.ApprovalBroker()._on_message(
            SimpleNamespace(chat_id=7, raw_text="yes")
        )

    asyncio.run(_run())  # no StopPropagation: message flows on to the agent


def test_agent_runner_excludes_call_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = ta.AgentRunner()
    config = SimpleNamespace(
        agent=SimpleNamespace(
            tools="telegram_me, telegram_call_start,get_weather,telegram_call_speak"
        )
    )
    runner._jarvis = SimpleNamespace(_config=config)
    assert runner.tools(("telegram_call_*",)) == ["telegram_me", "get_weather"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_telegram_log_file_is_private_and_records_tool_calls(
    fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import logging
    import stat

    from openjarvis.connectors import telegram_logging as tl

    monkeypatch.setattr(tl, "log_path", lambda: tmp_path / "logs" / "telegram.log")
    monkeypatch.setattr(tl, "_HANDLER", None)
    root = logging.getLogger("openjarvis")
    before = list(root.handlers)
    try:
        path = tl.ensure_telegram_logging()
        assert tl.ensure_telegram_logging() == path  # idempotent
        tt.TelegramSendMessageTool().execute(chat="@alice", text="hi")
        for handler in root.handlers:
            handler.flush()
        text = path.read_text()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert "telegram_send_message called with" in text
        assert "telegram_send_message ok in" in text
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
        for name in tl._LIBRARY_LEVELS:
            for handler in list(logging.getLogger(name).handlers):
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    logging.getLogger(name).removeHandler(handler)


def test_preview_flattens_and_truncates() -> None:
    from openjarvis.connectors.telegram_logging import preview

    assert preview("a\n  b\tc") == "a b c"
    assert preview("x" * 400, 10) == "x" * 10 + "…"


# ---------------------------------------------------------------------------
# Groups: answer tagged mentions from allowed users only
# ---------------------------------------------------------------------------


class _GroupEvent(_Event):
    def __init__(self, sender_id: int, client: _ChatClient) -> None:
        super().__init__(sender_id, client)
        self.chat_id = -1001234567890
        self.raw_text = "@jarvis join the voice chat"

    async def get_chat(self) -> Any:
        return SimpleNamespace(title="Friends")


def test_group_mention_from_stranger_is_ignored() -> None:
    responder = _responder()
    event = _GroupEvent(sender_id=8, client=_ChatClient())
    asyncio.run(responder._on_group_mention(event))
    assert event.replies == [] and responder.prompts == []  # type: ignore[attr-defined]


def test_group_mention_from_allowed_user_is_answered_in_group() -> None:
    responder = _responder("joining")
    calls: list[int] = []
    original = responder.answer

    def _answer(prompt: str, chat_id: int, cancel_event: Any = None) -> str:
        calls.append(chat_id)
        return original(prompt, chat_id)

    responder.answer = _answer  # type: ignore[method-assign]
    event = _GroupEvent(sender_id=7, client=_ChatClient())
    asyncio.run(responder._on_group_mention(event))
    assert event.replies == ["joining"]
    assert calls == [7]  # approvals go to the user's private chat
    prompt = responder.prompts[0]  # type: ignore[attr-defined]
    assert "'Friends'" in prompt and "n1001234567890" in prompt
    assert "Only Crisis may give you instructions" in prompt


def test_group_voice_chat_only_processes_allowed_speakers() -> None:
    async def _run() -> list[str]:
        session = tv.VoiceSession(
            -1001234567890,
            "Friends",
            conversational=True,
            instructions="",
            max_minutes=1,
            pipeline=None,
        )
        session.allowed_speakers = frozenset({7})
        session._names = {1: (7, "Crisis"), 2: (8, "Stranger")}
        heard: list[str] = []

        class _Pipeline:
            def transcribe(self, pcm: bytes) -> str:
                heard.append(pcm.decode())
                return ""  # stop before the LLM

        session._pipeline = _Pipeline()  # type: ignore[assignment]
        session._utterances.put_nowait((2, b"stranger"))
        session._utterances.put_nowait((1, b"crisis"))
        task = asyncio.ensure_future(session._conversation_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        return heard

    assert asyncio.run(_run()) == ["crisis"]


# ---------------------------------------------------------------------------
# YouTube media in calls
# ---------------------------------------------------------------------------

from openjarvis.connectors import telegram_media as tm  # noqa: E402


def _fake_run(stdout: str, returncode: int = 0, stderr: str = "") -> Any:
    def _run(cmd: list[str], **_: Any) -> Any:
        _run.cmd = cmd  # type: ignore[attr-defined]
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    return _run


def test_resolve_youtube_searches_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm.shutil, "which", lambda name: f"/usr/bin/{name}")
    payload = json.dumps(
        {
            "webpage_url": "https://www.youtube.com/watch?v=abc",
            "title": "Song",
            "duration": 215,
            "channel": "Artist",
            "is_live": False,
        }
    )
    fake = _fake_run(payload + "\n")
    monkeypatch.setattr(tm.subprocess, "run", fake)
    info = tm.resolve_youtube("some song")
    assert fake.cmd[-1] == "ytsearch1:some song"  # type: ignore[attr-defined]
    assert info.title == "Song" and info.duration_s == 215 and info.channel == "Artist"


def test_resolve_youtube_passes_links_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake = _fake_run(json.dumps({"webpage_url": "https://youtu.be/x", "title": "T"}))
    monkeypatch.setattr(tm.subprocess, "run", fake)
    tm.resolve_youtube("https://youtu.be/x")
    assert fake.cmd[-1] == "https://youtu.be/x"  # type: ignore[attr-defined]


def test_resolve_youtube_reports_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(tm.subprocess, "run", _fake_run("", 1, "ERROR: nothing"))
    with pytest.raises(tu.TelegramUserError, match="Nothing found"):
        tm.resolve_youtube("zzzz")


def test_resolve_youtube_requires_yt_dlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm.shutil, "which", lambda name: None)
    with pytest.raises(tu.TelegramUserError, match="yt-dlp"):
        tm.resolve_youtube("song")


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Jarvis stop", ("stop", None)),
        ("please pause", ("pause", None)),
        ("play", ("resume", None)),
        ("forward 30 seconds", ("seek_by", 30.0)),
        ("rewind ten seconds", ("seek_by", -10.0)),
        ("go to 1:23", ("seek_to", 83.0)),
        ("nice song", None),
    ],
)
def test_voice_commands_during_media(said: str, expected: Any) -> None:
    called: list[tuple[str, Any]] = []

    class _Manager:
        async def media_command(
            self, chat_id: int, action: str, seconds: Any, *, by: str = ""
        ) -> None:
            called.append((action, seconds))

    original = tv.VoiceCallManager.get
    tv.VoiceCallManager.get = classmethod(lambda cls: _Manager())  # type: ignore[assignment]
    try:
        session = _session()
        session.media = {"title": "Song"}
        asyncio.run(session._media_voice_command(said, "Crisis"))
    finally:
        tv.VoiceCallManager.get = original  # type: ignore[assignment]
    assert called == ([expected] if expected else [])


def test_sender_stays_quiet_while_media_plays() -> None:
    async def _run() -> list[bytes]:
        session = _session()
        session.media = {"title": "Song"}
        session.enqueue_pcm(b"\x01" * tv.FRAME_BYTES)
        sent: list[bytes] = []

        class _Calls:
            async def send_frame(self, chat_id: int, device: Any, data: bytes) -> None:
                sent.append(data)

        task = asyncio.ensure_future(session._sender(_Calls()))
        await asyncio.sleep(0.3)
        task.cancel()
        return sent

    assert asyncio.run(_run()) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("skip ahead a minute", ("seek_by", 60.0)),
        ("fast forward two minutes", ("seek_by", 120.0)),
        ("go back fifteen seconds", ("seek_by", -15.0)),
        ("rewind", ("seek_by", -tm.DEFAULT_STEP_S)),
        ("half a minute forward", ("seek_by", 30.0)),
        ("jump to 2 minutes", ("seek_to", 120.0)),
        ("start at 0:45", ("seek_to", 45.0)),
        ("I want to hear it 2 times", None),
    ],
)
def test_parse_media_command(text: str, expected: Any) -> None:
    assert tm.parse_media_command(text) == expected


def test_parse_timestamp_and_format() -> None:
    assert tm.parse_timestamp("1:02:03") == 3723
    assert tm.parse_timestamp("90") == 90
    assert tm.format_position(83) == "1:23"
    assert tm.format_position(3723) == "1:02:03"


def test_media_position_tracks_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(tv.time, "monotonic", lambda: now[0])
    media = {"offset": 10.0, "resumed_at": 100.0, "paused": False, "duration_s": 200}
    now[0] = 130.0
    assert tv.media_position(media) == 40.0
    media.update(paused=True, offset=40.0, resumed_at=None)
    now[0] = 500.0
    assert tv.media_position(media) == 40.0


class _MediaCalls:
    def __init__(self) -> None:
        self.played: list[Any] = []
        self.paused = 0

    async def play(self, chat_id: int, stream: Any) -> None:
        self.played.append(stream)

    async def pause(self, chat_id: int) -> None:
        self.paused += 1

    async def resume(self, chat_id: int) -> None:
        pass


def _manager_with_media(monkeypatch: pytest.MonkeyPatch, media: dict[str, Any]) -> Any:
    calls = _MediaCalls()

    class _Tg:
        async def calls(self) -> _MediaCalls:
            return calls

    monkeypatch.setattr(tv.TelegramUserClient, "get", classmethod(lambda cls: _Tg()))
    manager = tv.VoiceCallManager()
    session = _session()
    session.media = media
    manager.sessions[session.chat_id] = session
    return manager, calls


def _song(**overrides: Any) -> dict[str, Any]:
    media = {
        "title": "Song",
        "url": "https://youtu.be/x",
        "video": False,
        "paused": False,
        "duration_s": 200,
        "live": False,
        "offset": 50.0,
        "resumed_at": None,
        "ignore_end_until": 0,
    }
    media.update(overrides)
    return media


def test_seek_restarts_stream_at_clamped_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, calls = _manager_with_media(monkeypatch, _song())
    status = asyncio.run(manager.seek_media(42, by=30))
    assert status["position"] == "1:20" and status["paused"] is False
    assert "-ss 80.0" in str(calls.played[-1]._ffmpeg_parameters)
    status = asyncio.run(manager.seek_media(42, by=-500))
    assert status["position"] == "0:00"
    status = asyncio.run(manager.seek_media(42, to=999))
    assert status["position"] == "3:18"  # clamped just before the end


def test_seek_keeps_paused_state(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, calls = _manager_with_media(monkeypatch, _song(paused=True))
    status = asyncio.run(manager.seek_media(42, to=10))
    assert status["paused"] is True and calls.paused == 1


def test_livestream_cannot_seek(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager_with_media(monkeypatch, _song(live=True))
    with pytest.raises(tu.TelegramUserError, match="livestream"):
        asyncio.run(manager.seek_media(42, by=10))


def test_media_tool_seek_and_play_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[str, Any]] = []

    class _Manager:
        persistent = True
        sessions: dict[int, Any] = {}

        async def seek_media(
            self, chat_id: int, *, to: Any = None, by: Any = None
        ) -> Any:
            received.append(("seek", to if to is not None else by))
            return {"ok": True}

        async def resume_media(self, chat_id: int) -> Any:
            received.append(("resume", None))
            return {"ok": True}

    class _Tg:
        def submit(self, coro: Any) -> Any:
            result = asyncio.run(coro)
            return SimpleNamespace(result=lambda timeout=None: result)

    async def _resolve(chat: Any) -> int:
        return 42

    monkeypatch.setattr(tv.VoiceCallManager, "get", classmethod(lambda cls: _Manager()))
    monkeypatch.setattr(tv, "resolve_chat_id", _resolve)
    monkeypatch.setattr(tt.TelegramUserClient, "get", classmethod(lambda cls: _Tg()))
    tool = tt.TelegramCallMediaTool()
    assert tool.execute(action="seek", chat="u42", by=-10).success
    assert tool.execute(action="seek", chat="u42", to="1:23").success
    assert tool.execute(action="play", chat="u42").success
    assert received == [("seek", -10.0), ("seek", 83.0), ("resume", None)]


# ---------------------------------------------------------------------------
# Local media folders
# ---------------------------------------------------------------------------


@pytest.fixture
def media_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "media"
    (root / "music").mkdir(parents=True)
    for name in (
        "[01] seven deadly sins.mp4",
        "[03] seven deadly sins.mp4",
        "music/The Quiet Hand.mp3",
        "notes.txt",
    ):
        (root / name).write_bytes(b"x")
    monkeypatch.setattr(tm, "probe_duration", lambda path: 120)
    return root


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("seven deadly sins episode 3", "[03] seven deadly sins"),
        ("play the quiet hand", "The Quiet Hand"),
        ("deadly sins 1", "[01] seven deadly sins"),
        ("bohemian rhapsody", None),
    ],
)
def test_find_local_media(media_dir: Any, query: str, expected: Any) -> None:
    info = tm.find_local_media(query, [str(media_dir)])
    assert (info.title if info else None) == expected
    if info:
        assert info.channel == "local" and info.duration_s == 120


def test_local_media_ignores_non_media_and_lists(media_dir: Any) -> None:
    files = tm.describe_local_media([str(media_dir)])
    assert [f["title"] for f in files] == [
        "[01] seven deadly sins",
        "[03] seven deadly sins",
        "The Quiet Hand",
    ]
    assert {f["type"] for f in files} == {"video", "audio"}


def test_local_media_refuses_paths_outside_folders(media_dir: Any, tmp_path) -> None:
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(b"x")
    with pytest.raises(tu.TelegramUserError, match="not inside"):
        tm.local_media_info(outside, [str(media_dir)])


def test_resolve_media_prefers_local_then_youtube(
    media_dir: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tt, "_media_dirs", lambda: [str(media_dir)])
    youtube = tm.MediaInfo("https://youtu.be/x", "YT", 10, "c", False)
    monkeypatch.setattr(tm, "resolve_youtube", lambda query: youtube)
    assert tt._resolve_media("seven deadly sins 3", "auto").channel == "local"
    assert tt._resolve_media("bohemian rhapsody", "auto") is youtube
    assert tt._resolve_media("seven deadly sins 3", "youtube") is youtube
    with pytest.raises(tu.TelegramUserError, match="No local media"):
        tt._resolve_media("bohemian rhapsody", "local")


# ---------------------------------------------------------------------------
# Attachments in prompts and the media library tool
# ---------------------------------------------------------------------------


def _file_message(msg_id: int, name: str, size: int = 5_242_880, **kind: Any) -> Any:
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return SimpleNamespace(
        id=msg_id,
        message="",
        media=object(),
        file=SimpleNamespace(name=name, size=size, ext=ext, title=None),
        **kind,
    )


def test_describe_attachments_includes_own_and_replied_file() -> None:
    reply = _file_message(41, "Song.mp3", audio=True)

    class _Ev:
        chat_id = -1001234567890
        id = 42
        is_reply = True
        message = _file_message(42, "clip.mp4", video=True)

        async def get_reply_message(self) -> Any:
            return reply

    text = asyncio.run(tr.describe_attachments(_Ev()))
    assert "video 'clip.mp4', 5.0 MB" in text and "message id 42" in text
    assert "replies to message id 41" in text and "audio 'Song.mp3'" in text
    assert "telegram_media_library" in text


def test_describe_attachments_empty_without_files() -> None:
    class _Ev:
        chat_id = 7
        id = 1
        is_reply = False
        message = SimpleNamespace(media=None, file=None)

    assert asyncio.run(tr.describe_attachments(_Ev())) == ""


def test_media_tools_never_require_confirmation() -> None:
    for tool in (tt.TelegramCallMediaTool(), tt.TelegramMediaLibraryTool()):
        assert tool.spec.requires_confirmation is False


@pytest.fixture
def library(media_dir: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(tt, "_media_dirs", lambda: [str(media_dir)])
    return media_dir


def test_library_add_downloads_into_media_folder(
    library: Any, fake_client: FakeClient
) -> None:
    async def get_messages(entity: Any, ids: int) -> Any:
        return _file_message(ids, "New Song.mp3", audio=True)

    async def download_media(message: Any, file: str) -> str:
        Path(file).write_bytes(b"audio")
        return file

    fake_client.get_messages = get_messages  # type: ignore[attr-defined]
    fake_client.download_media = download_media  # type: ignore[attr-defined]
    tool = tt.TelegramMediaLibraryTool()
    result = tool.execute(action="add", chat="alice", message_id=5, name="../../evil")
    assert result.success, result.content
    assert json.loads(result.content)["added"] == "evil.mp3"
    assert (library / "evil.mp3").exists()
    again = json.loads(tool.execute(action="add", chat="alice", message_id=5).content)
    assert again["added"] == "New Song.mp3"


def test_library_add_rejects_non_media_and_foreign_folders(
    library: Any, fake_client: FakeClient, tmp_path: Any
) -> None:
    async def get_messages(entity: Any, ids: int) -> Any:
        return _file_message(ids, "report.pdf")

    fake_client.get_messages = get_messages  # type: ignore[attr-defined]
    tool = tt.TelegramMediaLibraryTool()
    bad = tool.execute(action="add", chat="alice", message_id=5)
    assert not bad.success and "audio/video" in bad.content
    outside = tool.execute(
        action="add", chat="alice", message_id=5, folder=str(tmp_path)
    )
    assert not outside.success and "not a configured media folder" in outside.content


def test_library_remove_exact_and_refuses_ambiguous(library: Any) -> None:
    tool = tt.TelegramMediaLibraryTool()
    ambiguous = tool.execute(action="remove", query="seven deadly sins")
    assert not ambiguous.success  # two episodes match equally
    done = tool.execute(action="remove", query="the quiet hand")
    assert done.success and json.loads(done.content)["removed"] == "The Quiet Hand.mp3"
    assert not (library / "music" / "The Quiet Hand.mp3").exists()


def test_safe_media_stem() -> None:
    assert tm.safe_media_stem("../../etc/passwd") == "passwd"
    assert tm.safe_media_stem('a:b*c?"d') == "a_b_c__d"


# ---------------------------------------------------------------------------
# Stop / cancel
# ---------------------------------------------------------------------------

from openjarvis.connectors import telegram_tasks as tk  # noqa: E402
from openjarvis.core import cancellation  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stop", True),
        ("STOP!", True),
        ("/stop", True),
        ("cancel", True),
        ("Jarvis, stop", True),
        ("@Pray1ngforyou stop it", True),
        ("abort", True),
        ("never mind", True),
        ("stop everything please", True),
        ("stop the music and play jazz", False),
        ("don't stop", False),
        ("what time is it", False),
    ],
)
def test_is_stop_command(text: str, expected: bool) -> None:
    assert tk.is_stop_command(text) is expected


def test_cancelled_run_refuses_further_tool_calls() -> None:
    from openjarvis.core.types import ToolCall
    from openjarvis.tools._stubs import ToolExecutor

    event = __import__("threading").Event()
    executor = ToolExecutor([tt.TelegramMeTool()])
    cancellation.bind(event)
    try:
        event.set()
        result = executor.execute(ToolCall(id="1", name="telegram_me", arguments="{}"))
    finally:
        cancellation.bind(None)
    assert not result.success and result.content == cancellation.CANCELLED_MESSAGE


def test_registry_cancels_handlers_and_flags_runs() -> None:
    async def _run() -> tuple[bool, bool, list[str]]:
        registry = tk.TaskRegistry()
        started = asyncio.Event()
        holder: dict[str, Any] = {}

        async def _handler() -> None:
            holder["run"] = registry.start(7, "long task")
            started.set()
            await asyncio.sleep(60)

        task = asyncio.ensure_future(_handler())
        await started.wait()
        names = registry.cancel_runs()
        with pytest.raises(asyncio.CancelledError):
            await task
        return holder["run"].event.is_set(), task.cancelled(), names

    flagged, cancelled, names = asyncio.run(_run())
    assert flagged and cancelled and names == ["long task"]


def test_broker_deny_all_resolves_pending() -> None:
    async def _run() -> tuple[int, bool]:
        broker = ta.ApprovalBroker()
        future = asyncio.get_running_loop().create_future()
        broker._pending[7] = future
        return broker.deny_all(), future.result()

    assert asyncio.run(_run()) == (1, False)


def test_stop_message_drops_running_task_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    gate = threading.Event()
    stops: list[str] = []

    async def _stop(reason: str) -> dict[str, Any]:
        stops.append(reason)
        names = tk.TaskRegistry.get().cancel_runs(exclude=asyncio.current_task())
        return {
            "tasks": names,
            "approvals_denied": 0,
            "operations_cancelled": 0,
            "media_stopped": [],
        }

    monkeypatch.setattr(tr, "stop_everything", _stop)
    monkeypatch.setattr(tk.TaskRegistry, "_instance", tk.TaskRegistry())

    responder = tr.TelegramResponder({7: "Crisis"})

    def _slow_answer(prompt: str, chat_id: int, cancel_event: Any = None) -> str:
        gate.wait(5)
        return "late answer"

    responder.answer = _slow_answer  # type: ignore[method-assign]

    async def _run() -> tuple[list[str], list[str]]:
        client = _ChatClient()
        busy = _Event(sender_id=7, client=client)
        busy.raw_text = "download everything"
        stop = _Event(sender_id=7, client=client)
        stop.raw_text = "stop"
        task = asyncio.ensure_future(responder._on_message(busy))
        await asyncio.sleep(0.2)
        await responder._on_message(stop)
        await asyncio.sleep(0.1)
        gate.set()
        await asyncio.gather(task, return_exceptions=True)
        return busy.replies, stop.replies

    busy_replies, stop_replies = asyncio.run(_run())
    assert busy_replies == []  # the late answer is never sent
    assert stop_replies and stop_replies[0].startswith("🛑 Stopped 1 task(s)")
    assert stops == ["requested by Crisis"]


def test_describe_stop_nothing_running() -> None:
    empty = {
        "tasks": [],
        "approvals_denied": 0,
        "operations_cancelled": 0,
        "media_stopped": [],
    }
    assert tk.describe_stop(empty) == "Nothing was running."


def test_cancel_inflight_cancels_pending_futures() -> None:
    from concurrent.futures import Future

    client = tu.TelegramUserClient.__new__(tu.TelegramUserClient)
    client._inflight = set()
    client._inflight_lock = __import__("threading").Lock()
    pending, done = Future(), Future()
    done.set_result(1)
    client._track(pending)
    client._track(done)
    assert client.cancel_inflight() == 1 and pending.cancelled()
