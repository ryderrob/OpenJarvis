"""Real-time voice for Telegram private calls and group voice chats.

A :class:`VoiceSession` joins a call through py-tgcalls with *external* audio
in both directions:

* incoming PCM frames (48 kHz mono s16le, one stream per speaker ``ssrc``) are
  cut into utterances with a simple energy VAD, transcribed with the
  configured speech backend, answered by the configured LLM, and
* the reply is synthesized with the configured TTS backend, decoded to PCM by
  ffmpeg, and paced out as 10 ms frames.

The same outgoing queue also plays arbitrary audio files and ad-hoc text, so
``mode="silent"`` sessions work as a remote-controlled speaker. When a
participant starts talking over the agent, the queued speech is dropped
(barge-in).
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openjarvis.connectors.telegram_logging import preview
from openjarvis.connectors.telegram_media import format_position
from openjarvis.connectors.telegram_user import (
    TelegramUserClient,
    TelegramUserError,
    peer_ref,
    resolve_entity,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_MS = 10
FRAME_BYTES = SAMPLE_RATE * CHANNELS * 2 * FRAME_MS // 1000  # 960 bytes

# Energy VAD tuning (int16 RMS). Telegram applies AGC, so a fixed floor works.
SPEECH_RMS = 500
END_SILENCE_S = 0.8
MIN_SPEECH_S = 0.35
MAX_UTTERANCE_S = 30.0
END_CALL_TOKEN = "[END_CALL]"

_DEFAULT_PERSONA = (
    "You are OpenJarvis, speaking out loud on a live Telegram voice call on "
    "behalf of the account owner. Replies are converted to speech: answer in "
    "one to three short, natural spoken sentences, with no markdown, lists, "
    "emojis, or URLs. If the conversation is finished or you are asked to "
    f"hang up, say a brief goodbye and end your reply with {END_CALL_TOKEN}."
)


def _rms(pcm: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def decode_to_pcm(*, data: bytes | None = None, path: str | None = None) -> bytes:
    """Decode any audio (bytes or file) to 48 kHz mono s16le with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise TelegramUserError("ffmpeg is required for call audio but is not on PATH.")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path or "pipe:0"]
    cmd += ["-f", "s16le", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "pipe:1"]
    proc = subprocess.run(
        cmd, input=data, capture_output=True, timeout=300, check=False
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()[:300]
        raise TelegramUserError(f"ffmpeg could not decode audio: {detail}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Speech pipeline (STT -> LLM -> TTS), all blocking; run in executor threads
# ---------------------------------------------------------------------------


class _SpeechPipeline:
    """Lazily-resolved STT, LLM, and TTS backends from the user's config."""

    def __init__(self) -> None:
        from openjarvis.core.config import load_config

        self.config = load_config()
        self._stt: Any = None
        self._local_stt: Any = None
        self._tts: Any = None
        self._engine: Any = None
        self._model = self.config.intelligence.default_model
        self._lock = threading.Lock()

    def transcribe(self, pcm: bytes) -> str:
        with self._lock:
            if self._stt is None:
                from openjarvis.speech._discovery import get_speech_backend

                self._stt = get_speech_backend(self.config)
                if self._stt is None:
                    raise TelegramUserError(
                        "No speech-to-text backend available. Install one with "
                        "`uv sync --extra speech` or set DEEPGRAM_API_KEY or "
                        "OPENAI_API_KEY."
                    )
        language = self.config.speech.language or None
        wav = pcm_to_wav(pcm)
        try:
            result = self._stt.transcribe(wav, format="wav", language=language)
        except Exception as exc:
            # A remote STT gateway can fail transiently or be misconfigured;
            # local Whisper keeps the call usable rather than going deaf.
            local = self._local_whisper()
            if local is None or local is self._stt:
                raise
            logger.warning("Speech-to-text failed (%s); using local Whisper", exc)
            result = local.transcribe(wav, format="wav", language=language)
        return (result.text or "").strip()

    def _local_whisper(self) -> Any:
        with self._lock:
            if self._local_stt is None:
                from openjarvis.speech._discovery import _create_backend

                self._local_stt = (
                    _create_backend("faster-whisper", self.config) or False
                )
        return self._local_stt or None

    def reply(
        self,
        history: list[dict[str, str]],
        system_prompt: str,
        confirm: Optional[Callable[[str], bool]] = None,
    ) -> str:
        if self.config.agent.tools.strip():
            return self._agent_reply(history, system_prompt, confirm)
        from openjarvis.core.types import Message, Role

        with self._lock:
            if self._engine is None:
                from openjarvis.engine._discovery import get_engine

                found = get_engine(self.config)
                if found is None:
                    raise TelegramUserError("No inference engine is reachable.")
                self._engine = found[1]
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]
        for turn in history[-20:]:
            role = Role.ASSISTANT if turn["role"] == "assistant" else Role.USER
            messages.append(Message(role=role, content=turn["content"]))
        result = self._engine.generate(
            messages, model=self._model, temperature=0.6, max_tokens=200
        )
        return str(result.get("content", "")).strip()

    def _agent_reply(
        self,
        history: list[dict[str, str]],
        system_prompt: str,
        confirm: Optional[Callable[[str], bool]],
    ) -> str:
        """Answer with the full agent so tools (weather, files, ...) work."""
        from openjarvis.connectors.telegram_agent import AgentRunner

        transcript = "\n".join(
            f"{'You' if t['role'] == 'assistant' else t.get('speaker', 'Caller')}: "
            f"{t['content']}"
            for t in history[-20:]
        )
        prompt = (
            f"{system_prompt}\n\nYou can use your tools (weather, web search, "
            "files, commands, Telegram) to answer. Reply with only the words to "
            f"say next.\n\nCall transcript so far:\n{transcript}"
        )
        from openjarvis.connectors.telegram_tasks import TaskRegistry

        # Registered without a handler task (this runs in an executor thread),
        # so a "stop" flags the run but leaves the call's own loop alone.
        last = history[-1]["content"] if history else "call reply"
        run = TaskRegistry.get().start(0, f"call: {last}")
        try:
            return AgentRunner.get(self.config).run(
                prompt,
                confirm=confirm,
                exclude_tools=("telegram_call_*",),
                cancel_event=run.event,
            )
        finally:
            TaskRegistry.get().finish(run)

    def synthesize(self, text: str) -> bytes:
        """Return 48 kHz mono PCM for *text*."""
        from openjarvis.speech._tts_discovery import (
            default_voice_for,
            get_tts_backend,
            voice_preferences,
        )

        with self._lock:
            if self._tts is None:
                preferred, _, _ = voice_preferences(self.config)
                self._tts = get_tts_backend(preferred) or False
        if self._tts:
            preferred, voice_id, speed = voice_preferences(self.config)
            if self._tts.backend_id != preferred or not voice_id:
                voice_id = default_voice_for(self._tts.backend_id)
            result = self._tts.synthesize(
                text, voice_id=voice_id, speed=speed, output_format="wav"
            )
            return decode_to_pcm(data=result.audio)
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak is None:
            raise TelegramUserError(
                "No text-to-speech backend available. Install Kokoro "
                "(`uv sync --extra voice`), set OPENAI_API_KEY/CARTESIA_API_KEY, "
                "or install espeak-ng."
            )
        wav = subprocess.run(
            [espeak, "--stdout", text], capture_output=True, timeout=60, check=True
        ).stdout
        return decode_to_pcm(data=wav)


# ---------------------------------------------------------------------------
# Voice session
# ---------------------------------------------------------------------------


@dataclass
class _Speaker:
    buffer: bytearray = field(default_factory=bytearray)
    speech_s: float = 0.0
    silence_s: float = 0.0
    talking: bool = False


class VoiceSession:
    """One joined call (private or group) with an optional conversation loop."""

    def __init__(
        self,
        chat_id: int,
        title: str,
        *,
        conversational: bool,
        instructions: str,
        max_minutes: float,
        pipeline: Optional[_SpeechPipeline],
    ) -> None:
        self.chat_id = chat_id
        self.title = title
        self.conversational = conversational
        self.instructions = instructions.strip()
        self.deadline = time.monotonic() + max(0.5, max_minutes) * 60
        self.started_at = time.time()
        self.transcript: list[dict[str, str]] = []
        self.ended = threading.Event()
        self.end_reason = ""
        self.muted = False
        self._pipeline = pipeline
        self._out: asyncio.Queue[bytes] = asyncio.Queue()
        self._speakers: dict[int, _Speaker] = {}
        self._names: dict[int, tuple[Optional[int], str]] = {}
        # Group voice chats: only these user ids are listened to (None = all).
        self.allowed_speakers: Optional[frozenset[int]] = None
        # Set while a YouTube track replaces the assistant's voice.
        self.media: Optional[dict[str, Any]] = None
        self._utterances: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._hang_up_after_speech = False

    @property
    def system_prompt(self) -> str:
        prompt = _DEFAULT_PERSONA
        if self.chat_id < 0:
            prompt += (
                "\n\nThis is a group voice chat. You only hear the people allowed "
                "to instruct you; answer them and follow their instructions."
            )
        if self.instructions:
            prompt += f"\n\nYour task on this call: {self.instructions}"
        return prompt

    def snapshot(self) -> dict[str, Any]:
        return {
            "chat": peer_ref(self.chat_id),
            "title": self.title,
            "mode": "conversation" if self.conversational else "silent",
            "active": not self.ended.is_set(),
            "muted": self.muted,
            "duration_s": round(time.time() - self.started_at, 1),
            "end_reason": self.end_reason or None,
            "media": media_status(self.media) if self.media else None,
            "transcript": self.transcript,
        }

    # -- outgoing audio -------------------------------------------------

    def enqueue_pcm(self, pcm: bytes) -> None:
        for i in range(0, len(pcm), FRAME_BYTES):
            chunk = pcm[i : i + FRAME_BYTES]
            if len(chunk) < FRAME_BYTES:
                chunk += b"\x00" * (FRAME_BYTES - len(chunk))
            self._out.put_nowait(chunk)

    def drop_queued_audio(self) -> None:
        while not self._out.empty():
            self._out.get_nowait()

    @property
    def speaking(self) -> bool:
        return not self._out.empty()

    async def _sender(self, calls: Any) -> None:
        from pytgcalls.types import Device

        silence = b"\x00" * FRAME_BYTES
        next_at = time.monotonic()
        while not self.ended.is_set():
            if self.media is not None:
                # The media stream owns the outgoing audio; external frames
                # would be rejected. Keep watching the time limit.
                await asyncio.sleep(0.2)
                next_at = time.monotonic()
                if time.monotonic() > self.deadline:
                    await VoiceCallManager.get().leave(
                        self.chat_id, "time limit reached"
                    )
                    return
                continue
            try:
                chunk = self._out.get_nowait()
            except asyncio.QueueEmpty:
                chunk = silence
                if self._hang_up_after_speech:
                    await VoiceCallManager.get().leave(self.chat_id, "agent ended call")
                    return
            try:
                await calls.send_frame(self.chat_id, Device.MICROPHONE, chunk)
            except Exception:
                logger.debug("send_frame failed", exc_info=True)
            next_at += FRAME_MS / 1000
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            if time.monotonic() > self.deadline:
                await VoiceCallManager.get().leave(self.chat_id, "time limit reached")
                return

    async def say(self, text: str) -> None:
        if self._pipeline is None:
            self._pipeline = _SpeechPipeline()
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, self._pipeline.synthesize, text)
        self.enqueue_pcm(pcm)

    # -- incoming audio -------------------------------------------------

    def on_frames(self, frames: list[Any]) -> None:
        if not self.conversational or self.ended.is_set():
            return
        for frame in frames:
            pcm = frame.frame
            if not pcm:
                continue
            speaker = self._speakers.setdefault(frame.ssrc & 0xFFFFFFFF, _Speaker())
            seconds = len(pcm) / (SAMPLE_RATE * CHANNELS * 2)
            loud = _rms(pcm) >= SPEECH_RMS
            if loud:
                if not speaker.talking and self.speaking:
                    self.drop_queued_audio()  # barge-in
                speaker.talking = True
                speaker.speech_s += seconds
                speaker.silence_s = 0.0
            elif speaker.talking:
                speaker.silence_s += seconds
            if speaker.talking:
                speaker.buffer += pcm
            too_long = speaker.speech_s >= MAX_UTTERANCE_S
            if speaker.talking and (speaker.silence_s >= END_SILENCE_S or too_long):
                if speaker.speech_s >= MIN_SPEECH_S:
                    self._utterances.put_nowait(
                        (frame.ssrc & 0xFFFFFFFF, bytes(speaker.buffer))
                    )
                self._speakers[frame.ssrc & 0xFFFFFFFF] = _Speaker()

    async def _speaker(self, ssrc: int) -> tuple[Optional[int], str]:
        """Map an audio source to ``(user_id, display name)``."""
        if ssrc in self._names:
            return self._names[ssrc]
        user_id: Optional[int] = self.chat_id if self.chat_id > 0 else None
        name = self.title or "Caller"
        if self.chat_id < 0:
            name = "Participant"
            try:
                tg = TelegramUserClient.get()
                calls = await tg.calls()
                client = await tg.client()
                for participant in await calls.get_participants(self.chat_id) or []:
                    if participant.source & 0xFFFFFFFF == ssrc:
                        from telethon import utils

                        user_id = participant.user_id
                        user = await client.get_entity(user_id)
                        name = utils.get_display_name(user) or name
                        break
            except Exception:
                logger.debug("participant lookup failed", exc_info=True)
        self._names[ssrc] = (user_id, name)
        return user_id, name

    async def _conversation_loop(self) -> None:
        assert self._pipeline is not None
        loop = asyncio.get_running_loop()
        while not self.ended.is_set():
            ssrc, pcm = await self._utterances.get()
            user_id, speaker = await self._speaker(ssrc)
            if (
                self.allowed_speakers is not None
                and user_id not in self.allowed_speakers
            ):
                logger.debug("Call %s: ignoring %s (not allowed)", self.title, speaker)
                continue
            try:
                text = await loop.run_in_executor(None, self._pipeline.transcribe, pcm)
            except Exception as exc:
                logger.warning("Call transcription failed: %s", exc)
                continue
            if not text:
                continue
            if self.media is not None:
                await self._media_voice_command(text, speaker)
                continue
            from openjarvis.connectors.telegram_tasks import (
                is_stop_command,
                stop_everything,
            )

            if is_stop_command(text):
                summary = await stop_everything(f"said by {speaker} on a call")
                logger.info("Call %s: stop by voice: %s", self.title, summary)
                await self.say("Stopped.")
                continue
            content = f"{speaker}: {text}" if self.chat_id < 0 else text
            logger.info("Call %s heard %s: %s", self.title, speaker, preview(text))
            self.transcript.append(
                {"role": "user", "speaker": speaker, "content": content}
            )
            try:
                reply = await loop.run_in_executor(
                    None,
                    self._pipeline.reply,
                    self.transcript,
                    self.system_prompt,
                    self._approval_callback(),
                )
            except Exception as exc:
                logger.warning("Call reply generation failed: %s", exc)
                continue
            from openjarvis.core.cancellation import CANCELLED_MESSAGE

            if reply == CANCELLED_MESSAGE:
                continue  # stopped meanwhile; don't speak a stale answer
            hang_up = END_CALL_TOKEN in reply
            reply = reply.replace(END_CALL_TOKEN, "").strip()
            logger.info(
                "Call %s reply%s: %s",
                self.title,
                " (hanging up)" if hang_up else "",
                preview(reply),
            )
            if reply:
                self.transcript.append(
                    {"role": "assistant", "speaker": "assistant", "content": reply}
                )
                try:
                    await self.say(reply)
                except Exception as exc:
                    logger.warning("Call speech synthesis failed: %s", exc)
            if hang_up:
                self._hang_up_after_speech = True

    async def _media_voice_command(self, text: str, speaker: str) -> None:
        """While music plays only transport commands are acted on by voice."""
        from openjarvis.connectors.telegram_media import parse_media_command

        command = parse_media_command(text)
        if command is None:
            logger.debug(
                "Call %s: ignoring %s during media: %s", self.title, speaker, text
            )
            return
        action, seconds = command
        logger.info(
            "Call %s: voice command %s%s from %s",
            self.title,
            action,
            f" {seconds:+.0f}s"
            if action == "seek_by"
            else f" {seconds:.0f}s"
            if action == "seek_to"
            else "",
            speaker,
        )
        try:
            await VoiceCallManager.get().media_command(
                self.chat_id, action, seconds, by=speaker
            )
        except Exception:
            logger.exception("Media voice command failed")

    def _approval_callback(self) -> Optional[Callable[[str], bool]]:
        """Approve tools via the caller's Telegram chat (private calls only)."""
        if self.chat_id <= 0:
            return None
        from openjarvis.connectors.telegram_agent import ApprovalBroker

        ask = ApprovalBroker.get().callback_for(self.chat_id)
        tg = TelegramUserClient.get()

        def _confirm(prompt: str) -> bool:
            try:
                tg.submit(
                    self.say(
                        "I need your approval for that. I sent you the details "
                        "on Telegram; reply yes or no there."
                    )
                ).result(60)
            except Exception:
                logger.debug("Could not announce approval request", exc_info=True)
            return ask(prompt)

        return _confirm

    def start_tasks(self, calls: Any) -> None:
        self._tasks.append(asyncio.ensure_future(self._sender(calls)))
        if self.conversational:
            self._tasks.append(asyncio.ensure_future(self._conversation_loop()))

    def finish(self, reason: str) -> None:
        if self.ended.is_set():
            return
        logger.info(
            "Call with %s ended after %.0fs: %s (%d transcript turns)",
            self.title,
            time.time() - self.started_at,
            reason,
            len(self.transcript),
        )
        self.end_reason = reason
        self.ended.set()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()


_RESTART_GRACE_S = 5.0


def media_position(media: Optional[dict[str, Any]]) -> Optional[float]:
    """Seconds into the current track (None when unknown)."""
    if not media:
        return None
    position = float(media.get("offset") or 0.0)
    if media.get("resumed_at") is not None and not media.get("paused"):
        position += time.monotonic() - media["resumed_at"]
    duration = media.get("duration_s")
    return min(position, float(duration)) if duration else position


def media_status(media: dict[str, Any]) -> dict[str, Any]:
    position = media_position(media)
    return {
        "title": media["title"],
        "video": media["video"],
        "paused": media["paused"],
        "position": format_position(position),
        "duration": format_position(media.get("duration_s")),
        "live": media.get("live", False),
    }


class VoiceCallManager:
    """Tracks live sessions and routes py-tgcalls updates to them."""

    _instance: Optional["VoiceCallManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.sessions: dict[int, VoiceSession] = {}
        self._handlers_installed = False
        self.auto_answer: Optional[dict[str, Any]] = None
        # User ids whose incoming calls may be auto-answered; others ring out.
        self.allowed_callers: frozenset[int] = frozenset()
        # Who may speak to the assistant in group voice chats; None = resolve
        # from [tools.telegram_user].allowed_users on first group join.
        self.allowed_speakers: Optional[frozenset[int]] = None
        self.history: list[dict[str, Any]] = []
        # True under `jarvis telegram listen`: calls and media may outlive a
        # single tool call, so tools return immediately instead of waiting.
        self.persistent = False

    @classmethod
    def get(cls) -> "VoiceCallManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def install_handlers(self, calls: Any) -> None:
        if self._handlers_installed:
            return
        from pytgcalls import filters
        from pytgcalls.types import ChatUpdate, Device, Direction

        @calls.on_update(
            filters.stream_frame(Direction.INCOMING, Device.MICROPHONE | Device.SPEAKER)
        )
        async def _frames(_: Any, update: Any) -> None:
            session = self.sessions.get(update.chat_id)
            if session is not None:
                session.on_frames(update.frames)

        @calls.on_update(filters.chat_update(ChatUpdate.Status.LEFT_CALL))
        async def _left(_: Any, update: Any) -> None:
            session = self.sessions.pop(update.chat_id, None)
            if session is not None:
                session.finish(f"call ended ({update.status.name.lower()})")
                self._archive(session)

        @calls.on_update(filters.stream_end())
        async def _media_ended(_: Any, update: Any) -> None:
            session = self.sessions.get(update.chat_id)
            if session is None or session.media is None:
                return
            # Replacing the stream (seek) can report the old one as ended.
            if time.monotonic() < session.media.get("ignore_end_until", 0):
                return
            await self.stop_media(update.chat_id, "finished")

        @calls.on_update(filters.chat_update(ChatUpdate.Status.INCOMING_CALL))
        async def _incoming(_: Any, update: Any) -> None:
            if self.auto_answer is None or update.chat_id in self.sessions:
                return
            if update.chat_id not in self.allowed_callers:
                logger.info("Ignoring call from non-allowed user %s", update.chat_id)
                return
            logger.info("Answering incoming call from allowed user %s", update.chat_id)
            try:
                await self.join(update.chat_id, **self.auto_answer)
            except Exception:
                logger.exception("Could not answer incoming call")

        self._handlers_installed = True

    def _archive(self, session: VoiceSession) -> None:
        self.history.append(session.snapshot())
        del self.history[:-10]

    async def join(
        self,
        chat: Any,
        *,
        conversational: bool = True,
        instructions: str = "",
        greeting: str = "",
        max_minutes: float = 30.0,
        invite_link: str = "",
    ) -> VoiceSession:
        """Start a private call (user) or join/start a group voice chat."""
        from pytgcalls.types import (
            CallConfig,
            ExternalMedia,
            GroupCallConfig,
            MediaStream,
            RecordStream,
        )
        from pytgcalls.types.raw import AudioParameters
        from telethon import utils

        tg = TelegramUserClient.get()
        client = await tg.client()
        entity = (
            await resolve_entity(client, chat) if not isinstance(chat, int) else None
        )
        chat_id = utils.get_peer_id(entity) if entity is not None else chat
        if chat_id in self.sessions:
            return self.sessions[chat_id]
        if entity is None:
            entity = await client.get_entity(chat_id)
        calls = await tg.calls()
        await self.install_handlers(calls)

        pipeline = _SpeechPipeline() if conversational or greeting else None
        session = VoiceSession(
            chat_id,
            utils.get_display_name(entity),
            conversational=conversational,
            instructions=instructions,
            max_minutes=max_minutes,
            pipeline=pipeline,
        )
        if chat_id < 0:
            session.allowed_speakers = await self._group_speakers(client)
        params = AudioParameters(SAMPLE_RATE, CHANNELS)
        if chat_id > 0:
            config: Any = CallConfig(timeout=60)
        else:
            from openjarvis.connectors.telegram_user import invite_hash

            config = GroupCallConfig(invite_hash=invite_hash(invite_link) or None)
        self.sessions[chat_id] = session
        kind = "Calling" if chat_id > 0 else "Joining voice chat of"
        logger.info(
            "%s %s (%s, max %.0f min)%s",
            kind,
            session.title,
            "conversation" if conversational else "silent",
            max_minutes,
            f": {preview(instructions, 160)}" if instructions else "",
        )
        try:
            await calls.play(
                chat_id,
                MediaStream(
                    ExternalMedia.AUDIO,
                    params,
                    video_flags=MediaStream.Flags.IGNORE,
                ),
                config,
            )
            await calls.record(chat_id, RecordStream(True, params))
        except Exception:
            logger.exception("Could not connect the call with %s", session.title)
            self.sessions.pop(chat_id, None)
            raise
        logger.info("Call with %s connected", session.title)
        session.start_tasks(calls)
        if greeting:
            await session.say(greeting)
            session.transcript.append(
                {"role": "assistant", "speaker": "assistant", "content": greeting}
            )
        return session

    async def play_media(
        self,
        chat: Any,
        info: Any,
        *,
        video: bool,
        invite_link: str = "",
    ) -> VoiceSession:
        """Stream a resolved YouTube item into the call, joining if needed."""
        from pytgcalls.types import AudioQuality, MediaStream, VideoQuality

        chat_id = chat if isinstance(chat, int) else await resolve_chat_id(chat)
        session = self.sessions.get(chat_id)
        if session is None:
            minutes = (info.duration_s or 0) / 60 + 10
            session = await self.join(
                chat_id,
                conversational=True,
                max_minutes=min(max(minutes, 30.0), 240.0),
                invite_link=invite_link,
            )
        else:
            # Leave room for the whole track on an already running call.
            needed = time.monotonic() + (info.duration_s or 0) + 300
            session.deadline = max(session.deadline, needed)
        calls = await TelegramUserClient.get().calls()
        session.drop_queued_audio()
        stream = MediaStream(
            info.url,
            AudioQuality.HIGH,
            VideoQuality.HD_720p,
            video_flags=MediaStream.Flags.AUTO_DETECT
            if video
            else MediaStream.Flags.IGNORE,
        )
        session.media = {
            "title": info.title,
            "url": info.url,
            "video": video,
            "paused": False,
            "started": time.time(),
            "duration_s": info.duration_s,
            "live": info.is_live,
            # Position bookkeeping: offset + time since (re)start while playing.
            "offset": 0.0,
            "resumed_at": None,
            "ignore_end_until": time.monotonic() + _RESTART_GRACE_S,
        }
        try:
            await calls.play(chat_id, stream)
        except Exception:
            session.media = None
            logger.exception("Could not stream %s", info.url)
            raise
        session.media["resumed_at"] = time.monotonic()
        logger.info(
            "Playing %s in %s: %s (%s)",
            "video" if video else "audio",
            session.title,
            info.title,
            info.url,
        )
        return session

    async def stop_media(self, chat_id: int, reason: str = "stopped") -> dict[str, Any]:
        """End the track and give the outgoing stream back to the assistant."""
        from pytgcalls.types import ExternalMedia, MediaStream
        from pytgcalls.types.raw import AudioParameters

        session = self.session(chat_id)
        media, session.media = session.media, None
        if media is None:
            return {"stopped": None}
        calls = await TelegramUserClient.get().calls()
        await calls.play(
            chat_id,
            MediaStream(
                ExternalMedia.AUDIO,
                AudioParameters(SAMPLE_RATE, CHANNELS),
                video_flags=MediaStream.Flags.IGNORE,
            ),
        )
        logger.info("Media in %s %s: %s", session.title, reason, media["title"])
        return {"stopped": media["title"], "reason": reason}

    def _media(self, chat_id: int) -> dict[str, Any]:
        media = self.session(chat_id).media
        if media is None:
            raise TelegramUserError("Nothing is playing.")
        return media

    async def pause_media(self, chat_id: int) -> dict[str, Any]:
        media = self._media(chat_id)
        if not media["paused"]:
            await (await TelegramUserClient.get().calls()).pause(chat_id)
            media["offset"] = media_position(media) or 0.0
            media["resumed_at"] = None
            media["paused"] = True
        return media_status(media)

    async def resume_media(self, chat_id: int) -> dict[str, Any]:
        media = self._media(chat_id)
        if media["paused"]:
            await (await TelegramUserClient.get().calls()).resume(chat_id)
            media["resumed_at"] = time.monotonic()
            media["paused"] = False
        return media_status(media)

    async def seek_media(
        self,
        chat_id: int,
        *,
        to: Optional[float] = None,
        by: Optional[float] = None,
    ) -> dict[str, Any]:
        """Jump to an absolute position or by a signed offset (seconds)."""
        from pytgcalls.types import AudioQuality, MediaStream, VideoQuality

        media = self._media(chat_id)
        if media.get("live"):
            raise TelegramUserError("This is a livestream; it can't be fast-forwarded.")
        current = media_position(media) or 0.0
        target = float(to) if to is not None else current + float(by or 0)
        duration = media.get("duration_s")
        if duration:
            target = min(target, max(0.0, duration - 2))
        target = max(0.0, target)
        was_paused = media["paused"]
        media["ignore_end_until"] = time.monotonic() + _RESTART_GRACE_S
        calls = await TelegramUserClient.get().calls()
        await calls.play(
            chat_id,
            MediaStream(
                media["url"],
                AudioQuality.HIGH,
                VideoQuality.HD_720p,
                video_flags=MediaStream.Flags.AUTO_DETECT
                if media["video"]
                else MediaStream.Flags.IGNORE,
                ffmpeg_parameters=f"-ss {target:.1f}",
            ),
        )
        media["offset"] = target
        media["resumed_at"] = time.monotonic()
        media["paused"] = False
        if was_paused:
            await self.pause_media(chat_id)
        logger.info(
            "Seeked %s in %s to %s",
            media["title"],
            self.session(chat_id).title,
            format_position(target),
        )
        return media_status(media)

    async def media_command(
        self, chat_id: int, action: str, seconds: Optional[float], *, by: str = ""
    ) -> dict[str, Any]:
        """Dispatch a parsed transport command (see ``parse_media_command``)."""
        if action == "pause":
            return await self.pause_media(chat_id)
        if action == "resume":
            return await self.resume_media(chat_id)
        if action == "seek_by":
            return await self.seek_media(chat_id, by=seconds)
        if action == "seek_to":
            return await self.seek_media(chat_id, to=seconds)
        if action == "stop":
            return await self.stop_media(
                chat_id, f"stopped by {by}" if by else "stopped"
            )
        raise TelegramUserError(f"Unknown media action {action!r}.")

    async def _group_speakers(self, client: Any) -> Optional[frozenset[int]]:
        """Allowed group-call speakers (None means everyone is heard)."""
        if self.allowed_speakers is None:
            try:
                from openjarvis.connectors.telegram_responder import (
                    parse_allowed,
                    resolve_allowed_users,
                )
                from openjarvis.core.config import load_config

                refs = parse_allowed(load_config().tools.telegram_user.allowed_users)
                if refs:
                    resolved = await resolve_allowed_users(client, refs)
                    self.allowed_speakers = frozenset(resolved)
            except Exception:
                logger.exception("Could not resolve allowed group-call speakers")
        return self.allowed_speakers

    def session(self, chat_id: int) -> VoiceSession:
        session = self.sessions.get(chat_id)
        if session is None:
            raise TelegramUserError("Not in a call with that chat.")
        return session

    async def leave(
        self, chat_id: int, reason: str = "left by agent"
    ) -> dict[str, Any]:
        session = self.sessions.pop(chat_id, None)
        tg = TelegramUserClient.get()
        calls = await tg.calls()
        try:
            await calls.leave_call(chat_id)
        except Exception:
            logger.debug("leave_call failed", exc_info=True)
        if session is None:
            return {"chat": peer_ref(chat_id), "active": False}
        session.finish(reason)
        self._archive(session)
        return session.snapshot()

    async def set_muted(self, chat_id: int, muted: bool) -> None:
        session = self.session(chat_id)
        calls = await TelegramUserClient.get().calls()
        if muted:
            await calls.mute(chat_id)
        else:
            await calls.unmute(chat_id)
        session.muted = muted


async def resolve_chat_id(chat: Any) -> int:
    from telethon import utils

    client = await TelegramUserClient.get().client()
    return utils.get_peer_id(await resolve_entity(client, chat))


__all__ = [
    "END_CALL_TOKEN",
    "VoiceCallManager",
    "VoiceSession",
    "decode_to_pcm",
    "pcm_to_wav",
    "resolve_chat_id",
]
