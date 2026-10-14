"""OpenAI Whisper API speech-to-text backend (cloud)."""

from __future__ import annotations

import io
import os
from typing import List, Optional

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import SpeechBackend, TranscriptionResult

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]


# Point transcription at any OpenAI-compatible server (e.g. a self-hosted
# gateway) without hijacking OPENAI_API_KEY, which cloud chat routing reads.
_BASE_URL_ENV = "OPENAI_STT_BASE_URL"
_API_KEY_ENV = "OPENAI_STT_API_KEY"
_MODEL_ENV = "OPENAI_STT_MODEL"


@SpeechRegistry.register("openai")
class OpenAIWhisperBackend(SpeechBackend):
    """Cloud speech-to-text using OpenAI Whisper API."""

    backend_id = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get(_API_KEY_ENV, "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._base_url = base_url or os.environ.get(_BASE_URL_ENV, "") or None
        self._model = model or os.environ.get(_MODEL_ENV, "") or "whisper-1"
        self._client: Optional[OpenAI] = None
        if self._api_key and OpenAI is not None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI's Whisper API."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized (missing API key?)")

        ext = format if not format.startswith(".") else format[1:]
        audio_file = io.BytesIO(audio)
        audio_file.name = f"audio.{ext}"

        kwargs: dict = {"model": self._model, "file": audio_file}
        if language:
            kwargs["language"] = language
        kwargs["response_format"] = "verbose_json"

        response = self._client.audio.transcriptions.create(**kwargs)

        return TranscriptionResult(
            text=getattr(response, "text", str(response)),
            language=getattr(response, "language", None),
            confidence=None,
            duration_seconds=getattr(response, "duration", 0.0),
            segments=[],
        )

    def health(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def supported_formats(self) -> List[str]:
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
