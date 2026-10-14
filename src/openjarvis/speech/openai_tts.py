"""OpenAI TTS backend — cloud-based voice synthesis via OpenAI API."""

from __future__ import annotations

import logging
import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

logger = logging.getLogger(__name__)

_OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
# Point TTS at any OpenAI-compatible server (e.g. a self-hosted gateway)
# without hijacking OPENAI_API_KEY, which cloud chat routing also reads.
_BASE_URL_ENV = "OPENAI_TTS_BASE_URL"
_API_KEY_ENV = "OPENAI_TTS_API_KEY"
_MODEL_ENV = "OPENAI_TTS_MODEL"


def _speech_url() -> str:
    base = os.environ.get(_BASE_URL_ENV, "").strip().rstrip("/")
    return f"{base}/audio/speech" if base else _OPENAI_TTS_URL


def _openai_tts_request(
    api_key: str,
    text: str,
    voice: str,
    model: str = "tts-1",
    speed: float = 1.0,
    response_format: str = "mp3",
) -> bytes:
    """Call the OpenAI TTS API and return raw audio bytes."""
    resp = httpx.post(
        _speech_url(),
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content


@TTSRegistry.register("openai_tts")
class OpenAITTSBackend(TTSBackend):
    """OpenAI TTS backend — cloud synthesis."""

    backend_id = "openai_tts"

    def __init__(self, *, api_key: str = "", model: str = "") -> None:
        self._api_key = (
            api_key
            or os.environ.get(_API_KEY_ENV, "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._model = model or os.environ.get(_MODEL_ENV, "") or "tts-1"

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "nova",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError(f"{_API_KEY_ENV} or OPENAI_API_KEY not set")

        # A comma-separated model list is tried in order, e.g. a voice-capable
        # model first and a gateway's "auto" router as a fallback.
        models = [m.strip() for m in self._model.split(",") if m.strip()]
        audio = b""
        for index, model in enumerate(models):
            try:
                audio = _openai_tts_request(
                    self._api_key,
                    text,
                    voice=voice_id,
                    model=model,
                    speed=speed,
                    response_format=output_format,
                )
                break
            except httpx.HTTPError as exc:
                if index == len(models) - 1:
                    raise
                logger.warning(
                    "TTS model %s failed (%s); trying %s", model, exc, models[index + 1]
                )

        return TTSResult(
            audio=audio,
            format=output_format,
            voice_id=voice_id,
            metadata={"backend": "openai_tts", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def health(self) -> bool:
        return bool(self._api_key)
