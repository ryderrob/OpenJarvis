"""Tests for OpenAI Whisper API speech backend."""

from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import TranscriptionResult
from openjarvis.speech.openai_whisper import OpenAIWhisperBackend


@pytest.fixture(autouse=True)
def _register_openai_whisper():
    """Re-register after any registry clear."""
    if not SpeechRegistry.contains("openai"):
        SpeechRegistry.register_value("openai", OpenAIWhisperBackend)


def test_openai_whisper_registers():
    assert SpeechRegistry.contains("openai")


def test_openai_whisper_transcribe():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from OpenAI"
    mock_response.language = "en"
    mock_response.duration = 2.0
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("openjarvis.speech.openai_whisper.OpenAI", return_value=mock_client):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend(api_key="test-key")
        result = backend.transcribe(b"fake audio", format="wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello from OpenAI"
        assert result.language == "en"


def test_openai_whisper_health():
    with patch("openjarvis.speech.openai_whisper.OpenAI"):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend(api_key="test-key")
        assert backend.health() is True


def test_openai_whisper_health_no_key():
    with patch("openjarvis.speech.openai_whisper.OpenAI"):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend.__new__(OpenAIWhisperBackend)
        backend._client = None
        backend._api_key = ""
        assert backend.health() is False


def test_openai_whisper_uses_compatible_server_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_STT_BASE_URL", "http://gateway:9999/v1")
    monkeypatch.setenv("OPENAI_STT_API_KEY", "gateway-key")
    monkeypatch.setenv("OPENAI_STT_MODEL", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = MagicMock(text="hi")

    with patch(
        "openjarvis.speech.openai_whisper.OpenAI", return_value=mock_client
    ) as ctor:
        OpenAIWhisperBackend().transcribe(b"audio", format="wav")

    ctor.assert_called_once_with(
        api_key="gateway-key", base_url="http://gateway:9999/v1"
    )
    assert mock_client.audio.transcriptions.create.call_args.kwargs["model"] == "auto"
