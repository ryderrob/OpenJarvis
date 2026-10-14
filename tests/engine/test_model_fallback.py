"""Tests for the model fallback engine wrapper."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.engine._base import EngineConnectionError, InferenceEngine
from openjarvis.engine._stubs import StreamChunk
from openjarvis.engine.fallback import ModelFallbackEngine, parse_fallback_models


class FlakyEngine(InferenceEngine):
    engine_id = "flaky"

    def __init__(self, failing: set[str], fail_mid_stream: bool = False) -> None:
        self.failing = failing
        self.fail_mid_stream = fail_mid_stream
        self.calls: list[str] = []
        self._host = "http://gateway"

    def generate(self, messages, *, model, **kwargs) -> dict[str, Any]:
        self.calls.append(model)
        if model in self.failing:
            raise EngineConnectionError(f"{model} returned HTTP 502")
        return {"content": f"from {model}", "model": model}

    async def stream(self, messages, *, model, **kwargs):
        self.calls.append(model)
        if model in self.failing and not self.fail_mid_stream:
            raise EngineConnectionError("502")
        yield f"{model}-1"
        if model in self.failing:
            raise EngineConnectionError("dropped")
        yield f"{model}-2"

    async def stream_full(self, messages, *, model, **kwargs):
        self.calls.append(model)
        if model in self.failing:
            raise EngineConnectionError("502")
        yield StreamChunk(content=model)

    def list_models(self) -> list[str]:
        return ["a", "b", "c"]

    def health(self) -> bool:
        return True


def test_parse_fallback_models_dedupes_and_strips() -> None:
    assert parse_fallback_models(" b, c ,b,,") == ["b", "c"]
    assert parse_fallback_models(["x", "x", "y"]) == ["x", "y"]
    assert parse_fallback_models("") == []


def test_generate_falls_back_in_order() -> None:
    inner = FlakyEngine({"a", "b"})
    engine = ModelFallbackEngine(inner, "b,c")
    assert engine.generate([], model="a")["content"] == "from c"
    assert inner.calls == ["a", "b", "c"]


def test_generate_skips_duplicate_of_requested_model() -> None:
    inner = FlakyEngine(set())
    engine = ModelFallbackEngine(inner, "a,b")
    engine.generate([], model="a")
    assert inner.calls == ["a"]


def test_generate_raises_last_error_when_all_fail() -> None:
    engine = ModelFallbackEngine(FlakyEngine({"a", "b"}), "b")
    with pytest.raises(EngineConnectionError, match="b returned"):
        engine.generate([], model="a")


def test_stream_falls_back_before_first_token() -> None:
    inner = FlakyEngine({"a"})
    engine = ModelFallbackEngine(inner, "b")

    async def _collect() -> list[str]:
        return [t async for t in engine.stream([], model="a")]

    assert asyncio.run(_collect()) == ["b-1", "b-2"]


def test_stream_does_not_splice_after_output_started() -> None:
    inner = FlakyEngine({"a"}, fail_mid_stream=True)
    engine = ModelFallbackEngine(inner, "b")

    async def _collect() -> list[str]:
        out = []
        async for token in engine.stream([], model="a"):
            out.append(token)
        return out

    with pytest.raises(EngineConnectionError, match="dropped"):
        asyncio.run(_collect())
    assert inner.calls == ["a"]


def test_stream_full_falls_back() -> None:
    engine = ModelFallbackEngine(FlakyEngine({"a"}), "b")

    async def _collect() -> list[str | None]:
        return [c.content async for c in engine.stream_full([], model="a")]

    assert asyncio.run(_collect()) == ["b"]


def test_wrapper_delegates_identity_and_attributes() -> None:
    inner = FlakyEngine(set())
    engine = ModelFallbackEngine(inner, "b")
    assert engine.engine_id == "flaky"
    assert engine._host == "http://gateway"
    assert engine.list_models() == ["a", "b", "c"]


def test_make_engine_wraps_only_default_engine(monkeypatch) -> None:
    from openjarvis.engine import _discovery

    monkeypatch.setattr(
        _discovery, "_make_base_engine", lambda key, cfg: FlakyEngine(set())
    )
    config = JarvisConfig()
    config.engine.default = "llamacpp"
    config.intelligence.fallback_models = "b,c"
    assert isinstance(_discovery._make_engine("llamacpp", config), ModelFallbackEngine)
    assert isinstance(_discovery._make_engine("ollama", config), FlakyEngine)
    config.intelligence.fallback_models = ""
    assert isinstance(_discovery._make_engine("llamacpp", config), FlakyEngine)
