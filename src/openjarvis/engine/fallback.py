"""Model fallback wrapper — retry a request on the next model when one fails.

Gateways that front many free providers (e.g. an OpenAI-compatible router)
regularly return 5xx/429 for an individual model while others are healthy.
:class:`ModelFallbackEngine` keeps the requested model first and, on an
:class:`EngineConnectionError`, retries the same request with each model in
``intelligence.fallback_models`` in order. Streams only fall back while no
output has been yielded yet, so callers never see two models' text spliced.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

from openjarvis.core.types import Message
from openjarvis.engine._base import EngineConnectionError, InferenceEngine
from openjarvis.engine._stubs import StreamChunk

logger = logging.getLogger(__name__)


def parse_fallback_models(value: str | Sequence[str] | None) -> List[str]:
    """Normalise a comma-separated string (or list) into unique model ids."""
    items = value.split(",") if isinstance(value, str) else list(value or [])
    return list(dict.fromkeys(m.strip() for m in items if m and m.strip()))


class ModelFallbackEngine(InferenceEngine):
    """Transparent wrapper that fails over across an ordered model list."""

    def __init__(self, inner: InferenceEngine, fallback_models: Sequence[str]) -> None:
        self._inner = inner
        self._fallbacks = parse_fallback_models(fallback_models)
        # Report the wrapped engine's identity (status UIs, routing by id).
        self.engine_id = getattr(inner, "engine_id", "")

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined here (host, api key, ...).
        return getattr(self._inner, name)

    def _chain(self, model: str) -> List[str]:
        return [model, *(m for m in self._fallbacks if m != model)]

    def _log_failover(self, model: str, nxt: str | None, exc: Exception) -> None:
        if nxt is not None:
            logger.warning("Model %s failed (%s); falling back to %s", model, exc, nxt)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        chain = self._chain(model)
        for index, candidate in enumerate(chain):
            try:
                return self._inner.generate(
                    messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except EngineConnectionError as exc:
                if index == len(chain) - 1:
                    raise
                self._log_failover(candidate, chain[index + 1], exc)
        raise AssertionError("unreachable")  # pragma: no cover

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        chain = self._chain(model)
        for index, candidate in enumerate(chain):
            started = False
            try:
                async for token in self._inner.stream(
                    messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    started = True
                    yield token
                return
            except EngineConnectionError as exc:
                if started or index == len(chain) - 1:
                    raise
                self._log_failover(candidate, chain[index + 1], exc)

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        chain = self._chain(model)
        for index, candidate in enumerate(chain):
            started = False
            try:
                async for chunk in self._inner.stream_full(
                    messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    started = True
                    yield chunk
                return
            except EngineConnectionError as exc:
                if started or index == len(chain) - 1:
                    raise
                self._log_failover(candidate, chain[index + 1], exc)

    def list_models(self) -> List[str]:
        return self._inner.list_models()

    def health(self) -> bool:
        return self._inner.health()

    def can_serve(self, model: str) -> bool:
        return self._inner.can_serve(model)

    def close(self) -> None:
        self._inner.close()

    def prepare(self, model: str) -> None:
        self._inner.prepare(model)


__all__ = ["ModelFallbackEngine", "parse_fallback_models"]
