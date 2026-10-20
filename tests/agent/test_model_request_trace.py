from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

import openswe.model_request as model_request


async def test_model_selection_trace_uses_a_langfuse_child_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = MagicMock()
    observations: list[tuple[str, dict[str, object]]] = []

    @asynccontextmanager
    async def fake_observation(name: str, **kwargs: object):
        observations.append((name, kwargs))
        yield span

    monkeypatch.setattr(model_request, "langfuse_observation", fake_observation, raising=False)

    async with model_request.model_selection_trace() as selected_span:
        assert selected_span is span

    assert observations == [("Model selection decision", {"input": {}, "metadata": {}})]
