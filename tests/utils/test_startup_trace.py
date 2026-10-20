from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import AgentState
from langgraph.runtime import Runtime

from openswe.middleware.prepare_run import BasePrepareRunMiddleware
from openswe.utils import startup_trace
from openswe.utils.startup_trace import aphase, flush_phases


class _FakeObservation(AbstractContextManager["_FakeObservation"]):
    def __init__(self, client: _FakeClient, name: str, fields: dict[str, object]) -> None:
        self.client = client
        self.record = {
            "name": name,
            "parent": client.stack[-1].record["name"] if client.stack else None,
            **fields,
        }
        client.created.append(self.record)

    def __enter__(self) -> _FakeObservation:
        self.client.stack.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        self.client.stack.pop()
        if exc_type is not None:
            self.update(level="ERROR", status_message=f"{exc_type.__name__}: {exc}")
        return False

    def update(self, **fields: object) -> None:
        self.record.update(fields)


class _FakeClient:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.stack: list[_FakeObservation] = []

    def start_as_current_observation(self, *, name: str, **fields: object) -> _FakeObservation:
        return _FakeObservation(self, name, fields)

    def get_current_trace_id(self) -> str | None:
        return "trace-1" if self.stack else None


async def _flush_in_traced_node(thread_id: str, client: _FakeClient) -> None:
    with client.start_as_current_observation(name="agent", as_type="agent"):
        flush_phases(thread_id)


@pytest.fixture(autouse=True)
def _clean_phases() -> Any:
    startup_trace._PHASES.clear()
    yield
    startup_trace._PHASES.clear()


async def test_failed_phase_is_replayed_as_an_error_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(startup_trace, "get_langfuse_client", lambda: client, raising=False)
    with pytest.raises(RuntimeError):
        async with aphase("thread-2", "sandbox.boot"):
            raise RuntimeError("boom")

    await _flush_in_traced_node("thread-2", client)

    boot = next(span for span in client.created if span["name"] == "sandbox.boot")
    startup = next(span for span in client.created if span["name"] == "startup")
    assert startup["parent"] == "agent"
    assert boot["parent"] == "startup"
    assert boot["level"] == "ERROR"
    assert boot["status_message"] == "RuntimeError: boom"


async def test_unfinished_phase_is_kept_for_the_next_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(startup_trace, "get_langfuse_client", lambda: client, raising=False)
    phase = startup_trace._open("thread-5", "sandbox.boot", {})
    assert phase is not None

    await _flush_in_traced_node("thread-5", client)

    assert [span["name"] for span in client.created] == ["agent"]
    startup_trace._close(phase, None)
    await _flush_in_traced_node("thread-5", client)
    assert [span["name"] for span in client.created] == [
        "agent",
        "agent",
        "startup",
        "sandbox.boot",
    ]
    boot = client.created[-1]
    assert boot["parent"] == "startup"
    output = boot["output"]
    assert isinstance(output, dict)
    assert output["elapsed_ms"] == phase.elapsed_ms


async def test_prepare_middleware_flushes_phases_even_when_prepare_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup_trace, "get_langfuse_client", lambda: None, raising=False)

    class _Failing(BasePrepareRunMiddleware):
        _thread_id = "thread-4"

        async def _prepare(self, state: Any, runtime: Any) -> dict[str, Any]:
            del state, runtime
            raise RuntimeError("no sandbox")

    async with aphase("thread-4", "sandbox.boot"):
        pass

    with pytest.raises(RuntimeError):
        await _Failing().abefore_agent(
            cast(AgentState, {"messages": []}), cast(Runtime[None], MagicMock())
        )

    assert "thread-4" not in startup_trace._PHASES
