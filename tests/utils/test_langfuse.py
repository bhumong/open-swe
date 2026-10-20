from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from openswe.utils import langfuse


def test_get_langfuse_client_requires_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")

    assert langfuse.get_langfuse_client() is None


def test_get_langfuse_client_initializes_and_caches_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "local-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "local-secret")
    monkeypatch.setattr(langfuse, "_LANGFUSE_CLIENT", None)
    monkeypatch.setattr(langfuse, "_LANGFUSE_CLIENT_PUBLIC_KEY", None)
    client = MagicMock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(langfuse, "Langfuse", constructor)

    assert langfuse.get_langfuse_client() is client
    assert langfuse.get_langfuse_client() is client

    constructor.assert_called_once_with(public_key="local-public", secret_key="local-secret")


def test_callback_handler_uses_the_initialized_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "local-public")
    handler = Mock()
    constructor = Mock(return_value=handler)
    monkeypatch.setattr(langfuse, "CallbackHandler", constructor)

    assert langfuse._new_callback_handler() is handler

    constructor.assert_called_once_with(public_key="local-public")


def test_trace_url_uses_the_configured_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SimpleNamespace(
        get_trace_url=Mock(return_value="http://localhost:3030/project/p/traces/t")
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    assert langfuse.trace_url("t") == "http://localhost:3030/project/p/traces/t"
    client.get_trace_url.assert_called_once_with(trace_id="t")


async def test_resolve_trace_id_prefers_run_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = SimpleNamespace(
        data=[
            SimpleNamespace(
                trace_id="trace-old",
                session_id="thread-1",
                metadata={"openswe.run_id": "invocation-old"},
                start_time="2026-10-20T10:00:00+00:00",
            ),
            SimpleNamespace(
                trace_id="trace-current",
                session_id="thread-1",
                metadata={"openswe.run_id": "invocation-current"},
                start_time="2026-10-20T10:05:00+00:00",
            ),
        ]
    )
    get_many = AsyncMock(return_value=observations)
    client = SimpleNamespace(
        async_api=SimpleNamespace(observations=SimpleNamespace(get_many=get_many))
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    result = await langfuse.resolve_trace_id("thread-1", "invocation-current")

    assert result == "trace-current"
    get_many.assert_awaited_once()


async def test_resolve_trace_id_matches_langgraph_run_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = SimpleNamespace(
        data=[
            SimpleNamespace(
                trace_id="trace-current",
                session_id="thread-1",
                metadata={"langgraph_run_id": "langgraph-run-1"},
                start_time="2026-10-20T10:05:00+00:00",
            )
        ]
    )
    get_many = AsyncMock(return_value=observations)
    client = SimpleNamespace(
        async_api=SimpleNamespace(observations=SimpleNamespace(get_many=get_many))
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    result = await langfuse.resolve_trace_id("thread-1", "langgraph-run-1")

    assert result == "trace-current"
    assert "langgraph_run_id" in get_many.await_args.kwargs["expand_metadata"]


async def test_resolve_trace_id_for_run_searches_langgraph_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_many(**kwargs: object) -> SimpleNamespace:
        if "langgraph_run_id" in str(kwargs.get("filter")):
            return SimpleNamespace(
                data=[SimpleNamespace(is_root_observation=True, trace_id="trace-current")]
            )
        return SimpleNamespace(data=[])

    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(get_many=AsyncMock(side_effect=get_many))
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    assert await langfuse.resolve_trace_id_for_run("langgraph-run-1") == "trace-current"


async def test_resolve_trace_id_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(get_many=AsyncMock(side_effect=RuntimeError("offline")))
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    assert await langfuse.resolve_trace_id("thread-1", "invocation-1") is None


async def test_resolve_trace_id_does_not_guess_between_recent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    observations = SimpleNamespace(
        data=[
            SimpleNamespace(
                trace_id=trace_id,
                session_id="thread-1",
                metadata={},
                start_time=start + offset,
            )
            for trace_id, offset in (
                ("trace-a", timedelta(minutes=1)),
                ("trace-b", timedelta(minutes=2)),
            )
        ]
    )
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(get_many=AsyncMock(return_value=observations))
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    result = await langfuse.resolve_trace_id(
        "thread-1", "missing-invocation", start, fallback_window=timedelta(minutes=5)
    )

    assert result is None


async def test_resolve_trace_id_ignores_runs_outside_fallback_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    observation = SimpleNamespace(
        trace_id="trace-old",
        session_id="thread-1",
        metadata={},
        start_time=start - timedelta(minutes=1),
    )
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(
                get_many=AsyncMock(return_value=SimpleNamespace(data=[observation]))
            )
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    result = await langfuse.resolve_trace_id(
        "thread-1", "missing-invocation", start, fallback_window=timedelta(minutes=5)
    )

    assert result is None


async def test_observation_is_a_noop_without_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: None)

    async with langfuse.langfuse_observation("model_retry", metadata={}) as span:
        assert span is None


async def test_observation_creation_failure_does_not_break_application_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(start_as_current_observation=Mock(side_effect=RuntimeError("offline")))
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    async with langfuse.langfuse_observation("model_retry", metadata={}) as span:
        assert span is None


async def test_observation_failure_preserves_the_application_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MagicMock()
    span = manager.__enter__.return_value
    client = SimpleNamespace(start_as_current_observation=Mock(return_value=manager))
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    error = RuntimeError("model call failed")

    with pytest.raises(RuntimeError) as raised:
        async with langfuse.langfuse_observation("model_retry", metadata={}):
            raise error

    assert raised.value is error
    span.update.assert_called_once_with(
        level="ERROR", status_message="RuntimeError: model call failed"
    )
    manager.__exit__.assert_called_once()
