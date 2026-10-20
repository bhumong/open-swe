from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openswe.utils import langfuse

_START = datetime(2026, 10, 20, 12, tzinfo=UTC)
_END = datetime(2026, 10, 20, 12, 1, tzinfo=UTC)


def _observation(
    *,
    trace_id: str,
    kind: str,
    start: datetime,
    end: datetime | None,
    cost: float | None = None,
    tokens: int | None = None,
    root: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id=trace_id,
        type=kind,
        start_time=start,
        end_time=end,
        total_cost=cost,
        cost_details={"total": cost} if cost is not None else {},
        usage_details={"total": tokens} if tokens is not None else {},
        is_root_observation=root,
    )


async def test_get_trace_cost_sums_session_generations_and_keeps_target_end_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = [
        _observation(trace_id="trace-old", kind="SPAN", start=_START, end=_END, root=True),
        _observation(
            trace_id="trace-old", kind="GENERATION", start=_START, end=_END, cost=0.1, tokens=10
        ),
        _observation(trace_id="trace-new", kind="SPAN", start=_END, end=_END, root=True),
        _observation(
            trace_id="trace-new", kind="GENERATION", start=_END, end=_END, cost=0.25, tokens=20
        ),
    ]
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(
                get_many=AsyncMock(return_value=SimpleNamespace(data=observations, meta=None))
            )
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value="trace-new"))

    result = await langfuse.get_trace_cost("thread-1", "run-1")

    assert result == langfuse.LangfuseThreadCost(
        total_cost=0.35, total_tokens=30, last_end_time=_END, target_end_time=_END
    )
    client.async_api.observations.get_many.assert_awaited_once_with(
        session_id="thread-1",
        fields="core,basic,time,usage",
        limit=1000,
    )


async def test_get_trace_cost_can_limit_to_one_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = [
        _observation(trace_id="trace-1", kind="SPAN", start=_START, end=_END, root=True),
        _observation(
            trace_id="trace-1", kind="GENERATION", start=_START, end=_END, cost=0.25, tokens=10
        ),
    ]
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(
                get_many=AsyncMock(return_value=SimpleNamespace(data=observations, meta=None))
            )
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value="trace-1"))

    result = await langfuse.get_trace_cost("thread-1", "run-1", run_only=True)

    assert result is not None and result.total_cost == 0.25
    client.async_api.observations.get_many.assert_awaited_once_with(
        trace_id="trace-1",
        fields="core,basic,time,usage",
        limit=1000,
    )


@pytest.mark.parametrize(
    "generation",
    [
        _observation(trace_id="trace-1", kind="GENERATION", start=_START, end=_END),
        _observation(trace_id="trace-1", kind="GENERATION", start=_START, end=None, cost=0.25),
    ],
)
async def test_get_trace_cost_waits_for_complete_usage(
    monkeypatch: pytest.MonkeyPatch,
    generation: SimpleNamespace,
) -> None:
    observations = [
        _observation(trace_id="trace-1", kind="SPAN", start=_START, end=_END, root=True),
        generation,
    ]
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            observations=SimpleNamespace(
                get_many=AsyncMock(return_value=SimpleNamespace(data=observations, meta=None))
            )
        )
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value="trace-1"))

    assert await langfuse.get_trace_cost("thread-1", "run-1") is None


async def test_get_trace_cost_does_not_report_without_trace_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_many = AsyncMock()
    client = SimpleNamespace(
        async_api=SimpleNamespace(observations=SimpleNamespace(get_many=get_many))
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value=None))

    assert await langfuse.get_trace_cost("thread-1", "missing-run") is None
    get_many.assert_not_called()


async def test_create_trace_score_retries_update_the_same_score_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 10, 20, 12, tzinfo=UTC)
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            scores=SimpleNamespace(
                get_by_id=AsyncMock(side_effect=[None, SimpleNamespace(timestamp=created_at)])
            ),
            legacy=SimpleNamespace(score_v1=SimpleNamespace(delete=AsyncMock())),
        ),
        create_score=Mock(),
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "resolve_trace_id_for_run", AsyncMock(return_value="trace-1"))

    first = await langfuse.create_trace_score(
        "run-1", "helpful", score=1.0, comment="good", timestamp=created_at
    )
    second = await langfuse.create_trace_score("run-1", "helpful", score=0.0, comment="updated")

    assert first is second is True
    calls = [call.kwargs for call in client.create_score.call_args_list]
    assert calls[0]["score_id"] == calls[1]["score_id"]
    assert calls[0]["name"] == calls[1]["name"] == "helpful"
    assert calls[0]["timestamp"] == calls[1]["timestamp"] == created_at
    assert calls[0]["trace_id"] == calls[1]["trace_id"] == "trace-1"
    assert [call["value"] for call in calls] == [1.0, 0.0]


async def test_thread_score_is_attached_to_resolved_trace_and_missing_trace_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        async_api=SimpleNamespace(
            scores=SimpleNamespace(get_by_id=AsyncMock(return_value=None)),
            legacy=SimpleNamespace(score_v1=SimpleNamespace(delete=AsyncMock())),
        ),
        create_score=Mock(),
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)
    resolve = AsyncMock(side_effect=["trace-1", None])
    monkeypatch.setattr(langfuse, "resolve_trace_id", resolve)

    assert await langfuse.create_thread_trace_score("thread-1", "rating", score=1.0, run_id="run-1")
    assert not await langfuse.create_thread_trace_score("thread-1", "rating", score=0.0)
    assert client.create_score.call_args.kwargs["trace_id"] == "trace-1"
    assert "session_id" not in client.create_score.call_args.kwargs
    assert client.create_score.call_count == 1


async def test_delete_trace_score_uses_langfuse_sdk_delete_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete = AsyncMock()
    client = SimpleNamespace(
        async_api=SimpleNamespace(legacy=SimpleNamespace(score_v1=SimpleNamespace(delete=delete)))
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    assert await langfuse.delete_trace_score("run-1", "helpful")

    delete.assert_awaited_once_with(score_id=langfuse._score_id("run", "run-1", "helpful"))
