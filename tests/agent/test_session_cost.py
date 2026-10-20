from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from openswe import session_cost


@pytest.mark.parametrize("run_cost", [3.25, None])
async def test_refresh_updates_exact_mapped_slack_message_in_place(
    monkeypatch: pytest.MonkeyPatch,
    run_cost: float | None,
) -> None:
    class _Store:
        async def get_item(self, namespace: Any, key: str) -> dict[str, Any] | None:
            if key == "run:run-1":
                return {
                    "value": {
                        "run_id": "run-1",
                        "thread_ts": "1.0",
                        "message_ts": "1.1",
                    }
                }
            return None

    client: Any = SimpleNamespace(store=_Store())
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "Done"}},
        {"type": "actions", "elements": [{"type": "button", "action_id": "approve"}]},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "<https://app/agents/t1|Open in Web> • model-a • calculating cost...",
                }
            ],
        },
    ]

    async def get_cost(
        thread_id: str,
        invocation_id: str,
        *,
        run_only: bool = False,
        lookup_start: str | None = None,
    ) -> SimpleNamespace | None:
        if run_only:
            return SimpleNamespace(total_cost=run_cost) if run_cost is not None else None
        return SimpleNamespace(total_cost=12.50)

    monkeypatch.setattr(session_cost, "get_trace_cost", get_cost)
    monkeypatch.setattr(
        session_cost,
        "lookup_slack_run_message_mapping",
        AsyncMock(return_value={"thread_ts": "1.0", "message_ts": "1.1"}),
    )
    monkeypatch.setattr(
        session_cost,
        "fetch_slack_thread_message_by_ts",
        AsyncMock(
            return_value={
                "text": "Done <https://app/agents/t1|Open in Web> • model-a • calculating cost...",
                "blocks": blocks,
            }
        ),
    )
    update = AsyncMock(return_value=None)
    monkeypatch.setattr(session_cost, "update_slack_message", update)

    status, reason = await session_cost._refresh_once(_state(0), client)

    if run_cost is None:
        assert status == "pending"
        update.assert_not_awaited()
        return
    assert (status, reason) == ("updated", "Slack footer updated")
    update.assert_awaited_once()
    args = update.await_args
    assert args is not None
    assert args.args[:2] == ("C1", "1.1")
    assert args.args[2].endswith("model-a • $12.50 ($3.25)")
    assert args.kwargs["blocks"][-1]["elements"][0]["text"].endswith("$12.50 ($3.25)")
    assert "main-agent tokens" not in args.args[2]
    assert args.kwargs["blocks"][1] == blocks[1]


class _Runs:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, thread_id: str | None, assistant_id: str, **kwargs: Any) -> None:
        self.created.append({"thread_id": thread_id, "assistant_id": assistant_id, **kwargs})


class _Client:
    def __init__(self) -> None:
        self.runs = _Runs()


def _state(attempt: int) -> dict[str, Any]:
    return {
        "task": "session_cost",
        "agent_thread_id": "thread-1",
        "run_id": "run-1",
        "prepare_run_id": "prepare-1",
        "channel_id": "C1",
        "thread_ts": "1.0",
        "attempt": attempt,
    }


async def test_refresh_stops_after_final_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    client: Any = _Client()
    monkeypatch.setattr(
        session_cost,
        "_refresh_once",
        AsyncMock(return_value=("pending", "Langfuse not ready")),
    )

    result = await session_cost.run_session_cost_refresh(_state(4), client=client)

    assert result == {"status": "exhausted", "reason": "Langfuse not ready"}
    assert client.runs.created == []
