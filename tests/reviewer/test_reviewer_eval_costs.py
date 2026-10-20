from types import SimpleNamespace
from unittest.mock import AsyncMock

from langfuse import Evaluation

from evals.reviewer import costs


async def test_cost_evaluator_scores_complete_cost_and_tokens(monkeypatch) -> None:  # noqa: ANN001
    snapshot = SimpleNamespace(total_cost=1.25, total_tokens=420)
    lookup = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(costs, "get_trace_cost", lookup)

    scores = await costs.evaluate_pr_cost(
        output={
            "thread_id": "thread-1",
            "invocation_id": "run-1",
            "invocation_started_at": "2026-10-20T12:00:00+00:00",
            "pr_url": "https://github.com/o/r/pull/1",
        }
    )

    assert [(score.name, score.value) for score in scores] == [
        ("cost_status", "priced"),
        ("cost_usd", 1.25),
        ("total_tokens", 420),
    ]
    lookup.assert_awaited_once_with(
        "thread-1",
        "run-1",
        run_only=True,
        lookup_start="2026-10-20T12:00:00+00:00",
    )


async def test_missing_cost_is_explicitly_unpriced_and_does_not_write_zero(
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(costs, "get_trace_cost", AsyncMock(return_value=None))
    monkeypatch.setattr(costs.asyncio, "sleep", AsyncMock())

    scores = await costs.evaluate_pr_cost(
        output={"thread_id": "thread-1", "invocation_id": "run-1", "pr_url": "pr"}
    )

    assert [(score.name, score.value) for score in scores] == [("cost_status", "unpriced")]
    assert not any(score.name == "cost_usd" for score in scores)


def test_experiment_cost_summary_counts_priced_and_unpriced_items() -> None:
    items = [
        SimpleNamespace(
            item=SimpleNamespace(input={"pr_url": "pr-1"}),
            output={"thread_id": "thread-1"},
            evaluations=[
                Evaluation(name="cost_status", value="priced"),
                Evaluation(name="cost_usd", value=1.25),
                Evaluation(name="total_tokens", value=420),
            ],
        ),
        SimpleNamespace(
            item=SimpleNamespace(input={"pr_url": "pr-2"}),
            output={"thread_id": "thread-2"},
            evaluations=[Evaluation(name="cost_status", value="unpriced")],
        ),
    ]

    summary = costs.summarize_experiment_costs(items)

    assert summary.total_usd == 1.25
    assert summary.total_tokens == 420
    assert summary.unpriced == ["pr-2"]
    scores = costs.record_experiment_costs(item_results=items)
    assert {item.name: item.value for item in scores}["cost_unpriced_prs"] == 1
