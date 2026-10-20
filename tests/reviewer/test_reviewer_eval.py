from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evals.reviewer import costs, judge, run_eval


def test_config_precedence_uses_langfuse_project_label(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'dataset_name = "file-dataset"\n'
        'experiment_prefix = "file-run"\n'
        'langfuse_project = "File project"\n'
    )
    monkeypatch.setattr(run_eval, "CONFIG_PATH", config_path)
    monkeypatch.setenv("REVIEWER_EVAL_DATASET_NAME", "env-dataset")
    monkeypatch.setenv("LANGFUSE_PROJECT_NAME", "Env project")

    config = run_eval._resolve_config({"experiment_prefix": "cli-run"})

    assert config["dataset_name"] == "env-dataset"
    assert config["experiment_prefix"] == "cli-run"
    assert config["langfuse_project"] == "Env project"


def test_load_eval_items_requires_dataset_and_applies_limit() -> None:
    items = [SimpleNamespace(id=str(i)) for i in range(3)]
    client = SimpleNamespace(get_dataset=Mock(return_value=SimpleNamespace(items=items)))

    assert run_eval._load_eval_items(client, "reviewer", 2) == items[:2]
    client.get_dataset.assert_called_once_with("reviewer")

    client.get_dataset.side_effect = RuntimeError("Langfuse unavailable")
    with pytest.raises(RuntimeError, match="Could not load Langfuse dataset reviewer"):
        run_eval._load_eval_items(client, "reviewer", None)


def test_langfuse_experiment_receives_dataset_items_and_existing_scoring() -> None:
    item = SimpleNamespace(id="golden-1")
    result = SimpleNamespace(dataset_run_url="http://127.0.0.1:3030/project/p/datasets/d/runs/r")
    client = SimpleNamespace(run_experiment=Mock(return_value=result))
    config = {
        "dataset_name": "reviewer",
        "experiment_prefix": "baseline",
        "max_concurrency": 7,
        "langfuse_project": "local project",
        "model_id": "anthropic:claude-haiku-5-5",
        "reasoning_effort": "high",
        "score_mode": "surfaced_findings",
        "severity_threshold": "medium",
    }

    actual = run_eval._run_dataset_experiment(client, config, [item])

    assert actual is result
    kwargs = client.run_experiment.call_args.kwargs
    assert kwargs["name"] == "baseline"
    assert kwargs["data"] == [item]
    assert kwargs["task"] is run_eval._run_item
    assert kwargs["evaluators"] == [judge.judge_match, costs.evaluate_pr_cost]
    assert kwargs["run_evaluators"] == [judge.aggregate_pr, costs.record_experiment_costs]
    assert kwargs["max_concurrency"] == 7
    assert kwargs["metadata"]["langfuse_project"] == "local project"


async def test_experiment_runner_offloads_sync_sdk_call(monkeypatch) -> None:  # noqa: ANN001
    client = object()
    config = {}
    items = [object()]
    result = object()
    offload = AsyncMock(return_value=result)
    monkeypatch.setattr(run_eval.asyncio, "to_thread", offload)

    assert await run_eval._run_dataset_experiment_async(client, config, items) is result

    offload.assert_awaited_once_with(run_eval._run_dataset_experiment, client, config, items)


def test_judge_match_preserves_metric_names_and_aggregate(monkeypatch) -> None:  # noqa: ANN001
    judge._judge = None
    monkeypatch.setattr(
        judge,
        "_get_judge",
        lambda: SimpleNamespace(
            invoke=Mock(
                return_value=SimpleNamespace(
                    content='{"reasoning":"same","match":true,"confidence":1.0}'
                )
            )
        ),
    )
    judge._drain_counts()
    evaluations = judge.judge_match(
        input={"repo": "org/repo"},
        output={"comments": [{"comment": "same bug", "severity": "high"}]},
        expected_output={
            "golden_comments": [{"comment": "same bug", "severity": "high", "category": "bug"}]
        },
        metadata={"source_identity": "org/repo#1:golden"},
    )

    scores = {item.name: item.value for item in evaluations}
    assert scores["f1"] == 1.0
    assert scores["precision"] == 1.0
    assert "pairwise_match_matrix" in scores

    aggregate = judge.aggregate_pr(item_results=[])
    aggregate_scores = {item.name: item.value for item in aggregate}
    assert aggregate_scores["core_micro_f2"] == 1.0
