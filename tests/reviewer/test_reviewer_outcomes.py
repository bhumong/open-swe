import asyncio
from collections.abc import Callable
from typing import TypedDict, cast

from langfuse.api import NotFoundError

from openswe.review.findings import Finding
from openswe.utils import reviewer_outcomes
from openswe.utils.reviewer_outcomes import (
    FALSE_POSITIVE,
    TRUE_POSITIVE,
    upsert_finding_outcome,
    upsert_run_outcome,
)
from openswe.utils.typed_json import JsonObject


class _DatasetRecord(TypedDict):
    name: str
    description: str | None


class _DatasetItemRecord(TypedDict):
    dataset_name: str
    id: str
    input: JsonObject
    expected_output: JsonObject
    metadata: JsonObject


class _OutcomeInput(TypedDict, total=False):
    repo: str | None
    pr_number: int | None
    pr_url: str
    file: str | None
    start_line: int | None
    end_line: int | None
    side: str
    diff_hunk: str | None
    base_sha: str
    head_sha: str
    run_id: str
    channel_id: str


class _FindingOutput(TypedDict, total=False):
    label: str
    label_source: str
    finding: dict[str, object]
    resolution_note: str | None


class _RunOutput(TypedDict):
    label: str
    label_source: str


class _FakeDatasets:
    def __init__(self, exists: bool = True) -> None:
        self.exists = exists
        self.lookups: list[str] = []

    def get(self, *, dataset_name: str) -> object:
        self.lookups.append(dataset_name)
        if not self.exists:
            raise NotFoundError({}, {})
        return object()


class _FakeApi:
    def __init__(self, exists: bool = True) -> None:
        self.datasets = _FakeDatasets(exists)


class _FakeClient:
    def __init__(self, exists: bool = True) -> None:
        self.api = _FakeApi(exists)
        self.dataset_items: list[_DatasetItemRecord] = []
        self.created_datasets: list[_DatasetRecord] = []

    def create_dataset(self, *, name: str, description: str | None = None) -> object:
        self.created_datasets.append({"name": name, "description": description})
        return object()

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        id: str,
        input: JsonObject,
        expected_output: JsonObject,
        metadata: JsonObject,
    ) -> object:
        self.dataset_items.append(
            {
                "dataset_name": dataset_name,
                "id": id,
                "input": input,
                "expected_output": expected_output,
                "metadata": metadata,
            }
        )
        return object()


def _patch_client(monkeypatch, fake: _FakeClient | None) -> None:  # noqa: ANN001
    monkeypatch.setattr(reviewer_outcomes, "get_langfuse_client", lambda: fake)


async def _run_inline[**P, T](func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    return func(*args, **kwargs)


def _patch_thread_offload(monkeypatch) -> None:  # noqa: ANN001
    # Python 3.14 thread wakeups are unavailable in this sandbox. These tests
    # exercise the dataset payload and don't need to validate asyncio's executor.
    monkeypatch.setattr(asyncio, "to_thread", _run_inline)


def test_outcome_item_ids_are_scoped_to_the_dataset(monkeypatch) -> None:  # noqa: ANN001
    finding_id = reviewer_outcomes._example_id("o/r", "f_abc", "resolved_by_commit")
    run_id = reviewer_outcomes._run_example_id("run_1", "slack_thumbs_up")

    monkeypatch.setattr(reviewer_outcomes, "OUTCOMES_DATASET_NAME", "alternate-outcomes")

    assert reviewer_outcomes._example_id("o/r", "f_abc", "resolved_by_commit") != finding_id
    assert reviewer_outcomes._run_example_id("run_1", "slack_thumbs_up") != run_id


def _finding() -> Finding:
    return cast(
        Finding,
        {
            "id": "f_abc",
            "file": "app/x.rb",
            "start_line": 10,
            "end_line": 12,
            "side": "RIGHT",
            "diff_hunk": "@@ -1 +1 @@\n-foo\n+bar",
            "title": "NoMethodError on nil",
            "description": "body",
            "severity": "high",
            "confidence": "high",
            "category": "correctness",
            "first_seen_sha": "aaa",
            "github_review_run_id": "run_1",
            "resolution_note": "fixed in def",
        },
    )


async def test_upsert_finding_outcome_creates_dataset_and_stable_item(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient(exists=False)
    _patch_client(monkeypatch, fake)
    _patch_thread_offload(monkeypatch)

    ok = await upsert_finding_outcome(
        _finding(),
        label=TRUE_POSITIVE,
        label_source="resolved_by_commit",
        repo="o/r",
        pr_number=7,
        pr_url="https://github.com/o/r/pull/7",
        base_sha="aaa",
        head_sha="bbb",
        thread_id="t1",
    )
    assert ok
    assert fake.api.datasets.lookups == [reviewer_outcomes.OUTCOMES_DATASET_NAME]
    assert fake.created_datasets[0]["name"] == reviewer_outcomes.OUTCOMES_DATASET_NAME
    assert len(fake.dataset_items) == 1
    call = fake.dataset_items[0]
    inputs = cast(_OutcomeInput, call["input"])
    expected_output = cast(_FindingOutput, call["expected_output"])
    assert call["dataset_name"] == reviewer_outcomes.OUTCOMES_DATASET_NAME
    assert call["id"] == str(reviewer_outcomes._example_id("o/r", "f_abc", "resolved_by_commit"))
    assert inputs["repo"] == "o/r"
    assert inputs["file"] == "app/x.rb"
    assert inputs["diff_hunk"] is not None and inputs["diff_hunk"].startswith("@@")
    assert expected_output["label"] == TRUE_POSITIVE
    assert expected_output["label_source"] == "resolved_by_commit"
    assert expected_output["finding"]["severity"] == "high"
    assert call["metadata"]["granularity"] == "finding"
    assert call["metadata"]["repo"] == "o/r"
    assert call["metadata"]["run_id"] == "run_1"


async def test_upsert_run_outcome_uses_deterministic_item_id(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient()
    _patch_client(monkeypatch, fake)
    _patch_thread_offload(monkeypatch)

    ok = await upsert_run_outcome(
        label=FALSE_POSITIVE,
        label_source="slack_thumbs_down",
        run_id="run_1",
        repo="o/r",
        extra={"channel_id": "C1"},
    )

    assert ok
    call = fake.dataset_items[0]
    inputs = cast(_OutcomeInput, call["input"])
    expected_output = cast(_RunOutput, call["expected_output"])
    assert call["id"] == str(reviewer_outcomes._run_example_id("run_1", "slack_thumbs_down"))
    assert inputs == {"repo": "o/r", "run_id": "run_1", "channel_id": "C1"}
    assert expected_output == {
        "label": FALSE_POSITIVE,
        "label_source": "slack_thumbs_down",
    }


async def test_outcome_write_is_best_effort_without_langfuse(monkeypatch) -> None:  # noqa: ANN001
    _patch_client(monkeypatch, None)
    assert not await upsert_finding_outcome(
        _finding(), label=FALSE_POSITIVE, label_source="dismissed", repo="o/r"
    )
