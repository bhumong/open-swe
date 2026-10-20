import uuid
from typing import Any

from langfuse.api import NotFoundError

from evals.reviewer import build_dataset


class _FakeDatasets:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.created: list[dict[str, Any]] = []
        self.lookups: list[str] = []

    def get(self, *, dataset_name: str) -> object:
        self.lookups.append(dataset_name)
        if not self.exists:
            raise NotFoundError({}, {})
        return object()


class _FakeApi:
    def __init__(self, exists: bool) -> None:
        self.datasets = _FakeDatasets(exists)


class _FakeClient:
    def __init__(self, exists: bool) -> None:
        self.api = _FakeApi(exists)
        self.created: list[dict[str, Any]] = []
        self.deleted: list[Any] = []

    def create_dataset(self, **kwargs: Any) -> object:
        self.api.datasets.created.append(kwargs)
        return object()

    def create_dataset_item(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()

    def flush(self) -> None:
        return None


def test_build_example_has_stable_source_identity(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        build_dataset,
        "gh_pr_view",
        lambda *_: {
            "baseRefOid": "base",
            "headRefOid": "head",
            "baseRefName": "main",
            "headRefName": "feature",
            "title": "Title",
            "state": "MERGED",
        },
    )
    entry = {
        "url": "https://github.com/org/repo/pull/12",
        "comments": [{"comment": "issue", "severity": "High"}],
        "_source_file": "golden-12",
    }

    first = build_dataset.build_example(entry)
    second = build_dataset.build_example(entry)
    different_dataset = build_dataset.build_example(entry, dataset_name="alternate-reviewer")

    assert first == second
    assert first["id"] != different_dataset["id"]
    assert first["id"] == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "open-swe-reviewer-golden:org/repo#12:golden-12")
    )
    assert first["id"]
    assert first["input"]["repo"] == "org/repo"
    assert first["expected_output"] == {"golden_comments": entry["comments"]}
    assert first["metadata"]["source_file"] == "golden-12"
    assert first["metadata"]["source_identity"] == "org/repo#12:golden-12"


def test_upload_upserts_canonical_items_without_deleting_unrelated_data(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient(exists=True)
    monkeypatch.setattr(build_dataset, "get_langfuse_client", lambda: fake)
    examples = [
        {
            "id": "source-a",
            "input": {"repo": "org/repo", "pr_number": 12},
            "expected_output": {"golden_comments": []},
            "metadata": {"source_identity": "org/repo#12:golden-12"},
        }
    ]

    build_dataset.upload("openswe-reviewer-v2", examples)

    assert fake.api.datasets.created == []
    assert fake.created == [
        {
            "dataset_name": "openswe-reviewer-v2",
            "id": "source-a",
            "input": examples[0]["input"],
            "expected_output": examples[0]["expected_output"],
            "metadata": examples[0]["metadata"],
        }
    ]
    assert fake.deleted == []
    assert fake.api.datasets.lookups == ["openswe-reviewer-v2"]


def test_upload_creates_dataset_when_missing(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient(exists=False)
    monkeypatch.setattr(build_dataset, "get_langfuse_client", lambda: fake)

    build_dataset.upload("openswe-reviewer-v2", [])

    assert fake.api.datasets.created == [
        {
            "name": "openswe-reviewer-v2",
            "description": build_dataset.DATASET_DESCRIPTION,
        }
    ]
