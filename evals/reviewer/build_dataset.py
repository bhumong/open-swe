"""Synchronize reviewer benchmark goldens into a Langfuse dataset.

Reads ``golden_comments/*.json`` and resolves each PR's base/head SHAs via
``gh``. Stable dataset-item ids make repeated syncs update canonical entries
without deleting unrelated Langfuse items.

Usage:
    uv run python -m evals.reviewer.build_dataset \\
        --dataset-name openswe-reviewer-v2
"""

import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api import NotFoundError

from openswe.utils.langfuse import get_langfuse_client
from openswe.utils.typed_json import JsonObject, JsonValue

GOLDENS_DIR = Path(__file__).parent / "golden_comments"
PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
DATASET_DESCRIPTION = (
    "Open SWE Reviewer baseline — 50 PRs from withmartian/code-review-benchmark goldens."
)
DEFAULT_DATASET_NAME = "openswe-reviewer-v2"


class GoldenEntry(TypedDict):
    url: str
    comments: list[JsonObject]
    pr_title: NotRequired[str]
    original_url: NotRequired[str]
    az_comment: NotRequired[JsonValue]
    _source_file: NotRequired[str]


class PullRequestDetails(TypedDict):
    baseRefOid: str
    headRefOid: str
    baseRefName: str
    headRefName: str
    title: NotRequired[str]
    state: NotRequired[str]


class DatasetInput(TypedDict):
    repo: str
    pr_number: int
    pr_url: str
    original_url: str | None
    pr_title: str | None
    base_sha: str
    head_sha: str
    base_ref: str
    head_ref: str


class DatasetMetadata(TypedDict):
    source: str
    source_file: str
    source_identity: str
    az_comment: JsonValue | None
    pr_state: str | None


class DatasetExample(TypedDict):
    id: str
    input: DatasetInput
    expected_output: dict[str, list[JsonObject]]
    metadata: DatasetMetadata


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(url)
    if not m:
        raise ValueError(f"Not a PR url: {url}")
    owner, repo, num = m.group(1), m.group(2).removesuffix(".git"), int(m.group(3))
    return owner, repo, num


def gh_pr_view(owner: str, repo: str, pr_number: int) -> PullRequestDetails:
    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            f"{owner}/{repo}",
            "--json",
            "baseRefOid,headRefOid,baseRefName,headRefName,title,state,url",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh pr view failed for {owner}/{repo}#{pr_number}: {result.stderr.strip()}"
        )
    return cast(PullRequestDetails, json.loads(result.stdout))


def load_goldens() -> list[GoldenEntry]:
    examples: list[GoldenEntry] = []
    for path in sorted(GOLDENS_DIR.glob("*.json")):
        with path.open() as f:
            entries = cast(list[GoldenEntry], json.load(f))
        for entry in entries:
            entry["_source_file"] = path.stem
            examples.append(entry)
    return examples


def _item_id(dataset_name: str, source_identity: str) -> str:
    dataset_scope = f"{dataset_name}:" if dataset_name != DEFAULT_DATASET_NAME else ""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"open-swe-reviewer-golden:{dataset_scope}{source_identity}",
        )
    )


def build_example(
    entry: GoldenEntry, *, dataset_name: str = DEFAULT_DATASET_NAME
) -> DatasetExample:
    url = entry["url"]
    owner, repo, pr_number = parse_pr_url(url)
    pr = gh_pr_view(owner, repo, pr_number)
    source_file = str(entry.get("_source_file") or "")
    source_identity = f"{owner}/{repo}#{pr_number}:{source_file}"
    inputs: DatasetInput = {
        "repo": f"{owner}/{repo}",
        "pr_number": pr_number,
        "pr_url": url,
        "original_url": entry.get("original_url"),
        "pr_title": entry.get("pr_title") or pr.get("title"),
        "base_sha": pr["baseRefOid"],
        "head_sha": pr["headRefOid"],
        "base_ref": pr["baseRefName"],
        "head_ref": pr["headRefName"],
    }
    expected_output = {"golden_comments": entry["comments"]}
    metadata: DatasetMetadata = {
        "source": "open_swe_golden_comments",
        "source_file": source_file,
        "source_identity": source_identity,
        "az_comment": entry.get("az_comment"),
        "pr_state": pr.get("state"),
    }
    return {
        "id": _item_id(dataset_name, source_identity),
        "input": inputs,
        "expected_output": expected_output,
        "metadata": metadata,
    }


def _ensure_dataset(client: Langfuse, dataset_name: str) -> None:
    try:
        client.api.datasets.get(dataset_name=dataset_name)
    except NotFoundError:
        client.create_dataset(name=dataset_name, description=DATASET_DESCRIPTION)


def upload(dataset_name: str, examples: list[DatasetExample]) -> None:
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse credentials are required to sync the reviewer dataset")
    _ensure_dataset(client, dataset_name)
    for example in examples:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=example["id"],
            input=example["input"],
            expected_output=example["expected_output"],
            metadata=example["metadata"],
        )
    client.flush()
    print(f"Synced {len(examples)} canonical examples to Langfuse dataset {dataset_name}.")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--dry-run", action="store_true", help="Build examples but don't upload.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw = load_goldens()
    if args.limit:
        raw = raw[: args.limit]
    print(f"Loaded {len(raw)} golden entries; resolving SHAs via gh...")

    examples: list[DatasetExample] = []
    for i, entry in enumerate(raw, 1):
        try:
            example = build_example(entry, dataset_name=args.dataset_name)
            examples.append(example)
            print(
                f"  [{i}/{len(raw)}] {example['input']['repo']}#{example['input']['pr_number']} ok"
            )
        except Exception as exc:
            print(f"  [{i}/{len(raw)}] FAILED for {entry.get('url')}: {exc}", file=sys.stderr)

    if args.dry_run:
        out = Path(__file__).parent / "dataset_dryrun.json"
        with out.open("w") as f:
            json.dump(examples, f, indent=2)
        print(f"Dry run: wrote {len(examples)} examples to {out}")
        return

    upload(args.dataset_name, examples)


if __name__ == "__main__":
    main()
