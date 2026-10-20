"""Persist reviewer finding outcomes to a Langfuse dataset.

Writes are best-effort and deterministic (one item id per
``finding_id`` + ``label_source``) so re-processing a transition updates the
same item instead of creating duplicates.
"""

import asyncio
import logging
import uuid
from typing import Any

from langfuse import Langfuse
from langfuse.api import NotFoundError

from openswe.config import ENV
from openswe.review.findings import Finding
from openswe.run_config import RunConfig
from openswe.utils.langfuse import get_langfuse_client
from openswe.utils.typed_json import JsonObject

logger = logging.getLogger(__name__)

OUTCOMES_DATASET_NAME = ENV.LANGFUSE_REVIEWER_OUTCOMES_DATASET.get()
_DEFAULT_OUTCOMES_DATASET_NAME = "openswe-reviewer-outcomes"
TRUE_POSITIVE = "true_positive"
FALSE_POSITIVE = "false_positive"
_DIFF_HUNK_MAX_CHARS = 4000


def outcome_from_status(
    status: str,
    *,
    first_seen_sha: str | None,
    head_sha: str | None,
) -> tuple[str, str] | None:
    """Map a finding status transition to ``(label, label_source)``."""
    if status == "resolved":
        if head_sha and first_seen_sha and head_sha != first_seen_sha:
            return TRUE_POSITIVE, "resolved_by_commit"
        return TRUE_POSITIVE, "resolved_same_sha"
    if status == "dismissed":
        return FALSE_POSITIVE, "dismissed"
    return None


def outcome_from_score(score: float | None, *, source: str) -> tuple[str, str] | None:
    """Map a reaction score (1.0 👍 / 0.0 👎) to ``(label, label_source)``."""
    if score is None:
        return None
    if score >= 1.0:
        return TRUE_POSITIVE, f"{source}_thumbs_up"
    return FALSE_POSITIVE, f"{source}_thumbs_down"


def _example_id(repo: str, finding_id: str, label_source: str) -> uuid.UUID:
    dataset_scope = (
        ""
        if OUTCOMES_DATASET_NAME == _DEFAULT_OUTCOMES_DATASET_NAME
        else f"{OUTCOMES_DATASET_NAME}:"
    )
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{dataset_scope}finding-outcome:{repo}:{finding_id}:{label_source}",
    )


def _run_example_id(run_id: str, label_source: str) -> uuid.UUID:
    dataset_scope = (
        ""
        if OUTCOMES_DATASET_NAME == _DEFAULT_OUTCOMES_DATASET_NAME
        else f"{OUTCOMES_DATASET_NAME}:"
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_scope}run-outcome:{run_id}:{label_source}")


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit]
    return value


def _ensure_dataset(client: Langfuse) -> None:
    try:
        client.api.datasets.get(dataset_name=OUTCOMES_DATASET_NAME)
    except NotFoundError:
        try:
            client.create_dataset(
                name=OUTCOMES_DATASET_NAME,
                description=(
                    "Open SWE reviewer finding outcomes (resolved / dismissed / 👍👎) "
                    "captured in production."
                ),
            )
        except Exception:
            # Another process may have created it after our lookup.
            client.api.datasets.get(dataset_name=OUTCOMES_DATASET_NAME)


def _upsert_item(
    client: Langfuse,
    *,
    item_id: uuid.UUID,
    inputs: JsonObject,
    outputs: JsonObject,
    metadata: JsonObject,
) -> None:
    _ensure_dataset(client)
    client.create_dataset_item(
        dataset_name=OUTCOMES_DATASET_NAME,
        id=str(item_id),
        input=inputs,
        expected_output=outputs,
        metadata=metadata,
    )


async def upsert_finding_outcome(
    finding: Finding,
    *,
    label: str,
    label_source: str,
    repo: str,
    pr_number: int | None = None,
    pr_url: str = "",
    base_sha: str = "",
    head_sha: str = "",
    run_id: str | None = None,
    thread_id: str | None = None,
) -> bool:
    """Upsert one finding-level outcome example. Best-effort; never raises."""
    finding_id = str(finding.get("id") or "")
    if not finding_id or not repo:
        return False
    client = get_langfuse_client()
    if client is None:
        logger.debug("No Langfuse client configured; skipping reviewer outcome")
        return False
    try:
        inputs: JsonObject = {
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "file": finding.get("file"),
            "start_line": finding.get("start_line"),
            "end_line": finding.get("end_line"),
            "side": finding.get("side", "RIGHT"),
            "diff_hunk": _truncate(finding.get("diff_hunk"), _DIFF_HUNK_MAX_CHARS),
            "base_sha": base_sha,
            "head_sha": head_sha,
        }
        outputs: JsonObject = {
            "label": label,
            "label_source": label_source,
            "finding": {
                "title": finding.get("title"),
                "description": _truncate(finding.get("description"), _DIFF_HUNK_MAX_CHARS),
                "severity": finding.get("severity"),
                "confidence": finding.get("confidence"),
                "category": finding.get("category"),
            },
            "resolution_note": finding.get("resolution_note"),
        }
        metadata: JsonObject = {
            "granularity": "finding",
            "repo": repo,
            "finding_id": finding_id,
            "label": label,
            "label_source": label_source,
            "run_id": run_id or finding.get("github_review_run_id"),
            "thread_id": thread_id,
            "first_seen_sha": finding.get("first_seen_sha"),
        }
        await asyncio.to_thread(
            _upsert_item,
            client,
            item_id=_example_id(repo, finding_id, label_source),
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        return True
    except Exception:  # noqa: BLE001 — dataset writes must never break a review
        logger.exception("Failed to upsert reviewer finding outcome for %s", finding_id)
        return False


async def upsert_run_outcome(
    *,
    label: str,
    label_source: str,
    run_id: str,
    repo: str | None = None,
    extra: JsonObject | None = None,
) -> bool:
    """Upsert a coarse run-level outcome such as a Slack 👍/👎 rating."""
    if not run_id:
        return False
    client = get_langfuse_client()
    if client is None:
        return False
    try:
        inputs: JsonObject = {"repo": repo, "run_id": run_id, **(extra or {})}
        outputs: JsonObject = {"label": label, "label_source": label_source}
        metadata: JsonObject = {
            "granularity": "run",
            "repo": repo,
            "run_id": run_id,
            "label": label,
            "label_source": label_source,
        }
        await asyncio.to_thread(
            _upsert_item,
            client,
            item_id=_run_example_id(run_id, label_source),
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Failed to upsert run outcome for run %s", run_id)
        return False


async def emit_finding_status_outcome(
    finding: Finding,
    status: str,
    *,
    cfg: RunConfig,
    thread_id: str | None = None,
) -> bool:
    """Map a resolve/dismiss transition and upsert its outcome example."""
    head_sha = cfg.head_sha or ""
    mapping = outcome_from_status(
        status, first_seen_sha=finding.get("first_seen_sha"), head_sha=head_sha
    )
    if mapping is None or not cfg.repo_full_name:
        return False
    label, label_source = mapping
    return await upsert_finding_outcome(
        finding,
        label=label,
        label_source=label_source,
        repo=cfg.repo_full_name,
        pr_number=cfg.pr_number,
        pr_url=cfg.pr_url or "",
        base_sha=cfg.base_sha or "",
        head_sha=head_sha,
        thread_id=thread_id,
    )
