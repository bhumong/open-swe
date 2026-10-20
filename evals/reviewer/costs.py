"""Per-PR and per-run LLM cost for a Langfuse reviewer experiment."""

import asyncio
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from langfuse import Evaluation
from langfuse.experiment import ExperimentItemResult

from openswe.utils.langfuse import LangfuseCostUnavailable, LangfuseThreadCost, get_trace_cost

logger = logging.getLogger(__name__)

_RETRY_DELAYS_SECONDS = (15, 45, 90)


@dataclass(frozen=True)
class PrCost:
    run_id: str
    pr_url: str
    thread_id: str
    cost_usd: float
    total_tokens: int | None


@dataclass(frozen=True)
class CostSummary:
    priced: list[PrCost]
    unpriced: list[str]

    @property
    def total_usd(self) -> float:
        return sum(item.cost_usd for item in self.priced)

    @property
    def mean_usd(self) -> float:
        return self.total_usd / len(self.priced) if self.priced else 0.0

    @property
    def median_usd(self) -> float:
        return statistics.median(item.cost_usd for item in self.priced) if self.priced else 0.0

    @property
    def max_usd(self) -> float:
        return max((item.cost_usd for item in self.priced), default=0.0)

    @property
    def total_tokens(self) -> int:
        return sum(item.total_tokens or 0 for item in self.priced)


def _field(value: object | None, key: str) -> object | None:
    if isinstance(value, Mapping):
        return cast(object | None, value.get(key))
    return cast(object | None, getattr(value, key, None))


async def _thread_cost(
    thread_id: str, invocation_id: str, lookup_start: str | None
) -> LangfuseThreadCost | None:
    for delay in (0, *_RETRY_DELAYS_SECONDS):
        if delay:
            await asyncio.sleep(delay)
        try:
            snapshot = await get_trace_cost(
                thread_id,
                invocation_id,
                run_only=True,
                **({"lookup_start": lookup_start} if lookup_start else {}),
            )
        except LangfuseCostUnavailable:
            logger.info("Langfuse cost is unavailable for reviewer thread %s", thread_id)
            return None
        except Exception:  # noqa: BLE001 — pricing must not stop evaluation progress
            logger.warning(
                "Could not read reviewer trace cost",
                extra={"reviewer_thread_id": thread_id},
                exc_info=True,
            )
            continue
        if snapshot is not None:
            return snapshot
    logger.warning("Reviewer thread has no priced trace", extra={"thread_id": thread_id})
    return None


async def evaluate_pr_cost(*, output: object, **_: object) -> list[Evaluation]:
    """Return cost/token scores when the target's Langfuse trace is fully priced."""
    thread_id = _field(output, "thread_id")
    invocation_id = _field(output, "invocation_id")
    pr_url = _field(output, "pr_url")
    if not isinstance(thread_id, str) or not thread_id or not isinstance(invocation_id, str):
        logger.warning("Reviewer output is missing trace correlation", extra={"pr_url": pr_url})
        return [
            Evaluation(
                name="cost_status",
                value="unpriced",
                data_type="CATEGORICAL",
                comment="The reviewer output has no thread and invocation id.",
            )
        ]
    started_at = _field(output, "invocation_started_at")
    snapshot = await _thread_cost(
        thread_id,
        invocation_id,
        started_at if isinstance(started_at, str) else None,
    )
    if snapshot is None:
        return [
            Evaluation(
                name="cost_status",
                value="unpriced",
                data_type="CATEGORICAL",
                comment="Langfuse has not returned a complete cost for this reviewer trace.",
            )
        ]
    evaluations = [
        Evaluation(name="cost_status", value="priced", data_type="CATEGORICAL"),
        Evaluation(name="cost_usd", value=snapshot.total_cost, data_type="NUMERIC"),
    ]
    if snapshot.total_tokens is not None:
        evaluations.append(
            Evaluation(name="total_tokens", value=snapshot.total_tokens, data_type="NUMERIC")
        )
    return evaluations


def _evaluations(result: object) -> dict[str, object | None]:
    values: dict[str, object | None] = {}
    evaluations = _field(result, "evaluations")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, str | bytes):
        return values
    for evaluation in evaluations:
        name = _field(evaluation, "name")
        if isinstance(name, str):
            values[name] = _field(evaluation, "value")
    return values


def summarize_experiment_costs(item_results: Sequence[object]) -> CostSummary:
    """Summarize already-recorded evaluator scores without issuing more API calls."""
    priced: list[PrCost] = []
    unpriced: list[str] = []
    for result in item_results:
        output = _field(result, "output")
        item = _field(result, "item")
        raw_inputs = _field(item, "input")
        inputs: Mapping[str, object] = (
            cast(Mapping[str, object], raw_inputs) if isinstance(raw_inputs, Mapping) else {}
        )
        scores = _evaluations(result)
        pr_url_value = inputs.get("pr_url") or _field(output, "pr_url")
        pr_url = str(pr_url_value or "unknown")
        cost_value = scores.get("cost_usd")
        if scores.get("cost_status") != "priced" or not isinstance(cost_value, (float, int)):
            unpriced.append(pr_url)
            continue
        thread_id = _field(output, "thread_id")
        invocation_id = _field(output, "invocation_id")
        cost = float(cost_value)
        if isinstance(cost_value, bool) or not math_is_finite_nonnegative(cost):
            unpriced.append(pr_url)
            continue
        total_tokens_value = scores.get("total_tokens")
        priced.append(
            PrCost(
                run_id=str(invocation_id or ""),
                pr_url=pr_url,
                thread_id=str(thread_id or ""),
                cost_usd=cost,
                total_tokens=(
                    int(total_tokens_value)
                    if isinstance(total_tokens_value, int)
                    and not isinstance(total_tokens_value, bool)
                    else None
                ),
            )
        )
    return CostSummary(priced=priced, unpriced=unpriced)


def math_is_finite_nonnegative(value: float) -> bool:
    return value >= 0 and value != float("inf") and value == value


def record_experiment_costs(
    *, item_results: Sequence[ExperimentItemResult], **_: object
) -> list[Evaluation]:
    """Attach aggregate cost scores to the Langfuse dataset run."""
    summary = summarize_experiment_costs(item_results)
    return [
        Evaluation(name="cost_total_usd", value=summary.total_usd, data_type="NUMERIC"),
        Evaluation(name="cost_mean_per_pr_usd", value=summary.mean_usd, data_type="NUMERIC"),
        Evaluation(name="cost_median_per_pr_usd", value=summary.median_usd, data_type="NUMERIC"),
        Evaluation(name="cost_max_per_pr_usd", value=summary.max_usd, data_type="NUMERIC"),
        Evaluation(name="cost_total_tokens", value=summary.total_tokens, data_type="NUMERIC"),
        Evaluation(name="cost_priced_prs", value=len(summary.priced), data_type="NUMERIC"),
        Evaluation(
            name="cost_unpriced_prs",
            value=len(summary.unpriced),
            data_type="NUMERIC",
            comment=", ".join(summary.unpriced) or None,
        ),
    ]


def print_summary(summary: CostSummary) -> None:
    print("\nReviewer evaluation cost")
    for item in sorted(summary.priced, key=lambda item: item.cost_usd, reverse=True):
        tokens = f"{item.total_tokens:,} tok" if item.total_tokens is not None else "tokens n/a"
        print(f"  ${item.cost_usd:8.4f}  {tokens:>14}  {item.pr_url}")
    for pr_url in summary.unpriced:
        print(f"  {'unpriced':>9}  {'':>14}  {pr_url}")
    print(
        f"  total ${summary.total_usd:.2f} over {len(summary.priced)} PRs | "
        f"mean ${summary.mean_usd:.4f} | median ${summary.median_usd:.4f} | "
        f"max ${summary.max_usd:.4f} | {summary.total_tokens:,} tokens"
    )
