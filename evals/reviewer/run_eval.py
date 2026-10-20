"""Run the reviewer benchmark against a Langfuse dataset.

Usage:
    uv run python -m evals.reviewer.run_eval
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import threading
import tomllib
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api import DatasetItem
from langfuse.experiment import ExperimentItem, ExperimentResult
from langgraph_sdk import get_client

from evals.reviewer import costs, judge
from evals.reviewer.store_reporter import StoreReporter, is_enabled
from evals.reviewer.target import (
    drain_thread_ids,
    get_completed_count,
    get_langgraph_url,
    review_pr,
)
from openswe.review.eval_store import EXPERIMENT_URL_RE, LOG_TAIL_CHARS
from openswe.utils.langfuse import get_langfuse_client
from openswe.utils.typed_json import JsonObject, JsonValue

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.toml")
DEFAULT_LANGFUSE_PROJECT = "Open SWE"
ScoreMode = Literal["all_findings", "surfaced_findings"]
Severity = Literal["low", "medium", "high", "critical"]
_VALID_SCORE_MODES: set[str] = {"all_findings", "surfaced_findings"}
_VALID_SEVERITIES: set[str] = {"low", "medium", "high", "critical"}

_ENV_MAPPING: dict[str, str] = {
    "dataset_name": "REVIEWER_EVAL_DATASET_NAME",
    "experiment_prefix": "REVIEWER_EVAL_EXPERIMENT_PREFIX",
    "max_concurrency": "REVIEWER_EVAL_MAX_CONCURRENCY",
    "langgraph_url": "LANGGRAPH_URL",
    "langfuse_project": "LANGFUSE_PROJECT_NAME",
    "assistant_id": "REVIEWER_ASSISTANT_ID",
    "model_id": "REVIEWER_EVAL_MODEL_ID",
    "reasoning_effort": "REVIEWER_EVAL_REASONING_EFFORT",
    "score_mode": "REVIEWER_EVAL_SCORE_MODE",
    "severity_threshold": "REVIEWER_EVAL_SEVERITY_THRESHOLD",
}


class ReviewerEvalConfig(TypedDict, total=False):
    dataset_name: str
    experiment_prefix: str
    max_concurrency: int
    langgraph_url: str
    langfuse_project: str
    assistant_id: str
    model_id: str
    reasoning_effort: str
    score_mode: ScoreMode
    severity_threshold: Severity


DEFAULT_CONFIG: ReviewerEvalConfig = {
    "dataset_name": "openswe-reviewer-v2",
    "experiment_prefix": "openswe-reviewer-baseline",
    "max_concurrency": 10,
    "langgraph_url": "",
    "langfuse_project": DEFAULT_LANGFUSE_PROJECT,
    "assistant_id": "reviewer",
    "model_id": "anthropic:claude-opus-5-5",
    "reasoning_effort": "high",
    "score_mode": "surfaced_findings",
    "severity_threshold": "low",
}


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_config(raw: dict[str, Any]) -> ReviewerEvalConfig:
    config: ReviewerEvalConfig = {}
    for key in ("dataset_name", "experiment_prefix", "langgraph_url", "langfuse_project"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            config[key] = value  # type: ignore[literal-required]
    for key in ("assistant_id", "model_id", "reasoning_effort"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            config[key] = value  # type: ignore[literal-required]
    max_concurrency = raw.get("max_concurrency")
    if (
        isinstance(max_concurrency, int)
        and not isinstance(max_concurrency, bool)
        and max_concurrency > 0
    ):
        config["max_concurrency"] = max_concurrency
    score_mode = raw.get("score_mode")
    if score_mode in _VALID_SCORE_MODES:
        config["score_mode"] = score_mode
    severity_threshold = raw.get("severity_threshold")
    if severity_threshold in _VALID_SEVERITIES:
        config["severity_threshold"] = severity_threshold
    return config


def _load_config() -> ReviewerEvalConfig:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("rb") as f:
        return _coerce_config(tomllib.load(f))


def _load_env_config(env: Mapping[str, str] = os.environ) -> ReviewerEvalConfig:
    raw: dict[str, Any] = {}
    for config_key, env_key in _ENV_MAPPING.items():
        value = env.get(env_key)
        if value is None or value == "":
            continue
        if config_key == "max_concurrency":
            parsed = _parse_int(value)
            if parsed is not None:
                raw[config_key] = parsed
        else:
            raw[config_key] = value
    return _coerce_config(raw)


def _config_from_args(args: argparse.Namespace) -> ReviewerEvalConfig:
    raw: dict[str, Any] = {}
    for key in _ENV_MAPPING:
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    return _coerce_config(raw)


def _resolve_config(cli_config: ReviewerEvalConfig | None = None) -> ReviewerEvalConfig:
    resolved: ReviewerEvalConfig = {
        **DEFAULT_CONFIG,
        **_load_config(),
        **_load_env_config(),
    }
    if cli_config is not None:
        resolved.update(cli_config)
    return resolved


def _apply_config_to_env(config: ReviewerEvalConfig) -> None:
    for config_key, env_key in _ENV_MAPPING.items():
        value = config.get(config_key)
        if value is not None:
            os.environ[env_key] = str(value)
    os.environ["LANGGRAPH_URL"] = config.get("langgraph_url") or get_langgraph_url()
    # LangGraph Platform may inject automatic LangSmith tracing variables. The
    # eval's graph and experiment traces belong in Langfuse instead.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class _TailCapture:
    """Thread-safe rolling tail of eval output and its Langfuse dataset-run URL."""

    def __init__(self) -> None:
        self._buf = ""
        self._url: str | None = None
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._buf = (self._buf + text)[-LOG_TAIL_CHARS:]
            found = EXPERIMENT_URL_RE.findall(self._buf)
            if found:
                self._url = found[-1].rstrip(".,)")

    def tail(self) -> str | None:
        with self._lock:
            return self._buf or None

    def url(self) -> str | None:
        with self._lock:
            return self._url


class _BufferingHandler(logging.Handler):
    """Mirror log records into a ``_TailCapture`` so the reporter can publish them."""

    def __init__(self, capture: _TailCapture) -> None:
        super().__init__()
        self._capture = capture

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._capture.append(self.format(record) + "\n")
        except Exception:
            pass


class _TeeStream:
    """Write to the original stream and mirror it into the capture."""

    def __init__(self, original: Any, capture: _TailCapture) -> None:
        self._original = original
        self._capture = capture

    def write(self, text: str) -> int:
        self._capture.append(text)
        return self._original.write(text)

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _load_eval_items(client: Langfuse, dataset_name: str, limit: int | None) -> list[DatasetItem]:
    try:
        dataset = client.get_dataset(dataset_name)
    except Exception as exc:  # noqa: BLE001 — dataset control-plane errors must be explicit
        raise RuntimeError(f"Could not load Langfuse dataset {dataset_name}: {exc}") from exc
    items = list(dataset.items)
    if limit:
        items = items[:limit]
    if not items:
        raise RuntimeError(f"Langfuse dataset {dataset_name} has no active items")
    return items


async def _run_item(*, item: ExperimentItem, **_: object) -> JsonObject:
    if isinstance(item, DatasetItem):
        raw_input: object = item.input
    elif isinstance(item, Mapping):
        raw_input = item.get("input")
    else:
        raise ValueError("Langfuse reviewer dataset item must be an object")
    if not isinstance(raw_input, Mapping):
        raise ValueError("Langfuse reviewer dataset item input must be an object")
    inputs: JsonObject = {}
    for key, value in raw_input.items():
        if not isinstance(key, str):
            raise ValueError("Langfuse reviewer dataset item input keys must be strings")
        value_object: object = value
        inputs[key] = typing.cast(JsonValue, value_object)
    return await review_pr(inputs)


def _run_dataset_experiment(
    client: Langfuse, config: ReviewerEvalConfig, items: list[DatasetItem]
) -> ExperimentResult:
    return client.run_experiment(
        name=config["experiment_prefix"],
        description=(
            f"Open SWE reviewer benchmark; score mode={config['score_mode']}, "
            f"severity threshold={config['severity_threshold']}."
        ),
        data=items,
        task=_run_item,
        evaluators=[judge.judge_match, costs.evaluate_pr_cost],
        run_evaluators=[judge.aggregate_pr, costs.record_experiment_costs],
        max_concurrency=config["max_concurrency"],
        metadata={
            "langfuse_project": config["langfuse_project"],
            "dataset_name": config["dataset_name"],
            "model_id": config["model_id"],
            "reasoning_effort": config["reasoning_effort"],
            "score_mode": config["score_mode"],
            "severity_threshold": config["severity_threshold"],
        },
    )


async def _run_dataset_experiment_async(
    client: Langfuse, config: ReviewerEvalConfig, items: list[DatasetItem]
) -> ExperimentResult:
    """Run the SDK's sync adapter off-loop so status heartbeats keep flowing."""
    return await asyncio.to_thread(_run_dataset_experiment, client, config, items)


async def _cleanup_threads(thread_ids: set[str]) -> None:
    """Delete LangGraph checkpoint/metadata records created by this evaluation."""
    client = get_client(url=get_langgraph_url())
    for thread_id in thread_ids:
        try:
            await client.threads.delete(thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete thread %s: %s", thread_id, exc)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("REVIEWER_EVAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N examples.")
    parser.add_argument("--dataset-name", dest="dataset_name")
    parser.add_argument("--experiment-prefix", dest="experiment_prefix")
    parser.add_argument("--max-concurrency", dest="max_concurrency", type=int)
    parser.add_argument("--langgraph-url", dest="langgraph_url")
    parser.add_argument("--langfuse-project", dest="langfuse_project")
    parser.add_argument("--assistant-id", dest="assistant_id")
    parser.add_argument("--model-id", dest="model_id")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort")
    parser.add_argument("--score-mode", dest="score_mode", choices=sorted(_VALID_SCORE_MODES))
    parser.add_argument(
        "--severity-threshold",
        dest="severity_threshold",
        choices=sorted(_VALID_SEVERITIES),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip deleting LangGraph threads after the experiment finishes.",
    )
    args = parser.parse_args()
    config = _resolve_config(_config_from_args(args))
    _apply_config_to_env(config)
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse credentials are required to run the reviewer evaluation")

    dataset_name = config["dataset_name"]
    items = _load_eval_items(client, dataset_name, args.limit)
    logger.info(
        "Starting reviewer eval: dataset=%s items=%s experiment=%s concurrency=%s "
        "model=%s effort=%s score_mode=%s severity_threshold=%s project=%s "
        "assistant_id=%s langgraph_url=%s limit=%s",
        dataset_name,
        len(items),
        config["experiment_prefix"],
        config["max_concurrency"],
        config["model_id"],
        config["reasoning_effort"],
        config["score_mode"],
        config["severity_threshold"],
        config["langfuse_project"],
        config["assistant_id"],
        config["langgraph_url"] or get_langgraph_url(),
        args.limit,
    )

    reporter: StoreReporter | None = None
    heartbeat: asyncio.Task[None] | None = None
    log_handler: logging.Handler | None = None
    original_stdout = sys.stdout
    if is_enabled():
        capture = _TailCapture()
        log_handler = _BufferingHandler(capture)
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(log_handler)
        sys.stdout = _TeeStream(original_stdout, capture)
        reporter = StoreReporter(
            config=dict(config),
            limit=args.limit,
            total=len(items),
            created_by=None,
            completed_getter=get_completed_count,
            tail_getter=capture.tail,
            experiment_url_getter=capture.url,
        )
        await reporter.start()
        heartbeat = reporter.run_heartbeat()

    eval_error: BaseException | None = None
    try:
        result = await _run_dataset_experiment_async(client, config, items)
        print(result.format())
        if result.dataset_run_url:
            print(f"Langfuse experiment URL: {result.dataset_run_url}")
        costs.print_summary(costs.summarize_experiment_costs(result.item_results))
    except BaseException as exc:
        eval_error = exc
        raise
    finally:
        if reporter is not None:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            status = "failed" if eval_error is not None else "completed"
            error = None if eval_error is None else f"{type(eval_error).__name__}: {eval_error}"
            await reporter.finish(status=status, error=error)
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
        sys.stdout = original_stdout
        try:
            client.flush()
        except Exception:  # noqa: BLE001 — flush failures must not hide the eval result
            logger.warning("Could not flush Langfuse client", exc_info=True)
        if not args.no_cleanup:
            thread_ids = drain_thread_ids()
            if thread_ids:
                logger.info("Cleaning up %d LangGraph threads", len(thread_ids))
                await _cleanup_threads(thread_ids)


if __name__ == "__main__":
    asyncio.run(main())
