"""Launch and track Langfuse reviewer evaluations for the admin dashboard."""

import asyncio
import base64
import logging
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit

from openswe.config import ENV
from openswe.dashboard.dev_login import dev_login_enabled
from openswe.review.eval_store import (
    EVALS_NAMESPACE,
    HEARTBEAT_STALE_SECONDS,
    LOG_TAIL_CHARS,
    REVIEWER_EVAL_KEY,
)
from openswe.store import get_value, now_iso, put_value
from openswe.utils.build_info import backend_build_info
from openswe.utils.gateway import gateway_base_url
from openswe.utils.typed_json import JsonObject

logger = logging.getLogger(__name__)

EvalStatus = Literal["idle", "starting", "running", "completed", "failed"]
ScoreMode = Literal["all_findings", "surfaced_findings"]
Severity = Literal["low", "medium", "high", "critical"]


class ReviewerEvalConfig(TypedDict):
    dataset_name: str
    experiment_prefix: str
    max_concurrency: int
    langfuse_project: str
    langgraph_url: str
    assistant_id: str
    model_id: str
    reasoning_effort: str
    score_mode: ScoreMode
    severity_threshold: Severity


ACTIVE_STATUSES: frozenset[str] = frozenset({"starting", "running"})
# A launched sandbox clones and syncs before the harness writes its first heartbeat.
STARTING_STALE_SECONDS = 15 * 60
EVAL_TIMEOUT_SECONDS = 8 * 60 * 60
DELETE_AFTER_STOP_SECONDS = 7 * 24 * 60 * 60
REPO_URL = "https://github.com/langchain-ai/open-swe"
KEY_PLACEHOLDER = "sandbox-proxy-injected"
LOG_PATH = "/root/reviewer-eval.log"
_HARNESS_ARGS = (
    "dataset_name",
    "experiment_prefix",
    "max_concurrency",
    "langfuse_project",
    "assistant_id",
    "model_id",
    "reasoning_effort",
    "score_mode",
    "severity_threshold",
)
_LOCAL_TASKS: set[asyncio.Task[None]] = set()

DEFAULT_REVIEWER_EVAL_CONFIG: ReviewerEvalConfig = {
    "dataset_name": "openswe-reviewer-v2",
    "experiment_prefix": "openswe-review-confidence",
    "max_concurrency": 5,
    "langfuse_project": "Open SWE",
    "langgraph_url": "",
    "assistant_id": "reviewer",
    "model_id": "anthropic:claude-opus-5-5",
    "reasoning_effort": "high",
    "score_mode": "surfaced_findings",
    "severity_threshold": "low",
}


def _resolve_langgraph_url() -> str | None:
    return ENV.LANGGRAPH_URL.optional()


def _langfuse_project() -> str:
    return ENV.LANGFUSE_PROJECT_NAME.get("Open SWE")


def resolve_eval_config(config: ReviewerEvalConfig | None = None) -> ReviewerEvalConfig:
    resolved: ReviewerEvalConfig = {
        **DEFAULT_REVIEWER_EVAL_CONFIG,
        "langfuse_project": _langfuse_project(),
        "langgraph_url": _resolve_langgraph_url() or "",
    }
    if config is not None:
        resolved.update(config)
    return resolved


def _idle_record() -> dict[str, Any]:
    config = resolve_eval_config()
    return {
        "name": REVIEWER_EVAL_KEY,
        "status": "idle",
        "run_name": config["experiment_prefix"],
        "langfuse_project": config["langfuse_project"],
        "limit": None,
        "config_snapshot": config,
        "started_at": None,
        "finished_at": None,
        "created_by": None,
        "pid": None,
        "exit_code": None,
        "experiment_url": None,
        "error": None,
        "log_tail": None,
        "worker_id": None,
        "heartbeat": None,
        "progress": None,
        "trigger": None,
        "updated_at": now_iso(),
    }


async def _get_record() -> dict[str, Any] | None:
    return await get_value(EVALS_NAMESPACE, REVIEWER_EVAL_KEY)


async def _put_record(record: dict[str, Any]) -> dict[str, Any]:
    record = {**record, "updated_at": now_iso()}
    await put_value(EVALS_NAMESPACE, REVIEWER_EVAL_KEY, record)
    return record


def _heartbeat_age_seconds(record: dict[str, Any]) -> float | None:
    """Seconds since the record's heartbeat, or ``None`` if absent/unparseable."""
    hb = record.get("heartbeat")
    if not isinstance(hb, str) or not hb:
        return None
    try:
        ts = datetime.fromisoformat(hb)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).total_seconds()


def _is_heartbeat_fresh(record: dict[str, Any]) -> bool:
    age = _heartbeat_age_seconds(record)
    limit = (
        STARTING_STALE_SECONDS if record.get("status") == "starting" else HEARTBEAT_STALE_SECONDS
    )
    return age is not None and age <= limit


async def get_reviewer_eval_status() -> dict[str, Any]:
    """Return the latest reviewer-eval status, reconciling a stale ``running``.

    The eval sandbox refreshes the record's heartbeat while it runs. A poll
    only marks the run failed once the heartbeat is stale, so a healthy run is
    left untouched and a stopped sandbox surfaces as ``failed`` within the stale
    threshold.
    """
    record = await _get_record()
    if record is None:
        return _idle_record()
    if "langfuse_project" not in record:
        record = {**record, "langfuse_project": _langfuse_project()}
        record.pop("langsmith_project", None)
    if record.get("status") not in ACTIVE_STATUSES:
        return record
    if _is_heartbeat_fresh(record):
        return record
    return await _fail(record, _stale_error(record))


def _stale_error(record: dict[str, Any]) -> str:
    if record.get("status") == "starting":
        return f"Eval never reported in; see {LOG_PATH} in sandbox {record.get('worker_id')}."
    return "Eval process is no longer tracked (eval sandbox stopped?)."


async def _fail(record: dict[str, Any], error: str) -> dict[str, Any]:
    finished = record.get("finished_at") or now_iso()
    return await _put_record(
        {**record, "status": "failed", "finished_at": finished, "error": error}
    )


def _key_rule(
    name: str, urls: list[str], key: str, *, header_name: str = "x-api-key"
) -> JsonObject:
    return {
        "name": name,
        "match_hosts": [urlsplit(url).netloc for url in urls],
        "headers": [{"name": header_name, "type": "opaque", "value": key}],
    }


def _langfuse_auth_rule(base_url: str, public_key: str, secret_key: str) -> JsonObject:
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return _key_rule(
        "langfuse",
        [base_url],
        f"Basic {credentials}",
        header_name="Authorization",
    )


def _harness_args(config: ReviewerEvalConfig, limit: int | None) -> list[str]:
    args: list[str] = []
    for key in _HARNESS_ARGS:
        args += [f"--{key.replace('_', '-')}", str(config[key])]
    return args + (["--limit", str(limit)] if limit else [])


def _sandbox_command(ref: str, args: list[str], stop_url: str) -> str:
    script = (
        'export PATH="/root/.local/bin:$PATH"; '
        "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        "git init -q /root/open-swe && cd /root/open-swe && "
        f"git fetch -q --depth 1 {REPO_URL} {shlex.quote(ref)} && git checkout -q FETCH_HEAD && "
        f"uv sync --locked && timeout {EVAL_TIMEOUT_SECONDS} "
        f"uv run python -m evals.reviewer.run_eval {shlex.join(args)}"
    )
    stop = (
        f"curl -fsS -X POST -H 'x-api-key: {KEY_PLACEHOLDER}' "
        f"-H 'content-type: application/json' -d '{{}}' {shlex.quote(stop_url)}"
    )
    return f"({script}) > {LOG_PATH} 2>&1; {stop}"


async def _launch_sandbox(
    config: ReviewerEvalConfig, limit: int | None, created_by: str | None, ref: str
) -> str:
    """Run the harness detached in a sandbox that stops itself when the eval exits.

    API keys never enter the sandbox: its proxy injects them on the wire.
    """
    from openswe.sandboxes.providers.langsmith import get_async_sandbox_client

    api_key = ENV.LANGSMITH_API_KEY.get()
    endpoint = ENV.LANGSMITH_ENDPOINT.get().rstrip("/")
    langgraph_url = config["langgraph_url"] or ENV.LANGGRAPH_URL.get()
    gateway = gateway_base_url()
    gateway_key = ENV.LANGSMITH_GATEWAY_API_KEY.optional() or api_key
    langfuse_base_url = ENV.LANGFUSE_BASE_URL.get("https://cloud.langfuse.com").rstrip("/")
    langfuse_public_key = ENV.LANGFUSE_PUBLIC_KEY.require()
    langfuse_secret_key = ENV.LANGFUSE_SECRET_KEY.require()
    env = {
        "LANGSMITH_API_KEY": KEY_PLACEHOLDER,
        "LANGSMITH_ENDPOINT": endpoint,
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_GATEWAY_BASE_URL": gateway,
        "LANGGRAPH_URL": langgraph_url,
        "LANGFUSE_BASE_URL": langfuse_base_url,
        "LANGFUSE_PUBLIC_KEY": KEY_PLACEHOLDER,
        "LANGFUSE_SECRET_KEY": KEY_PLACEHOLDER,
        "LANGFUSE_PROJECT_NAME": config["langfuse_project"],
        "REVIEWER_EVAL_REPORT_STORE": "1",
        "REVIEWER_EVAL_CREATED_BY": created_by or "",
        "REVIEWER_EVAL_TRIGGER": "sandbox",
    }
    async with get_async_sandbox_client() as client:
        sandbox = await client.create_sandbox(
            idle_ttl_seconds=0,
            delete_after_stop_seconds=DELETE_AFTER_STOP_SECONDS,
            proxy_config={
                "rules": [
                    _key_rule("langsmith", [endpoint, langgraph_url], api_key),
                    _key_rule("gateway", [gateway], gateway_key),
                    _langfuse_auth_rule(
                        langfuse_base_url, langfuse_public_key, langfuse_secret_key
                    ),
                ]
            },
            run_config={"env_vars": env},
            timeout=180,
        )
        stop_url = f"{endpoint}/v2/sandboxes/boxes/{sandbox.name}/stop"
        try:
            await sandbox.run(
                _sandbox_command(ref, _harness_args(config, limit), stop_url),
                timeout=EVAL_TIMEOUT_SECONDS + 3600,
                idle_timeout=-1,
                wait=False,
            )
        except Exception:
            await client.delete_sandbox(sandbox.name)
            raise
    return sandbox.name


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _launch_local(
    config: ReviewerEvalConfig, limit: int | None, created_by: str | None
) -> asyncio.subprocess.Process:
    """Run the evaluator in this local checkout with inherited local credentials."""
    env = os.environ.copy()
    env.update(
        {
            "LANGGRAPH_URL": config["langgraph_url"] or ENV.LANGGRAPH_URL.get(),
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "REVIEWER_EVAL_REPORT_STORE": "1",
            "REVIEWER_EVAL_CREATED_BY": created_by or "",
            "REVIEWER_EVAL_TRIGGER": "local",
            "REVIEWER_EVAL_WORKER_ID": "local-dev",
        }
    )
    args = [
        sys.executable,
        "-m",
        "evals.reviewer.run_eval",
        *_harness_args(config, limit),
    ]
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=_repository_root(),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    return process


async def _monitor_local_eval(process: asyncio.subprocess.Process) -> None:
    """Drain child output and reconcile its final status into the shared store."""
    output_tail = ""
    if process.stdout is not None:
        while line := await process.stdout.readline():
            output_tail = (output_tail + line.decode(errors="replace"))[-LOG_TAIL_CHARS:]
    exit_code = await process.wait()
    record = await _get_record() or _idle_record()
    current_status = record.get("status")
    status = (
        current_status
        if current_status in {"failed", "completed"}
        else ("completed" if exit_code == 0 else "failed")
    )
    error = record.get("error")
    if exit_code != 0 and not error:
        error = f"Local reviewer eval process exited with status {exit_code}."
    await _put_record(
        {
            **record,
            "status": status,
            "pid": process.pid,
            "exit_code": exit_code,
            "finished_at": record.get("finished_at") or now_iso(),
            "heartbeat": now_iso(),
            "log_tail": record.get("log_tail") or output_tail or None,
            "error": error,
            "trigger": "local",
        }
    )


async def start_reviewer_eval(
    config: ReviewerEvalConfig, limit: int | None, created_by: str | None
) -> dict[str, Any]:
    """Launch locally under LangGraph dev, otherwise in the retained sandbox provider."""
    if (await get_reviewer_eval_status()).get("status") in ACTIVE_STATUSES:
        raise RuntimeError("a reviewer eval is already running")
    local = dev_login_enabled()
    ref = None if local else backend_build_info()["commit"]
    if not local and not ref:
        raise RuntimeError("the deployment's commit is unknown, so the eval can't match its code")
    started = now_iso()
    record = await _put_record(
        {
            **_idle_record(),
            "status": "starting",
            "run_name": config["experiment_prefix"],
            "langfuse_project": config["langfuse_project"],
            "limit": limit,
            "config_snapshot": config,
            "started_at": started,
            "created_by": created_by,
            "heartbeat": started,
            "trigger": "local" if local else "sandbox",
        }
    )
    try:
        if local:
            process = await _launch_local(config, limit, created_by)
            record = await _put_record(
                {**record, "worker_id": "local-dev", "pid": process.pid, "trigger": "local"}
            )
            task = asyncio.create_task(_monitor_local_eval(process))
            _LOCAL_TASKS.add(task)
            task.add_done_callback(_LOCAL_TASKS.discard)
            return record
        assert ref is not None
        sandbox_name = await _launch_sandbox(config, limit, created_by, ref)
    except Exception as exc:
        target = "local process" if local else "eval sandbox"
        await _fail(record, f"Couldn't launch the {target}: {exc}")
        raise
    return await _put_record({**record, "worker_id": sandbox_name})
