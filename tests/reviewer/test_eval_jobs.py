import asyncio
import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from openswe.review import eval_jobs


def _config() -> eval_jobs.ReviewerEvalConfig:
    return {
        "dataset_name": "openswe-reviewer-v2",
        "experiment_prefix": "baseline",
        "max_concurrency": 4,
        "langfuse_project": "Local Open SWE",
        "langgraph_url": "http://127.0.0.1:2024",
        "assistant_id": "reviewer",
        "model_id": "anthropic:claude-haiku-5-5",
        "reasoning_effort": "high",
        "score_mode": "surfaced_findings",
        "severity_threshold": "low",
    }


async def test_local_launcher_uses_current_python_and_local_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def create_process(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=456)

    monkeypatch.setattr(eval_jobs.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "local-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "local-secret")
    process = await eval_jobs._launch_local(_config(), 3, "admin@example.test")

    args = cast(tuple[object, ...], captured["args"])
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert args[:3] == (sys.executable, "-m", "evals.reviewer.run_eval")
    assert "--langfuse-project" in args
    assert "--limit" in args and "3" in args
    assert kwargs["cwd"] == Path(eval_jobs.__file__).resolve().parents[2]
    env = cast(dict[str, str], kwargs["env"])
    assert env["LANGFUSE_PUBLIC_KEY"] == "local-public"
    assert env["LANGFUSE_SECRET_KEY"] == "local-secret"
    assert env["LANGGRAPH_URL"] == "http://127.0.0.1:2024"
    assert env["LANGSMITH_TRACING"] == "false"
    assert env["REVIEWER_EVAL_REPORT_STORE"] == "1"
    assert env["REVIEWER_EVAL_TRIGGER"] == "local"
    assert env["REVIEWER_EVAL_CREATED_BY"] == "admin@example.test"
    assert process.pid == 456


async def test_local_start_records_progress_and_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_store: dict[str, object] = {}
    finished = asyncio.Event()

    class _Process:
        pid = 321
        stdout = None

        async def wait(self) -> int:
            await finished.wait()
            return 0

    async def put_record(record: dict[str, object]) -> dict[str, object]:
        record_store.update(record)
        return record

    monkeypatch.setattr(
        eval_jobs, "get_reviewer_eval_status", AsyncMock(return_value={"status": "idle"})
    )
    monkeypatch.setattr(eval_jobs, "dev_login_enabled", lambda: True)
    monkeypatch.setattr(eval_jobs, "_put_record", put_record)
    monkeypatch.setattr(eval_jobs, "_get_record", lambda: asyncio.sleep(0, result=record_store))
    monkeypatch.setattr(eval_jobs, "_launch_local", AsyncMock(return_value=_Process()))

    record = await eval_jobs.start_reviewer_eval(_config(), 2, "admin")
    assert record["status"] == "starting"
    assert record["trigger"] == "local"
    assert record["worker_id"] == "local-dev"
    assert record["pid"] == 321
    assert record["langfuse_project"] == "Local Open SWE"
    assert "langsmith_project" not in record

    finished.set()
    tasks = list(eval_jobs._LOCAL_TASKS)
    if tasks:
        await asyncio.gather(*tasks)
    assert record_store["status"] == "completed"
    assert record_store["exit_code"] == 0


async def test_deployed_sandbox_injects_langfuse_auth_only_through_host_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Sandbox:
        name = "eval-box"

        async def run(self, command: str, **kwargs: object) -> None:
            captured["command"] = command
            captured["sandbox_run"] = kwargs

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def create_sandbox(self, **kwargs: object) -> _Sandbox:
            captured.update(kwargs)
            return _Sandbox()

    @asynccontextmanager
    async def sandbox_client():
        yield _Client()

    from openswe.sandboxes.providers import langsmith as sandbox_provider

    monkeypatch.setattr(sandbox_provider, "get_async_sandbox_client", sandbox_client)
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "langfuse-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "langfuse-secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example.test")
    monkeypatch.setenv("LANGGRAPH_URL", "https://graph.example.test")

    await eval_jobs._launch_sandbox(_config(), 2, "admin", "commit-1")

    run_config = cast(dict[str, object], captured["run_config"])
    env = cast(dict[str, str], run_config["env_vars"])
    assert env["LANGFUSE_PUBLIC_KEY"] == eval_jobs.KEY_PLACEHOLDER
    assert env["LANGFUSE_SECRET_KEY"] == eval_jobs.KEY_PLACEHOLDER
    assert env["LANGFUSE_BASE_URL"] == "https://lf.example.test"
    assert env["LANGSMITH_TRACING"] == "false"
    assert "langfuse-secret" not in str(env)
    rules = captured["proxy_config"]["rules"]
    rule = next(rule for rule in rules if rule["name"] == "langfuse")
    assert rule["match_hosts"] == ["lf.example.test"]
    auth = rule["headers"][0]
    assert auth["name"] == "Authorization"
    assert auth["value"] == "Basic " + base64.b64encode(b"langfuse-public:langfuse-secret").decode()
    assert "langfuse-secret" not in cast(str, captured["command"])
