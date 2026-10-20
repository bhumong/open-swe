"""Langfuse client and trace-correlation helpers."""

import ipaddress
import json
import logging
import math
import os
import threading
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote, urlsplit

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langfuse import Langfuse, LangfuseSpan
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

_TRACE_RUN_METADATA_KEYS = (
    "openswe.run_id",
    "invocation_id",
    "prepare_run_id",
    "langgraph_run_id",
)

_LANGFUSE_CLIENT: Langfuse | None = None
_LANGFUSE_CLIENT_PUBLIC_KEY: str | None = None
_LANGFUSE_CLIENT_LOCK = threading.Lock()


@dataclass(frozen=True)
class LangfuseLocator:
    trace_id: str


@dataclass(frozen=True)
class LangfuseThreadCost:
    total_cost: float
    total_tokens: int | None
    last_end_time: datetime
    target_end_time: datetime


class LangfuseCostUnavailable(RuntimeError):
    """Langfuse usage cannot currently be queried for this deployment."""


def get_langfuse_client() -> Langfuse | None:
    """Return the configured SDK client, or ``None`` when telemetry is disabled."""
    global _LANGFUSE_CLIENT, _LANGFUSE_CLIENT_PUBLIC_KEY

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        with _LANGFUSE_CLIENT_LOCK:
            if _LANGFUSE_CLIENT is None or _LANGFUSE_CLIENT_PUBLIC_KEY != public_key:
                _LANGFUSE_CLIENT = Langfuse(public_key=public_key, secret_key=secret_key)
                _LANGFUSE_CLIENT_PUBLIC_KEY = public_key
            return _LANGFUSE_CLIENT
    except Exception:  # noqa: BLE001
        logger.warning("Could not initialize Langfuse", exc_info=True)
        return None


def _new_callback_handler() -> CallbackHandler:
    """Construct a callback scoped to one graph factory invocation."""
    return CallbackHandler(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"))


def create_langfuse_callback() -> CallbackHandler | None:
    """Create a fresh handler when Langfuse credentials are configured."""
    if get_langfuse_client() is None:
        return None
    try:
        return _new_callback_handler()
    except Exception:  # noqa: BLE001
        logger.warning("Could not initialize Langfuse graph callback", exc_info=True)
        return None


def configure_graph_tracing(config: RunnableConfig, *, graph_name: str) -> RunnableConfig:
    """Add Langfuse tracing metadata and a per-run callback to a graph config."""
    callback = create_langfuse_callback()
    if callback is None:
        return config

    configurable_value = config.get("configurable")
    configurable = configurable_value if isinstance(configurable_value, Mapping) else {}
    metadata_value = config.get("metadata")
    metadata = (
        {key: value for key, value in metadata_value.items() if isinstance(key, str)}
        if isinstance(metadata_value, Mapping)
        else {}
    )

    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        metadata["conversation.id"] = thread_id
        metadata["langfuse_session_id"] = thread_id

    run_id = next(
        (
            value
            for key in ("invocation_id", "prepare_run_id", "run_id")
            if isinstance((value := configurable.get(key)), str) and value
        ),
        None,
    )
    if run_id is not None:
        metadata["openswe.run_id"] = run_id
    langgraph_run_id = config.get("run_id")
    if isinstance(langgraph_run_id, (str, uuid.UUID)) and langgraph_run_id:
        metadata["langgraph_run_id"] = str(langgraph_run_id)
    metadata["openswe.graph_name"] = graph_name

    return merge_configs(
        config,
        {
            "callbacks": [callback],
            "metadata": metadata,
            "run_name": graph_name,
        },
    )


def trace_url(trace_id: str) -> str | None:
    """Return an SDK-generated trace URL when Langfuse is configured."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001
        logger.debug("Could not build Langfuse trace URL", exc_info=True)
        return None


def _url_origin(value: str) -> str | None:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme.lower()}://{parsed.hostname.lower()}{suffix}"


def parse_langfuse_locator(locator: str) -> LangfuseLocator | None:
    """Parse a trace URL only when it belongs to the configured Langfuse site."""
    value = locator.strip().strip("<>")
    if "|" in value:
        value = value.split("|", 1)[0]
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    configured_origin = _url_origin(os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
    if not configured_origin or _url_origin(value) != configured_origin:
        return None
    segments = parsed.path.split("/")
    if len(segments) != 5 or segments[0] or segments[1] != "project" or segments[3] != "traces":
        return None
    try:
        project_id, trace_id = (unquote(part, errors="strict") for part in segments[2:5:2])
    except UnicodeDecodeError:
        return None
    if not project_id or not trace_id or "/" in project_id or "/" in trace_id:
        return None
    return LangfuseLocator(trace_id=trace_id)


async def get_open_swe_thread_id_from_langfuse(locator: str) -> str | None:
    """Resolve a configured Langfuse trace link to its Open SWE thread id."""
    parsed = parse_langfuse_locator(locator)
    client = get_langfuse_client()
    if parsed is None or client is None:
        return None
    try:
        trace = await client.async_api.trace.get(trace_id=parsed.trace_id)
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve Langfuse trace %s", parsed.trace_id, exc_info=True)
        return None
    thread_id = _field(trace, "session_id")
    return thread_id if isinstance(thread_id, str) and thread_id else None


def external_trace_url(value: str | None) -> str | None:
    """Return a trace URL safe to publish outside this process."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    normalized_host = hostname.lower()
    if normalized_host == "localhost" or normalized_host.endswith(
        (".localhost", ".local", ".internal", ".test")
    ):
        return None
    try:
        if not ipaddress.ip_address(hostname).is_global:
            return None
    except ValueError:
        pass
    return value


async def get_thread_trace_url(
    thread_id: str,
    run_id: str | None = None,
    started_after: datetime | None = None,
    *,
    external: bool = False,
) -> str | None:
    """Resolve a thread/run to its Langfuse trace URL, optionally for an external target."""
    resolved = await resolve_trace_id(thread_id, run_id, started_after)
    if not resolved:
        return None
    url = trace_url(resolved)
    return external_trace_url(url) if external else url


@asynccontextmanager
async def langfuse_observation(
    name: str,
    *,
    input: object = None,
    metadata: Mapping[str, object] | None = None,
) -> AsyncIterator[LangfuseSpan | None]:
    """Open a current span when tracing is configured; export failures stay non-fatal."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        manager = client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input,
            metadata=metadata,
        )
        span = manager.__enter__()
    except Exception:  # noqa: BLE001
        logger.debug("Could not start Langfuse observation %s", name, exc_info=True)
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        try:
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            logger.debug("Could not mark Langfuse observation %s as failed", name, exc_info=True)
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:  # noqa: BLE001
            logger.debug("Could not end Langfuse observation %s", name, exc_info=True)
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.debug("Could not end Langfuse observation %s", name, exc_info=True)


def _field(value: object | None, name: str) -> object | None:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _rows(value: object | None) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


async def _observation_rows(
    client: Langfuse, *, trace_id: str | None = None, session_id: str | None = None
) -> list[object]:
    if trace_id is not None:
        filters = {"trace_id": trace_id}
    elif session_id is not None:
        filters = {"session_id": session_id}
    else:
        raise ValueError("A trace id or session id is required")
    rows: list[object] = []
    cursor = None
    while True:
        response = await client.async_api.observations.get_many(
            **filters,
            fields="core,basic,time,usage",
            limit=1000,
            **({"cursor": cursor} if cursor else {}),
        )
        rows.extend(_rows(_field(response, "data")))
        next_cursor = _field(_field(response, "meta"), "cursor")
        cursor = next_cursor if isinstance(next_cursor, str) else None
        if not cursor or len(rows) >= 20_000:
            return rows


def _parse_time(value: object | None) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def resolve_trace_id(
    thread_id: str,
    run_id: str | None = None,
    started_after: datetime | None = None,
    *,
    fallback_window: timedelta = timedelta(minutes=15),
) -> str | None:
    """Resolve one Langfuse trace for an Open SWE thread and invocation."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        response = await client.async_api.observations.get_many(
            session_id=thread_id,
            is_root_observation=True,
            fields="core,basic,metadata,time,trace_context",
            expand_metadata=",".join(_TRACE_RUN_METADATA_KEYS),
            limit=1000,
        )
        rows = []
        for observation in _rows(_field(response, "data")):
            trace_id = _field(observation, "trace_id")
            if isinstance(trace_id, str) and trace_id:
                rows.append(observation)

        if run_id:
            exact: set[str] = set()
            for observation in rows:
                metadata = _field(observation, "metadata")
                trace_id = _field(observation, "trace_id")
                if (
                    isinstance(metadata, Mapping)
                    and isinstance(trace_id, str)
                    and run_id in tuple(metadata.get(key) for key in _TRACE_RUN_METADATA_KEYS)
                ):
                    exact.add(trace_id)
            if len(exact) == 1:
                return next(iter(exact))
            if len(exact) > 1:
                return None

        fallback_start = _parse_time(started_after) if started_after else None
        end = fallback_start + fallback_window if fallback_start else datetime.now(UTC)
        start = fallback_start or end - fallback_window
        matches: set[str] = set()
        for observation in rows:
            trace_id = _field(observation, "trace_id")
            observation_time = _parse_time(_field(observation, "start_time"))
            if (
                isinstance(trace_id, str)
                and observation_time is not None
                and start <= observation_time <= end
            ):
                matches.add(trace_id)
        if len(matches) == 1:
            return next(iter(matches))
        return None
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve Langfuse trace for thread %s", thread_id, exc_info=True)
        return None


async def resolve_trace_id_for_run(run_id: str) -> str | None:
    """Resolve a root trace by exact Open SWE run metadata when the thread is unknown."""
    client = get_langfuse_client()
    if client is None or not run_id:
        return None
    try:
        matches: set[str] = set()
        for key in _TRACE_RUN_METADATA_KEYS:
            response = await client.async_api.observations.get_many(
                fields="core,basic,metadata,time",
                expand_metadata=key,
                limit=1000,
                filter=json.dumps(
                    [
                        {
                            "type": "boolean",
                            "column": "isRootObservation",
                            "operator": "=",
                            "value": True,
                        },
                        {
                            "type": "stringObject",
                            "column": "metadata",
                            "key": key,
                            "operator": "=",
                            "value": run_id,
                        },
                    ]
                ),
            )
            for observation in _rows(_field(response, "data")):
                if _field(observation, "is_root_observation") is not True:
                    continue
                trace_id = _field(observation, "trace_id")
                if isinstance(trace_id, str) and trace_id:
                    matches.add(trace_id)
        return next(iter(matches)) if len(matches) == 1 else None
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve Langfuse trace for run %s", run_id, exc_info=True)
        return None


def _http_status(exc: BaseException) -> int | None:
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        status = getattr(cause, "status_code", None) or getattr(
            getattr(cause, "response", None), "status_code", None
        )
        if isinstance(status, int):
            return status
        cause = cause.__cause__
    return None


async def get_trace_cost(
    thread_id: str,
    invocation_id: str,
    *,
    run_only: bool = False,
    lookup_start: datetime | str | None = None,
) -> LangfuseThreadCost | None:
    """Read complete generation costs for one run or its full Langfuse session."""
    client = get_langfuse_client()
    if client is None:
        raise LangfuseCostUnavailable("Langfuse credentials are not configured")
    trace_id = await resolve_trace_id(
        thread_id,
        invocation_id,
        _parse_time(lookup_start) if lookup_start else None,
    )
    if not trace_id:
        return None
    try:
        rows = (
            await _observation_rows(client, trace_id=trace_id)
            if run_only
            else await _observation_rows(client, session_id=thread_id)
        )
    except Exception as exc:  # noqa: BLE001
        status_code = _http_status(exc)
        if (
            isinstance(status_code, int)
            and 400 <= status_code < 500
            and status_code
            not in {
                408,
                429,
            }
        ):
            raise LangfuseCostUnavailable(
                f"Langfuse cost lookup rejected (HTTP {status_code})"
            ) from exc
        logger.warning(
            "Langfuse cost lookup failed",
            extra={"cost_error_type": type(exc).__name__, "cost_http_status": status_code},
        )
        return None

    target_end_times = [
        end_time
        for observation in rows
        if _field(observation, "trace_id") == trace_id
        and _field(observation, "is_root_observation") is True
        and (end_time := _parse_time(_field(observation, "end_time"))) is not None
    ]
    generations = [
        observation
        for observation in rows
        if str(_field(observation, "type") or "").upper() == "GENERATION"
    ]
    if not target_end_times or not generations:
        return None

    generation_costs: list[float] = []
    generation_tokens: list[int] = []
    for observation in generations:
        if _parse_time(_field(observation, "end_time")) is None:
            return None
        cost_details = _field(observation, "cost_details")
        raw_cost = (
            cost_details.get("total")
            if isinstance(cost_details, Mapping)
            else _field(observation, "total_cost")
        )
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (str, int, float)):
            return None
        try:
            cost = float(raw_cost)
        except ValueError:
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        generation_costs.append(cost)
        usage_details = _field(observation, "usage_details")
        raw_tokens = usage_details.get("total") if isinstance(usage_details, Mapping) else None
        if isinstance(raw_tokens, int) and not isinstance(raw_tokens, bool) and raw_tokens >= 0:
            generation_tokens.append(raw_tokens)

    end_times = [
        end_time
        for observation in rows
        if (end_time := _parse_time(_field(observation, "end_time"))) is not None
    ]
    target_end_time = max(target_end_times)
    last_end_time = max(end_times) if end_times else None
    if last_end_time is None or last_end_time < target_end_time:
        return None
    return LangfuseThreadCost(
        total_cost=sum(generation_costs),
        total_tokens=(
            sum(generation_tokens) if len(generation_tokens) == len(generations) else None
        ),
        last_end_time=last_end_time,
        target_end_time=target_end_time,
    )


def _score_id(scope: str, subject_id: str, name: str) -> str:
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"open-swe-langfuse-score:{scope}:{subject_id}:{name}")
    )


async def _write_trace_score(
    client: Langfuse,
    *,
    scope: str,
    subject_id: str,
    trace_id: str,
    name: str,
    score: float,
    comment: str | None,
    source_info: Mapping[str, object] | None,
    timestamp: datetime | None,
) -> bool:
    score_id = _score_id(scope, subject_id, name)
    existing_timestamp = None
    try:
        existing = await client.async_api.scores.get_by_id(score_id)
        existing_timestamp = _parse_time(_field(existing, "timestamp"))
    except Exception:  # noqa: BLE001
        pass
    stable_timestamp = existing_timestamp or (_parse_time(timestamp) if timestamp else None)
    try:
        client.create_score(
            name=name,
            value=score,
            trace_id=trace_id,
            score_id=score_id,
            data_type="NUMERIC",
            comment=comment,
            metadata=source_info or {},
            timestamp=stable_timestamp or datetime.now(UTC),
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not write Langfuse score %s", name, exc_info=True)
        return False
    return True


async def create_trace_score(
    run_id: str,
    name: str,
    *,
    score: float,
    comment: str | None = None,
    source_info: Mapping[str, object] | None = None,
    thread_id: str | None = None,
    timestamp: datetime | None = None,
) -> bool:
    """Create or update an idempotent numeric score on a run's trace."""
    client = get_langfuse_client()
    if client is None:
        return False
    trace_id = (
        await resolve_trace_id(thread_id, run_id)
        if thread_id
        else await resolve_trace_id_for_run(run_id)
    )
    if not trace_id:
        logger.debug("Skipping Langfuse score %s without a trace for run %s", name, run_id)
        return False
    return await _write_trace_score(
        client,
        scope="run",
        subject_id=run_id,
        trace_id=trace_id,
        name=name,
        score=score,
        comment=comment,
        source_info=source_info,
        timestamp=timestamp,
    )


async def create_thread_trace_score(
    thread_id: str,
    name: str,
    *,
    score: float | None,
    comment: str | None = None,
    source_info: Mapping[str, object] | None = None,
    run_id: str | None = None,
    timestamp: datetime | None = None,
) -> bool:
    """Create or update a thread feedback score on its associated trace."""
    if score is None:
        return await delete_thread_trace_score(thread_id, name)
    client = get_langfuse_client()
    if client is None:
        return False
    trace_id = await resolve_trace_id(thread_id, run_id)
    if not trace_id:
        logger.debug("Skipping Langfuse score %s without a trace for thread %s", name, thread_id)
        return False
    return await _write_trace_score(
        client,
        scope="thread",
        subject_id=thread_id,
        trace_id=trace_id,
        name=name,
        score=score,
        comment=comment,
        source_info=source_info,
        timestamp=timestamp,
    )


async def _delete_trace_score(score_id: str) -> bool:
    client = get_langfuse_client()
    if client is None:
        return False
    try:
        await client.async_api.legacy.score_v1.delete(score_id=score_id)
        return True
    except Exception as exc:  # noqa: BLE001
        if _http_status(exc) == 404:
            return True
        logger.warning("Could not delete Langfuse score", exc_info=True)
        return False


async def delete_trace_score(run_id: str, name: str) -> bool:
    """Delete a run score by its deterministic Langfuse score id."""
    return await _delete_trace_score(_score_id("run", run_id, name))


async def delete_thread_trace_score(thread_id: str, name: str) -> bool:
    """Delete a thread score by its deterministic Langfuse score id."""
    return await _delete_trace_score(_score_id("thread", thread_id, name))
