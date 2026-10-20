from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openswe.utils import langfuse
from openswe.utils.langfuse import parse_langfuse_locator


@pytest.fixture(autouse=True)
def _langfuse_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")


@pytest.mark.parametrize(
    "locator",
    [
        "trace-id",
        "https://evil.example/project/project-id/traces/trace-id",
        "https://user@langfuse.example/project/project-id/traces/trace-id",
        "ftp://langfuse.example/project/project-id/traces/trace-id",
        "https://langfuse.example/project/project-id/traces/trace-id/extra",
        "https://langfuse.example/project/project-id/trace/trace-id",
        "https://langfuse.example/project/project-id/traces/%2f%2fevil",
    ],
)
def test_parse_langfuse_locator_rejects_invalid_trace_urls(locator: str) -> None:
    assert parse_langfuse_locator(locator) is None


def test_parse_langfuse_locator_accepts_only_configured_origin() -> None:
    locator = parse_langfuse_locator(
        "<https://langfuse.example/project/project-id/traces/trace-id|Open trace>"
    )

    assert locator == langfuse.LangfuseLocator(trace_id="trace-id")


async def test_get_open_swe_thread_id_from_langfuse_resolves_trace_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = SimpleNamespace(session_id="thread-1")
    client = SimpleNamespace(
        async_api=SimpleNamespace(trace=SimpleNamespace(get=AsyncMock(return_value=trace)))
    )
    monkeypatch.setattr(langfuse, "get_langfuse_client", lambda: client)

    result = await langfuse.get_open_swe_thread_id_from_langfuse(
        "https://langfuse.example/project/project-id/traces/trace-id"
    )

    assert result == "thread-1"
    client.async_api.trace.get.assert_awaited_once_with(trace_id="trace-id")


def test_external_trace_url_omits_loopback_hosts() -> None:
    assert langfuse.external_trace_url("http://localhost:3030/project/p/traces/t") is None
    assert langfuse.external_trace_url("http://127.0.0.1:3030/project/p/traces/t") is None
    assert langfuse.external_trace_url("https://langfuse.example/project/p/traces/t") == (
        "https://langfuse.example/project/p/traces/t"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://user@langfuse.example/project/p/traces/t",
        "https://user:secret@langfuse.example/project/p/traces/t",
    ],
)
def test_external_trace_url_rejects_embedded_credentials(url: str) -> None:
    assert langfuse.external_trace_url(url) is None


async def test_thread_trace_url_resolves_run_before_building_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value="trace-id"))
    make_url = Mock(return_value="https://langfuse.example/project/project-id/traces/trace-id")
    monkeypatch.setattr(langfuse, "trace_url", make_url)

    result = await langfuse.get_thread_trace_url("thread-1", "run-1")

    assert result == "https://langfuse.example/project/project-id/traces/trace-id"
    langfuse.resolve_trace_id.assert_awaited_once_with("thread-1", "run-1", None)
    make_url.assert_called_once_with("trace-id")


async def test_thread_trace_url_hides_loopback_links_from_external_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langfuse, "resolve_trace_id", AsyncMock(return_value="trace-id"))
    monkeypatch.setattr(
        langfuse,
        "trace_url",
        Mock(return_value="http://127.0.0.1:3030/project/p/traces/trace-id"),
    )

    assert await langfuse.get_thread_trace_url("thread-1", external=True) is None
