from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

publish_review = import_module("openswe.tools.publish_review")


async def test_github_review_trace_link_uses_external_url_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        publish_review,
        "get_workspace_settings",
        AsyncMock(return_value=SimpleNamespace(review_trace_links_enabled=True)),
    )
    get_trace_url = AsyncMock(return_value=None)
    monkeypatch.setattr(publish_review, "get_thread_trace_url", get_trace_url)

    assert await publish_review._resolve_review_trace_url("thread-1", None) is None
    get_trace_url.assert_awaited_once_with("thread-1", external=True)
