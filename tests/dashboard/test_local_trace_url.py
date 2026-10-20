from unittest.mock import AsyncMock
from uuid import UUID

from openswe.threads import routes

THREAD_ID = "a743f4f9-7a4b-4f02-aef8-55297febfc30"


async def test_local_trace_url_uses_configured_langfuse_project(monkeypatch) -> None:
    resolve = AsyncMock(return_value="https://langfuse.example/project/p/traces/t")
    monkeypatch.setattr(routes, "get_thread_trace_url", resolve)

    result = await routes.api_get_local_trace_url(UUID(THREAD_ID), _session={"sub": "alice"})

    assert result == {"trace_url": "https://langfuse.example/project/p/traces/t"}
    resolve.assert_awaited_once_with(THREAD_ID)
