from unittest.mock import AsyncMock, Mock

from openswe.slack import client


async def test_slack_trace_reply_does_not_include_unavailable_trace_url(
    monkeypatch,
) -> None:
    get_trace_url = AsyncMock(return_value=None)
    post_reply = AsyncMock(return_value="1700000000.000001")
    monkeypatch.setattr(client, "get_thread_trace_url", get_trace_url)
    monkeypatch.setattr(client, "post_slack_thread_reply_with_ts", post_reply)
    monkeypatch.setattr(client, "dashboard_thread_url", Mock(return_value=None))

    result = await client.post_slack_trace_reply(
        "C123", "1700000000.000000", "thread-1", include_dashboard_link=False
    )

    assert result == "1700000000.000001"
    get_trace_url.assert_awaited_once_with("thread-1", external=True)
    assert post_reply.await_args.args[2] == ""
