from importlib import import_module
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from openswe.utils import langfuse


def test_graph_config_adds_correlation_and_preserves_existing_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = object()
    monkeypatch.setattr(langfuse, "create_langfuse_callback", lambda: handler)
    existing_callback = object()
    config = {
        "run_id": "langgraph-run-1",
        "configurable": {"thread_id": "thread-1", "invocation_id": "invocation-1"},
        "metadata": {"source": "dashboard"},
        "callbacks": [existing_callback],
    }

    result = langfuse.configure_graph_tracing(config, graph_name="reviewer")

    assert result["metadata"] == {
        "source": "dashboard",
        "conversation.id": "thread-1",
        "langfuse_session_id": "thread-1",
        "openswe.run_id": "invocation-1",
        "langgraph_run_id": "langgraph-run-1",
        "openswe.graph_name": "reviewer",
    }
    assert result["callbacks"] == [existing_callback, handler]
    assert result["run_name"] == "reviewer"


def test_graph_config_keeps_existing_configuration_without_langfuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"configurable": {"thread_id": "thread-1"}, "metadata": {"source": "cron"}}
    monkeypatch.setattr(langfuse, "create_langfuse_callback", lambda: None)

    assert langfuse.configure_graph_tracing(config, graph_name="scheduler") == config


def test_graph_config_serializes_uuid_run_id_for_trace_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid4()
    monkeypatch.setattr(langfuse, "create_langfuse_callback", lambda: object())

    result = langfuse.configure_graph_tracing({"run_id": run_id}, graph_name="reviewer")

    assert result["metadata"]["langgraph_run_id"] == str(run_id)


def test_callback_factory_returns_a_new_handler_for_each_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = iter((object(), object()))
    monkeypatch.setattr(langfuse, "get_langfuse_client", Mock(return_value=object()))
    monkeypatch.setattr(langfuse, "_new_callback_handler", lambda: next(handlers))

    assert langfuse.create_langfuse_callback() is not langfuse.create_langfuse_callback()


@pytest.mark.parametrize(
    ("module_name", "factory_name", "graph_name"),
    [
        ("openswe.server", "get_agent", "agent"),
        ("openswe.reviewer", "get_reviewer_agent", "reviewer"),
        ("openswe.chat", "get_chat_agent", "chat"),
        ("openswe.review_scout.graph", "get_review_scout", "review-scout"),
    ],
)
async def test_langgraph_factories_use_the_shared_tracing_config(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    factory_name: str,
    graph_name: str,
) -> None:
    module = import_module(module_name)
    factory = getattr(module, factory_name)
    graph = MagicMock()
    graph.with_config.return_value = graph
    configure = Mock(side_effect=lambda config, *, graph_name: config)
    monkeypatch.setattr(module, "create_deep_agent", Mock(return_value=graph))
    monkeypatch.setattr(module, "configure_graph_tracing", configure)

    result = await factory({"configurable": {}})

    assert result is graph
    configure.assert_called_once()
    assert configure.call_args.kwargs == {"graph_name": graph_name}
    graph.with_config.assert_called_once()


def test_scheduler_factory_uses_the_shared_tracing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from openswe import scheduler

    configure = Mock(side_effect=lambda config, *, graph_name: config)
    monkeypatch.setattr(scheduler, "configure_graph_tracing", configure)

    scheduler.get_scheduler({"metadata": {"source": "cron"}})

    configure.assert_called_once_with({"metadata": {"source": "cron"}}, graph_name="scheduler")
