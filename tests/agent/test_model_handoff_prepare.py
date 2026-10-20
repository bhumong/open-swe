from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

import openswe.model_request as model_request
import openswe.server as server
from openswe.dashboard.options import available_requested_models
from openswe.middleware.model_selection import (
    ModelSelectionMiddleware,
    ModelSelectionState,
    SelectedRoute,
)
from openswe.middleware.prepare_run import PrepareRunState
from openswe.model_request import ModelRequestIntent
from openswe.utils.jev import JevDecision
from openswe.utils.thread_settings import ThreadSettings


@dataclass
class Handoff:
    middleware: server.PrepareAgentRunMiddleware
    settings: ThreadSettings
    store: AsyncMock
    record: AsyncMock

    async def prepare(self, state: PrepareRunState | None = None) -> dict[str, object]:
        return await self.middleware._prepare(state or {"messages": []}, MagicMock())


@dataclass
class CapturedObservation:
    name: str
    input: object
    metadata: object
    updates: dict[str, object] = field(default_factory=dict)

    def update(self, **values: object) -> None:
        self.updates.update(values)


@pytest.fixture
def observations(monkeypatch: pytest.MonkeyPatch) -> list[CapturedObservation]:
    captured: list[CapturedObservation] = []

    @asynccontextmanager
    async def observe(
        name: str, *, input: object = None, metadata: object = None
    ) -> AsyncIterator[CapturedObservation]:
        observation = CapturedObservation(name, input, metadata)
        captured.append(observation)
        yield observation

    monkeypatch.setattr(model_request, "langfuse_observation", observe)
    return captured


@pytest.fixture
def handoff(monkeypatch: pytest.MonkeyPatch) -> Handoff:
    settings: ThreadSettings = {"model_id": "openai:gpt-6.1-sol", "repo_instructions": "retain"}

    async def persist(
        client: object, thread_id: str, value: ThreadSettings, *, strict: bool
    ) -> None:
        settings.update(value)

    store = AsyncMock(side_effect=persist)
    record = AsyncMock()
    monkeypatch.setattr(server, "load_thread_settings", AsyncMock(return_value=settings))
    monkeypatch.setattr(server, "store_thread_settings", store)
    monkeypatch.setattr(server, "record_agent_invocation_usage", record)
    monkeypatch.setattr(server, "resolve_github_token", AsyncMock(return_value=("token", None)))
    monkeypatch.setattr(server, "schedule_thread_title_generation", lambda **kw: None)
    monkeypatch.setattr(
        server,
        "get_or_create_sandbox_backend_proxy",
        lambda _: MagicMock(ready=AsyncMock(return_value=MagicMock(id="sandbox-1"))),
    )
    monkeypatch.setattr(server, "resolve_sandbox_work_dir", AsyncMock(return_value="/workspace"))
    for name in (
        "resolve_triggering_user_identity",
        "load_workspace",
        "_resolve_prompt_default_repo",
        "_resolve_user_custom_instructions",
    ):
        monkeypatch.setattr(server, name, AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_thread_participant_identities", AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "construct_system_prompt", lambda *args, **kw: "system prompt")
    monkeypatch.setattr(
        server,
        "client",
        MagicMock(
            threads=MagicMock(
                update=AsyncMock(),
                get=AsyncMock(return_value={"metadata": {"visibility": "public"}}),
            )
        ),
    )
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "source": "slack",
            "invocation_id": "inv-1",
            "slack_thread": {
                "channel_id": "C1",
                "thread_ts": "1.2",
                "triggering_user_id": "U1",
                "triggering_event_ts": "2.0",
            },
        }
    }
    middleware = server.PrepareAgentRunMiddleware(
        thread_id="thread-1",
        config=config,
        profile_login=None,
        repo_instructions=None,
        model_id="openai:gpt-6.1-sol",
        effort=None,
        title_model=MagicMock(),
        source="slack",
        user_email="",
        linear_project_id="",
        linear_issue_number="",
        draft_prs=False,
        recent_thread_context_enabled=False,
        admin_workspaces=False,
    )
    middleware._requested_models = available_requested_models(fable_enabled=False)
    middleware._model_selection = ModelSelectionMiddleware(
        {"fast": MagicMock()},
        MagicMock(),
        routing_mode="fast",
        requested_model_factory=lambda _model, _effort: MagicMock(),
    )
    middleware._routing_defaults = {"fast": ("openai:gpt-6-luna", "low")}
    return Handoff(middleware, settings, store, record)


@pytest.mark.parametrize(
    "requested", ["anthropic:claude-opus-5-5", None, "inference_failure", "low_confidence"]
)
async def test_initial_handoff_persists_before_work_and_attributes_selected_model(
    handoff: Handoff,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[CapturedObservation],
    requested: str | None,
) -> None:
    intent = (
        None
        if requested in {"inference_failure", "low_confidence"}
        else ModelRequestIntent(requested_model=requested)
    )
    outcome = requested
    requested = intent.requested_model if intent else None

    async def infer_intent(*, decision: JevDecision, **kwargs: object) -> ModelRequestIntent | None:
        if outcome == "inference_failure":
            decision.outcome = "classifier_failure"
            decision.reason = "classifier_error"
        else:
            decision.choice = requested or ("no_request" if intent else "anthropic:claude-opus-5-5")
            decision.confidence = 0.4 if outcome == "low_confidence" else 0.95
            decision.outcome = "low_confidence" if outcome == "low_confidence" else "accepted"
            decision.reason = (
                "confidence_below_threshold_or_invalid"
                if outcome == "low_confidence"
                else "confident_choice"
            )
        return intent

    infer = AsyncMock(side_effect=infer_intent)
    monkeypatch.setattr(server, "infer_requested_model", infer)
    prepared = await handoff.prepare()
    decision_span = observations[0]
    assert decision_span.name == "Model selection decision"
    assert decision_span.input == {}
    assert decision_span.metadata == {}
    outputs = cast(dict[str, object], decision_span.updates["output"])
    assert outputs["selected_model_id"] == prepared["selected_model_id"]
    assert outputs["selected_effort"] == prepared["selected_effort"]
    assert outputs["requested_model"] == requested
    assert outputs["pin_persisted"] is bool(requested)
    assert outputs["outcome"] == (
        "accepted_request"
        if requested
        else "classifier_failure"
        if outcome == "inference_failure"
        else "low_confidence"
        if outcome == "low_confidence"
        else "no_request"
    )
    classifier = cast(dict[str, object], outputs["classifier"])
    assert classifier["confidence"] == (
        None if outcome == "inference_failure" else 0.4 if outcome == "low_confidence" else 0.95
    )
    assert handoff.settings["model_handoff_complete"] is True
    assert handoff.settings["repo_instructions"] == "retain"
    assert prepared["selected_model_id"] == (requested or "openai:gpt-6-luna")
    assert prepared["selected_effort"] == ("high" if requested else "low")
    assert prepared["requested_model"] == requested
    assert handoff.record.call_args.kwargs["model_id"] == prepared["selected_model_id"]
    if requested:
        assert handoff.settings["model_id"] == requested
        assert handoff.settings["model_routing_enabled"] is False
    handoff.store.assert_awaited_once_with(server.client, "thread-1", handoff.settings, strict=True)
    assert (await handoff.prepare())["selected_model_id"] == prepared["selected_model_id"]
    reused = cast(dict[str, object], observations[1].updates["output"])
    assert reused["outcome"] == ("reused_saved_choice" if requested else "not_classified")
    reused_classifier = cast(dict[str, object], reused["classifier"])
    assert reused_classifier["outcome"] == "not_run"
    infer.assert_awaited_once()


@pytest.mark.parametrize("previous_route", ["fast", "default"])
@pytest.mark.parametrize("fresh_route", ["fast", "default"])
async def test_explicit_auto_selection_replaces_checkpoint_route(
    handoff: Handoff,
    monkeypatch: pytest.MonkeyPatch,
    previous_route: SelectedRoute,
    fresh_route: SelectedRoute,
) -> None:
    handoff.middleware._config["configurable"].update(
        source="dashboard", model_selection="auto", model_selection_changed=True
    )
    handoff.middleware._requested_models = None
    selection = handoff.middleware._model_selection
    assert selection is not None
    selection._routing_mode = "auto"
    classify = AsyncMock(return_value=fresh_route)
    monkeypatch.setattr("openswe.middleware.model_selection._select_jev_route", classify)
    state: ModelSelectionState = {
        "messages": [HumanMessage(content="Fix the typo")],
        "model_route": previous_route,
        "requested_model": "anthropic:claude-opus-5-5",
    }
    prepared = await handoff.prepare(cast(PrepareRunState, state))
    state.update(cast(ModelSelectionState, prepared))
    assert state["requested_model"] is None
    assert state["model_route"] == fresh_route
    assert prepared["selected_model_id"] == (
        "openai:gpt-6-luna" if fresh_route == "fast" else "openai:gpt-6.1-sol"
    )
    assert (await selection.abefore_model(state, MagicMock()))["model_route"] == fresh_route
    classify.assert_awaited_once_with("Fix the typo")


@pytest.mark.parametrize(
    ("model", "image_type", "failure"),
    [
        ("fireworks:accounts/fireworks/models/kimi-k3", "image", "image"),
        ("fireworks:accounts/fireworks/models/kimi-k3", "image_url", "image"),
        ("fireworks:accounts/fireworks/models/kimi-k3", None, None),
        ("anthropic:claude-opus-5-5", "image", None),
        ("anthropic:claude-fable-5-1", None, "unavailable"),
        ("openai:gpt-6.1-sol", None, "unavailable"),
        ("openai:gpt-6.1-sol", None, "persistence"),
    ],
)
async def test_handoff_validates_and_persists_before_selecting_model(
    handoff: Handoff,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[CapturedObservation],
    image_type: str | None,
    model: str,
    failure: str | None,
) -> None:
    monkeypatch.setattr(
        server,
        "infer_requested_model",
        AsyncMock(
            return_value=ModelRequestIntent(
                requested_model=model,
                unavailable_model=failure == "unavailable" and model == "openai:gpt-6.1-sol",
            )
        ),
    )
    if failure == "persistence":
        handoff.store.side_effect = RuntimeError("write failed")
    content: list[str | dict[str, object]] = [{"type": "text", "text": "Use this model to inspect"}]
    if image_type == "image":
        content.append({"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"})
    elif image_type == "image_url":
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
        )
    state: PrepareRunState = {"messages": [HumanMessage(content=content)]}
    if failure:
        select = MagicMock()
        monkeypatch.setattr(handoff.middleware._model_selection, "use_requested_model", select)
        error = RuntimeError if failure == "persistence" else ValueError
        match = {
            "persistence": "write failed",
            "image": "does not support image input",
            "unavailable": "unavailable",
        }[failure]
        with pytest.raises(error, match=match):
            await handoff.prepare(state)
        decision_span = cast(dict[str, object], observations[0].updates["output"])
        assert decision_span["selected_model_id"] is None
        assert decision_span["pin_persisted"] is False
        assert (
            decision_span["outcome"]
            == {
                "image": "incompatible_request",
                "unavailable": "unavailable_request",
                "persistence": "persistence_failure",
            }[failure]
        )
        select.assert_not_called()
        handoff.record.assert_not_awaited()
        if failure != "persistence":
            handoff.store.assert_not_awaited()
        assert handoff.settings == {"model_id": "openai:gpt-6.1-sol", "repo_instructions": "retain"}
    else:
        assert (await handoff.prepare(state))["selected_model_id"] == model
        handoff.store.assert_awaited_once()
        assert handoff.settings["requested_model"] == model


async def test_later_run_traces_saved_choice_without_classification(
    handoff: Handoff,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[CapturedObservation],
) -> None:
    model = "anthropic:claude-opus-5-5"
    handoff.middleware._requested_models = None
    handoff.middleware._saved_requested_model = model
    handoff.middleware._model_id = model
    handoff.middleware._effort = "high"
    assert handoff.middleware._model_selection is not None
    handoff.middleware._model_selection._routing_mode = None
    infer = AsyncMock()
    monkeypatch.setattr(server, "infer_requested_model", infer)
    metadata = {
        "title_seed": "private opening request",
        "prompt": "private system prompt",
        "credentials": "synthetic-secret",
    }
    prepared = await RunnableLambda(handoff.prepare).ainvoke(
        {"messages": [HumanMessage(content="private conversation")]},
        config={"metadata": metadata},
    )
    assert len(observations) == 1
    decision_span = observations[0]
    assert decision_span.name == "Model selection decision"
    assert decision_span.input == {}
    assert decision_span.metadata == {}
    assert "credentials" not in cast(dict[str, object], decision_span.metadata)
    outputs = cast(dict[str, object], decision_span.updates["output"])
    assert outputs["outcome"] == "reused_saved_choice"
    assert outputs["requested_model"] == model
    assert outputs["selected_model_id"] == prepared["selected_model_id"] == model
    assert outputs["pin_persisted"] is True
    classifier = cast(dict[str, object], outputs["classifier"])
    assert classifier["outcome"] == "not_run"
    infer.assert_not_awaited()
    handoff.store.assert_not_awaited()


@pytest.mark.parametrize(
    ("model", "effort", "invalid"),
    [
        ("anthropic:claude-opus-5-5", "max", False),
        (None, "low", False),
        ("openai:gpt-6.1-sol", "none", False),
        ("openai:gpt-6.1-sol", "max", True),
        (None, "max", True),
    ],
)
async def test_requested_effort_is_validated_persisted_and_reused(
    handoff: Handoff,
    monkeypatch: pytest.MonkeyPatch,
    model: str | None,
    effort: str,
    invalid: bool,
) -> None:
    monkeypatch.setattr(
        server,
        "infer_requested_model",
        AsyncMock(return_value=ModelRequestIntent(requested_model=model, requested_effort=effort)),
    )
    if invalid:
        with pytest.raises(ValueError, match="not supported"):
            await handoff.prepare()
        handoff.store.assert_not_awaited()
        return
    selected_model = model or "openai:gpt-6.1-sol"
    for _ in range(2):
        prepared = await handoff.prepare()
        assert prepared["selected_model_id"] == selected_model
        assert prepared["selected_effort"] == effort
        assert prepared["requested_effort"] == effort
    assert handoff.settings["model_id"] == selected_model
    assert handoff.settings["effort"] == effort
    assert handoff.settings["model_routing_enabled"] is False
    assert handoff.record.call_args.kwargs["effort"] == effort
    handoff.store.assert_awaited_once()
