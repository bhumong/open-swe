from pathlib import Path

import pytest

from scripts import setup_local_env


def test_private_env_file_is_owner_only_and_never_overwritten(tmp_path: Path) -> None:
    env_file = tmp_path / "infra" / ".env"
    setup_local_env._write_private_file(
        env_file,
        {"LANGFUSE_INIT_USER_PASSWORD": "generated-secret"},
        setup_local_env.LANGFUSE_ENV_MARKER,
    )

    assert env_file.stat().st_mode & 0o777 == 0o600
    original = env_file.read_text()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        setup_local_env._write_private_file(
            env_file,
            {"LANGFUSE_INIT_USER_PASSWORD": "replacement-secret"},
            setup_local_env.LANGFUSE_ENV_MARKER,
        )
    assert env_file.read_text() == original


def test_local_settings_keep_langsmith_features_but_disable_tracing() -> None:
    settings = setup_local_env._open_swe_settings(
        anthropic_key="anthropic-secret",
        github_login="local-user",
        langfuse_settings={
            "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "pk-local",
            "LANGFUSE_INIT_PROJECT_SECRET_KEY": "sk-local",
        },
        retained_langsmith_settings={
            "LANGSMITH_API_KEY": "gateway-secret",
            "LANGSMITH_GATEWAY_ENABLED": "true",
            "LANGSMITH_TRACING": "true",
            "SLACK_BOT_TOKEN": "unrelated-secret",
        },
    )

    assert settings["LANGSMITH_API_KEY"] == "gateway-secret"
    assert settings["LANGSMITH_GATEWAY_ENABLED"] == "true"
    assert settings["LANGSMITH_TRACING"] == "false"
    assert settings["LANGCHAIN_TRACING_V2"] == "false"
    assert "SLACK_BOT_TOKEN" not in settings
    assert settings["LLM_MODEL_ID"] == "anthropic:claude-haiku-5-5"
    assert settings["ALLOWED_GITHUB_USERS"] == "local-user"
    assert settings["CONFIGURED_ADMINS"] == "local-user"
