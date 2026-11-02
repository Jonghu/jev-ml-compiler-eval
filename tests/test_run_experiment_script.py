import os
from pathlib import Path

from jev_compiler_eval.config.environment import load_cli_environment


def test_loads_cli_environment_without_overriding_exported_value(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TYPESAFE_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "exported-value")

    load_cli_environment(env_path)

    assert os.environ["TYPESAFE_API_KEY"] == "exported-value"


def test_loads_cli_environment_when_value_is_not_exported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TYPESAFE_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    load_cli_environment(env_path)

    assert os.environ["TYPESAFE_API_KEY"] == "file-value"
