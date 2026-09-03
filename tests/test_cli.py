"""CLI 评测入口必须与用户配置的 Vault/数据库隔离。"""
from __future__ import annotations

import json

from memory_garden.cli import main
from memory_garden.config import Settings


def _private_settings(tmp_path) -> Settings:
    return Settings(
        vault_path=tmp_path / "do-not-read-private-vault",
        database_path=tmp_path / "do-not-create-user.db",
        llm_chat_model="",
    )


def test_eval_discovery_uses_synthetic_temp_database(
    tmp_path, monkeypatch, capsys
) -> None:
    settings = _private_settings(tmp_path)
    monkeypatch.setattr(Settings, "load", classmethod(lambda _cls: settings))
    output = tmp_path / "discovery-output"

    assert main(["eval-discovery", "--output", str(output)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["snapshots_extracted"] > 0
    assert report["candidates_returned"] > 0
    assert not settings.database_path.exists()


def test_eval_agent_does_not_initialize_user_database(
    tmp_path, monkeypatch, capsys
) -> None:
    settings = _private_settings(tmp_path)
    monkeypatch.setattr(Settings, "load", classmethod(lambda _cls: settings))
    output = tmp_path / "agent-output"

    assert main(["eval-agent", "--output", str(output)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["total_cases"] == 12
    assert report["configs"]["full_agent"]["pass_rate"] == 1.0
    assert not settings.database_path.exists()
