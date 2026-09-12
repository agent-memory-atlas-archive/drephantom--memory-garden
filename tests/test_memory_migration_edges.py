"""Interrupted upgrades must resume; untraceable legacy feedback cannot be recalled."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_garden.cognitive import DiscoveryService, VerdictService
from memory_garden.db import SCHEMA_VERSION, Database
from memory_garden.memory import MemoryService
from memory_garden.tools import fallback_topic_key


def _old_message_verdict(database: Database, topic: str, revision: str) -> int:
    thread_id = database.execute(
        "INSERT INTO threads(title,created_at) VALUES(?,'2024-01-01')", (topic,)
    )
    database.execute(
        "INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,'2024-01-01')",
        (thread_id, topic),
    )
    answer = json.dumps({"topic": topic, "answer_type": "no_clear_change", "summary": revision})
    message_id = database.execute(
        "INSERT INTO messages(thread_id,role,content,answer_json,created_at) "
        "VALUES(?,'assistant',?,?,'2024-01-01')", (thread_id, revision, answer),
    )
    return database.execute(
        "INSERT INTO verdicts(message_id,topic_key,verdict,user_revision,created_at) "
        "VALUES(?,?,'partly_accurate',?,'2024-01-01')",
        (message_id, fallback_topic_key(topic), revision),
    )


def test_interrupted_backfill_resumes_without_losing_or_duplicating_memories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "resume.db"
    database = Database(path)
    database.initialize()
    _old_message_verdict(database, "写作", "写作的第一份修正。")
    second_id = _old_message_verdict(database, "阅读", "阅读的第二份修正。")
    database.execute("UPDATE schema_meta SET value='4' WHERE key='schema_version'")
    original = MemoryService.record_verdict

    def interrupted(service: MemoryService, verdict_id: int) -> None:
        if verdict_id == second_id:
            raise RuntimeError("simulated interruption after first durable memory")
        original(service, verdict_id)

    with monkeypatch.context() as context:
        context.setattr(MemoryService, "record_verdict", interrupted)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            database.initialize()
    assert database.fetchone("SELECT value FROM schema_meta WHERE key='schema_version'")[0] == "4"
    assert database.fetchone("SELECT COUNT(*) FROM memory_items")[0] == 1
    database.close()

    resumed = Database(path)
    try:
        resumed.initialize()
        assert resumed.fetchone("SELECT value FROM schema_meta WHERE key='schema_version'")[0] == str(SCHEMA_VERSION)
        assert resumed.fetchone("SELECT COUNT(*) FROM memory_items")[0] == 2
        assert MemoryService(resumed).context_for_topic("写作")[0]["statement"] == "写作的第一份修正。"
        assert MemoryService(resumed).context_for_topic("阅读")[0]["statement"] == "阅读的第二份修正。"
        resumed.initialize()
        assert resumed.fetchone("SELECT COUNT(*) FROM memory_items")[0] == 2
    finally:
        resumed.close()


def test_legacy_discovery_feedback_with_unknown_provenance_requires_review(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy.db")
    database.initialize()
    topic = fallback_topic_key("自主判断")
    # This is the exact v4 discovery-review shape: no message and no discovery
    # identity were retained. Two reviews may contradict; linking them is a guess.
    for verdict, revision in [("accurate", "旧判定"), ("not_my_view", "后来已经否认")]:
        database.execute(
            "INSERT INTO verdicts(message_id,topic_key,verdict,user_revision,created_at) "
            "VALUES(NULL,?,?,?,'2024-01-01')", (topic, verdict, revision),
        )
    database.execute("UPDATE schema_meta SET value='4' WHERE key='schema_version'")
    try:
        database.initialize()
        assert MemoryService(database).context_for_topic("自主判断") == []
        assert VerdictService(database).for_topic(topic) == []
        archive = MemoryService(database).list_items(include_inactive=True)
        assert len(archive) == 2
        assert all(item["status"] == "needs_review" for item in archive)
        assert all(item["available_for_recall"] is False for item in archive)
        assert {item["statement"] for item in archive} == {"旧判定", "后来已经否认"}
        database.initialize()
        assert len(MemoryService(database).list_items(include_inactive=True)) == 2
    finally:
        database.close()


def test_legacy_upgrade_does_not_reactivate_revoked_feedback(tmp_path: Path) -> None:
    database = Database(tmp_path / "revoked.db")
    database.initialize()
    verdict_id = _old_message_verdict(database, "写作", "已经撤回的修正。")
    database.execute("UPDATE verdicts SET status='revoked' WHERE id=?", (verdict_id,))
    database.execute("UPDATE schema_meta SET value='4' WHERE key='schema_version'")
    try:
        database.initialize()
        assert MemoryService(database).context_for_topic("写作") == []
        archive = MemoryService(database).list_items(include_inactive=True)
        assert len(archive) == 1
        assert archive[0]["status"] == "revoked"
        assert archive[0]["available_for_recall"] is False
    finally:
        database.close()


def test_discovery_review_failure_keeps_previous_judgment_and_memory(
    database: Database, retriever, monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = database.fetchone("SELECT id FROM source_atoms WHERE is_current=1 LIMIT 1")
    discovery_id = database.execute(
        "INSERT INTO discoveries(topic_key,early_atom_id,recent_atom_id,early_excerpt,recent_excerpt,created_at) "
        "VALUES(?,?,?,'early','recent','2024-01-01')",
        (fallback_topic_key("写作"), atom["id"], atom["id"]),
    )
    service = DiscoveryService(database, retriever)
    service.review(discovery_id, "accurate", user_revision="已经保存的旧判断。")

    def unavailable(*args) -> None:
        raise RuntimeError("simulated memory persistence failure")

    with monkeypatch.context() as context:
        context.setattr(MemoryService, "record_verdict", unavailable)
        with pytest.raises(RuntimeError, match="simulated memory persistence failure"):
            service.review(discovery_id, "not_my_view", user_revision="未能保存的新判断。")
    saved = database.fetchone("SELECT review_verdict,review_revision FROM discoveries WHERE id=?", (discovery_id,))
    assert saved["review_verdict"] == "accurate"
    assert saved["review_revision"] == "已经保存的旧判断。"
    assert database.fetchone("SELECT COUNT(*) FROM verdicts WHERE source_discovery_id=?", (discovery_id,))[0] == 1
    recalled = MemoryService(database).context_for_topic("写作")
    assert len(recalled) == 1 and recalled[0]["statement"] == "已经保存的旧判断。"
