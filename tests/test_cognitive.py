"""认知服务测试：发现扫描、评审沉淀、判定复用。"""
from __future__ import annotations

import pytest

from memory_garden.cognitive import DiscoveryService, VerdictService


def test_discovery_scan_finds_candidates(database, retriever) -> None:
    service = DiscoveryService(database, retriever)
    scan_id, candidates = service.run_scan(limit=5)
    assert scan_id > 0
    assert candidates, "合成库应发现至少一个候选"
    for item in candidates:
        assert item.early_date < item.recent_date
        assert item.early_atom_id != item.recent_atom_id
        assert "这是表达更具体了" in item.question_to_user or "变了" in item.question_to_user


def test_review_sinks_into_verdict_memory(database, retriever) -> None:
    service = DiscoveryService(database, retriever)
    _, candidates = service.run_scan(limit=3)
    target = candidates[0]
    outcome = service.review(target.discovery_id, "not_my_view", user_revision="这是误解")
    assert outcome["verdict"] == "not_my_view"
    row = database.fetchone("SELECT status, review_verdict FROM discoveries WHERE id=?", (target.discovery_id,))
    assert row["status"] == "reviewed"
    # 评审结果成为主题级判定记忆
    verdicts = VerdictService(database).for_topic(target.topic_key)
    assert any(v["verdict"] == "not_my_view" for v in verdicts)


def test_verdict_service_persists_and_decodes(database) -> None:
    service = VerdictService(database)
    # 造一条 assistant 消息
    database.execute("INSERT INTO threads(title, created_at) VALUES('t', '2026-01-01')")
    message_id = database.execute(
        "INSERT INTO messages(thread_id, role, content, created_at)"
        " VALUES(1, 'assistant', '答案', '2026-01-01')"
    )
    saved = service.save_verdict(
        message_id=message_id,
        verdict="partly_accurate",
        user_revision="方向对但缺了一段经历",
        missing_event="2021 年的项目复盘",
        rejected_atom_ids=[3, 4],
    )
    assert saved["verdict"] == "partly_accurate"
    assert saved["rejected_atom_ids"] == [3, 4]
    with pytest.raises(ValueError):
        service.save_verdict(message_id=message_id, verdict="胡说")
