"""快照层测试：抽取、变化分类学、显式对比句、发现引擎评分与反应写回。"""
from __future__ import annotations

from memory_garden.cognitive import DiscoveryService
from memory_garden.snapshots import (
    CANDIDATE_TYPES,
    DeterministicChangeClassifier,
    DeterministicExtractor,
    DiscoveryEngine,
    ensure_snapshots,
    find_contrast_sentences,
)


def _snapshots(database) -> int:
    return ensure_snapshots(database, DeterministicExtractor())["extracted"]


def test_extract_fills_only_user_atoms(database) -> None:
    filled = _snapshots(database)
    assert filled > 0
    # 引用/AI 草稿不产生有效立场快照（has_stance=0）
    rows = database.fetchall(
        "SELECT has_stance FROM stance_snapshots WHERE topic IN ('独立', '引用')"
    )
    assert all(row["has_stance"] == 0 for row in rows)
    # 幂等：第二次抽取为 0
    assert ensure_snapshots(database, DeterministicExtractor())["extracted"] == 0


def test_classifier_maps_overlap_to_taxonomy() -> None:
    classifier = DeterministicChangeClassifier()
    results = classifier.classify(
        [
            {"lexical_overlap": 0.5, "early_tone": 0.5, "recent_tone": 0.5},
            {"lexical_overlap": 0.25, "early_tone": 0.5, "recent_tone": 0.5},
            {"lexical_overlap": 0.1, "early_tone": 0.8, "recent_tone": 0.3},
        ]
    )
    assert results[0]["change_type"] == "wording_drift"
    assert results[1]["change_type"] == "deepening"
    assert results[2]["change_type"] == "true_change"


def test_engine_excludes_wording_drift_from_candidates(database) -> None:
    _snapshots(database)
    engine = DiscoveryEngine(database)
    candidates = engine.snapshot_candidates(limit=10)
    assert candidates, "快照引擎应产出候选"
    assert all(item.change_type in CANDIDATE_TYPES for item in candidates)
    # 谨慎决策（措辞深化）不得进入候选
    assert all(item.topic != "谨慎决策" for item in candidates)
    # 每个候选都有可定位的端点
    for item in candidates:
        assert item.early["atom_id"] != item.recent["atom_id"]


def test_engine_scoped_topic_resolution(database) -> None:
    _snapshots(database)
    engine = DiscoveryEngine(database)
    topics = engine.resolve_topics("自主判断这个主题有没有变化")
    assert topics, "应解析到快照主题"
    scoped = engine.snapshot_candidates(topics=topics, limit=3)
    assert all(item.topic in topics for item in scoped)


def test_contrast_sentence_detected(database) -> None:
    hits = find_contrast_sentences(database, limit=5)
    assert hits, "合成库包含显式对比句"
    assert any("以前" in hit["sentence"] and "现在" in hit["sentence"] for hit in hits)


def test_contrast_signal_becomes_candidate(database) -> None:
    _snapshots(database)
    engine = DiscoveryEngine(database)
    candidates = engine.contrast_candidates(limit=3)
    assert candidates
    assert all(item.signal_type == "explicit_contrast" for item in candidates)
    assert all(item.change_confidence >= 0.8 for item in candidates)


def test_scan_persists_taxonomy_and_marks_shown(database, retriever) -> None:
    _snapshots(database)
    service = DiscoveryService(database, retriever)
    _, candidates = service.run_scan(limit=5)
    assert candidates
    service.mark_shown([item.discovery_id for item in candidates])
    row = database.fetchone(
        "SELECT shown_count, last_shown_at FROM discoveries WHERE id=?",
        (candidates[0].discovery_id,),
    )
    assert row["shown_count"] == 1
    assert row["last_shown_at"]


def test_reaction_writeback_changes_ranking(database, retriever) -> None:
    _snapshots(database)
    service = DiscoveryService(database, retriever)
    _, candidates = service.run_scan(limit=5)
    target = candidates[0]
    service.react(target.discovery_id, "wrong")
    # 负反应候选被沉底，不再出现在 pending 列表
    pending_ids = {row["id"] for row in DiscoveryService(database, retriever).pending_candidates()}
    assert target.discovery_id not in pending_ids
    # 反应记录进入排序权重表
    row = database.fetchone(
        "SELECT reaction FROM candidate_reactions WHERE discovery_id=?",
        (target.discovery_id,),
    )
    assert row["reaction"] == "wrong"


def test_novelty_demotion_after_showing(database, retriever) -> None:
    _snapshots(database)
    engine = DiscoveryEngine(database)
    first = engine.run(limit=3)
    assert first
    service = DiscoveryService(database, retriever)
    _, candidates = service.run_scan(limit=3)
    service.mark_shown([item.discovery_id for item in candidates])
    fresh_engine = DiscoveryEngine(database)
    rescored = {item.topic_key: item.score for item in fresh_engine.run(limit=5)}
    original = {item.topic_key: item.score for item in first}
    for key, score in rescored.items():
        if key in original:
            assert score <= original[key] + 1e-6, "已呈现主题的新颖度降权不应提分"
