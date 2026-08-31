"""工具层测试：只读边界、最小必要访问、端点配对身份、区间约束。"""
from __future__ import annotations

import pytest

from memory_garden.tools import build_tool_registry


@pytest.fixture()
def registry(database, retriever) -> dict:
    return build_tool_registry(database, retriever, allow_discovery=True)


@pytest.fixture()
def scoped_registry(database, retriever) -> dict:
    return build_tool_registry(database, retriever, allow_discovery=False)


def test_read_source_rejects_undiscovered_atom(registry) -> None:
    observation = registry["read_source"].handler({"atom_id": 99999})
    assert "不在本轮已发现列表" in observation.data["error"]


def test_read_source_allows_discovered_atom(registry) -> None:
    registry["search_sources"].handler({"query": "自主判断"})
    hits = registry["get_topic_timeline"].handler({"topic": "自主判断"}).data["timeline"]
    atom_id = int(hits[0]["atom_id"])
    observation = registry["read_source"].handler({"atom_id": atom_id})
    assert observation.data["atom_id"] == atom_id
    assert observation.data["citation"] == f"[A{atom_id}]"


def test_discovery_tool_removed_when_topic_scoped(scoped_registry) -> None:
    assert "discover_cognitive_shifts" not in scoped_registry
    assert "find_change_candidates" in scoped_registry  # 定向配对始终可用


def test_find_change_candidates_requires_topic(registry) -> None:
    observation = registry["find_change_candidates"].handler({"topic": ""})
    assert "error" in observation.data


def test_candidate_pair_shares_identity_and_spans_time(registry) -> None:
    observation = registry["find_change_candidates"].handler({"topic": "自主判断"})
    candidates = observation.data["candidates"]
    assert candidates, "自主判断应构成候选对"
    pair = candidates[0]
    assert pair["early"]["date"] < pair["recent"]["date"]
    assert pair["recent_status"] == "latest_memory_candidate"
    assert float(pair["lexical_overlap"]) < 0.35


def test_wording_not_change_pair_detected_as_high_overlap(registry) -> None:
    observation = registry["find_change_candidates"].handler({"topic": "谨慎决策"})
    pair = observation.data["candidates"][0]
    assert float(pair["lexical_overlap"]) >= 0.35


def test_interval_events_constrained_to_dates(registry) -> None:
    observation = registry["find_interval_events"].handler(
        {"topic": "自主判断", "date_from": "2019-03-10", "date_to": "2024-09-12"}
    )
    assert observation.data["interval"] == ["2019-03-10", "2024-09-12"]
    for event in observation.data["events"]:
        date = str(event.get("event_time") or event.get("recorded_at") or "")
        assert "2019-03-10" <= date <= "2024-09-12"


def test_stance_search_requires_both_sides(registry) -> None:
    hypothesis = "这一变化与区间内的经历有关"
    support = registry["search_hypothesis_evidence"].handler(
        {"hypothesis": hypothesis, "query": "自主判断", "stance": "support"}
    )
    challenge = registry["search_hypothesis_evidence"].handler(
        {"hypothesis": hypothesis, "query": "自主判断", "stance": "challenge"}
    )
    assert support.data["stance"] == "support"
    assert challenge.data["stance"] == "challenge"
    assert "单侧命中不能证明假设" in support.boundary


def test_user_verdicts_boundary(registry) -> None:
    observation = registry["get_user_verdicts"].handler({"topic": "自主判断"})
    assert "用户判定" in observation.boundary
