"""Agent 层测试：本地确定性回答协议、判定遵守、引用守卫、降级路径。"""
from __future__ import annotations

import re

import pytest

from memory_garden.agent import AgentHarness, OpenAIProvider
from memory_garden.evaluation import ScriptedProvider, standard_trace_script
from memory_garden.retrieval import build_retriever


@pytest.fixture()
def harness(settings, database, retriever) -> AgentHarness:
    return AgentHarness(database, retriever, settings)


def test_true_change_traced_with_citations(harness) -> None:
    result = harness.run("自主判断这个主题，我的想法以前到现在有没有变化？")
    assert result.answer.answer_type == "traced_change"
    assert result.answer.early_position is not None
    assert result.answer.recent_position is not None
    assert result.answer.recent_position.status == "latest_memory_candidate"
    import re

    refs = re.findall(r"\[A(\d+)\]", result.reply)
    assert refs, "本地回答必须带引用"
    trace_refs = {int(r) for ev in result.trace for r in ev.get("refs") or []}
    assert {int(r) for r in refs} <= trace_refs


def test_wording_not_change(harness) -> None:
    result = harness.run("谨慎决策")
    assert result.answer.answer_type == "no_clear_change"


def test_synonym_gap_traced(harness) -> None:
    result = harness.run("一个人恢复精力")
    assert result.answer.answer_type == "traced_change"


def test_unknown_topic_abstains(harness) -> None:
    result = harness.run("火星殖民")
    assert result.answer.answer_type == "insufficient_evidence"
    assert result.answer.abstained
    assert result.answer.question_to_user


def test_quoted_view_not_treated_as_user_stance(harness) -> None:
    result = harness.run("书摘中的独立")
    assert result.answer.abstained


def test_prior_denial_is_respected(settings, database, retriever) -> None:
    harness = AgentHarness(database, retriever, settings)
    first = harness.run("自主判断")
    from memory_garden.cognitive import VerdictService

    assert first.message_id is not None
    VerdictService(database).save_verdict(
        message_id=first.message_id,
        verdict="no_change",
        user_revision="前后不是立场变化，只是表达更具体。",
    )
    second = harness.run("自主判断")
    assert second.answer.answer_type == "no_clear_change"
    assert any("不构成" in item.statement or "否认" in item.statement
               for item in second.answer.counter_evidence) or second.answer.counter_evidence


def test_discovery_routing_scope(harness) -> None:
    """主题明确 → 不注册全库发现；发现请求 → 注册且走全库发现。"""
    scoped = harness.run("自主判断这个主题有没有变化？")
    assert not any(ev["tool"] == "discover_cognitive_shifts" for ev in scoped.trace)
    discovery = harness.run("帮我看看有没有我自己没注意到的变化")
    assert any(ev["tool"] == "discover_cognitive_shifts" for ev in discovery.trace)


def test_scripted_loop_citation_valid(settings, database, retriever) -> None:
    provider = ScriptedProvider(standard_trace_script("自主判断", "这一变化与区间内的经历有关"))
    harness = AgentHarness(database, retriever, settings, provider=provider)
    result = harness.run("自主判断")
    assert result.backend == "scripted"
    assert result.answer.answer_type == "traced_change"
    import re

    refs = {int(r) for r in re.findall(r"\[A(\d+)\]", result.reply)}
    trace_refs = {int(r) for ev in result.trace for r in ev.get("refs") or []}
    assert refs <= trace_refs
    assert result.private_vault_sent is False


def test_cloud_tool_observation_is_recorded_as_private_vault_sent(
    settings, database, retriever
) -> None:
    class FakeCloudClient:
        def __init__(self) -> None:
            self.cursor = 0

        def chat_with_tools(self, messages, tools, timeout_seconds=None, tool_choice=None):
            if tools and tools[0]['name'] == 'plan_turn':
                return {'tool_calls': [{'function': {'name': 'plan_turn', 'arguments':
                    '{"intent":"cognitive_trace","query":"自主判断","reply":"","updates_current_view":false}'}}]}
            tool_messages = [message for message in messages if message.get("role") == "tool"]
            calls = [
                ("search_sources", '{"query":"自主判断","limit":8}'),
                ("get_topic_timeline", '{"topic":"自主判断"}'),
                ("find_change_candidates", '{"topic":"自主判断","limit":1}'),
                (
                    "find_interval_events",
                    '{"topic":"自主判断","date_from":"2010-01-01",'
                    '"date_to":"2030-01-01"}',
                ),
                (
                    "search_hypothesis_evidence",
                    '{"hypothesis":"变化与区间经历有关","query":"自主判断",'
                    '"stance":"support"}',
                ),
                (
                    "search_hypothesis_evidence",
                    '{"hypothesis":"变化与区间经历有关","query":"自主判断",'
                    '"stance":"challenge"}',
                ),
            ]
            if tools and self.cursor < len(calls):
                name, arguments = calls[self.cursor]
                self.cursor += 1
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{self.cursor}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            if tools:
                match = re.search(r'"atom_id":\s*(\d+)', str(tool_messages[-1]["content"]))
                assert match is not None
                return {
                    "content": "这段工具过程不能呈现给用户。",
                    "tool_calls": [{'function': {'name': 'finish_response', 'arguments':
                        '{"reply":"这条记录可以作为本轮核对起点 [A'+match.group(1)+']，但证据仍有限。"}'}}],
                }
            return {
                "content": "这条记录可以作为这一轮核对的起点，不过现有证据仍然有限，需要你确认。",
                "tool_calls": [],
            }

    cloud_settings = settings
    cloud_settings.llm_chat_model = "fake-cloud-model"
    provider = OpenAIProvider(FakeCloudClient())  # type: ignore[arg-type]
    result = AgentHarness(database, retriever, cloud_settings, provider=provider).run("自主判断")
    assert result.private_vault_sent is True
    # 模型最终回答保留通过守卫的引用，不再另调模型压缩并改写一次。
    assert re.search(r"\[A\d+\]", result.reply)
    assert result.trace[0]['tool'] == 'plan_turn'
    assert not any(event.get('tool', '').startswith('presentation_rewrite') for event in result.trace)
    row = database.fetchone(
        "SELECT private_vault_sent FROM agent_runs WHERE message_id=?", (result.message_id,)
    )
    assert row is not None and int(row["private_vault_sent"]) == 1


def test_provider_final_is_rejected_until_evidence_plan_is_complete(
    settings, database, retriever
) -> None:
    """模型读完时间线就抢答时，Harness 要求补齐端点、区间与正反证据。"""
    standard = standard_trace_script("自主判断", "这一变化与区间内的经历有关")
    provider = ScriptedProvider(
        standard[:2]
        + [{"final": "我已经可以直接作答。"}]
        + standard[2:]
    )
    result = AgentHarness(database, retriever, settings, provider=provider).run("自主判断")
    assert result.answer.answer_type == "traced_change"
    assert any(event.get("tool") == "evidence_plan_guard" for event in result.trace)
    assert not any(
        event.get("tool") == "evidence_plan_guard_failed" for event in result.trace
    )


def test_cloud_embedding_upload_is_recorded_as_private_vault_sent(
    settings, database
) -> None:
    class FakeCloudEmbedding:
        provider = "test-cloud"
        model = "fake-embedding-v1"
        dimension = 4
        is_cloud = True

        @staticmethod
        def _vector(text: str) -> list[float]:
            total = sum(text.encode("utf-8"))
            vector = [0.0, 0.0, 0.0, 0.0]
            vector[total % 4] = 1.0
            return vector

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [self._vector(text) for text in texts]

        def embed_one(self, text: str) -> list[float]:
            return self._vector(text)

    retriever = build_retriever(
        database,
        settings,
        mode="embedding",
        embedding_backend=FakeCloudEmbedding(),
    )
    result = AgentHarness(database, retriever, settings).run("自主判断")
    assert result.private_vault_sent is True
    row = database.fetchone(
        "SELECT private_vault_sent FROM agent_runs WHERE message_id=?", (result.message_id,)
    )
    assert row is not None and int(row["private_vault_sent"]) == 1


def test_fake_citation_never_reaches_user(settings, database, retriever) -> None:
    """模型引用越界 → 有界修复成功则替换；失败则本地降级。伪引用都不能到达用户。"""
    provider = ScriptedProvider(
        [
            {"tool": "search_sources", "args": {"query": "自主判断"}},
            {"final": "结论 [A77777] 毫无依据。"},
        ]
    )
    harness = AgentHarness(database, retriever, settings, provider=provider)
    result = harness.run("自主判断")
    assert result.backend in {"scripted", "local_fallback"}
    assert "A77777" not in result.reply


def test_reused_refs_remain_attributed_to_each_tool(settings, database, retriever) -> None:
    """同一 atom 被正反证据工具重复返回时，trace 仍保留完整引用集合。"""
    provider = ScriptedProvider(
        standard_trace_script("自主判断", "这一变化与区间内的经历有关")
    )
    result = AgentHarness(database, retriever, settings, provider=provider).run("自主判断")
    evidence_events = [
        event for event in result.trace
        if event.get("tool") == "search_hypothesis_evidence"
    ]
    assert len(evidence_events) == 2
    assert all(event.get("refs") for event in evidence_events)
    assert "new_refs" in evidence_events[1]
