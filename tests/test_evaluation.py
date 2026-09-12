"""评测与接口层测试：对比评测可复现、MCP 工具注册、Web API 全链路。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memory_garden.config import Settings
from memory_garden.evaluation import (
    audit_retrieval_golden_groups,
    eval_retrieval_detailed,
    run_comparative_eval,
)
from memory_garden.retrieval import RetrievalHit
from memory_garden.web import create_app


def test_comparative_eval_reproducible(settings, database, retriever, tmp_path: Path) -> None:
    cases = json.loads(
        (Path(__file__).resolve().parents[1] / "evals" / "agent_cases.json").read_text(encoding="utf-8")
    )["cases"]
    summary = run_comparative_eval(database, retriever, cases, tmp_path)
    full = summary["configs"]["full_agent"]
    baseline = summary["configs"]["one_shot_baseline"]
    assert full["pass_rate"] >= 0.9
    assert full["citation_valid"] == 1.0
    assert full["feedback_adherence"] == 1.0
    assert baseline["feedback_adherence"] == 0.0  # 无判定记忆的基线必须失败
    assert (tmp_path / "results.json").exists()


def test_retrieval_metrics_are_path_level_and_true_recall() -> None:
    class FixedRetriever:
        def search(self, _query):
            return [
                RetrievalHit(1, 1, 1.0, fields={"path": "a.md"}),
                RetrievalHit(2, 1, 0.9, fields={"path": "a.md"}),
                RetrievalHit(3, 2, 0.8, fields={"path": "c.md"}),
                RetrievalHit(4, 3, 0.7, fields={"path": "b.md"}),
            ]

    detailed = eval_retrieval_detailed(
        {"route": FixedRetriever()},
        [
            {"query": "q1", "relevant_paths": ["a.md", "b.md"]},
            {"query": "q2", "relevant_paths": ["missing.md"]},
        ],
    )
    metrics = detailed["summary"]["route"]
    assert metrics["hit_rate@5"] == 0.5
    assert metrics["recall@5"] == 0.5
    assert metrics["precision@5"] == 0.2
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg@5"] == 0.4599
    assert detailed["cases"][0]["routes"]["route"]["first_relevant_rank"] == 1
    serialized = json.dumps(detailed, ensure_ascii=False)
    assert "q1" not in serialized and "a.md" not in serialized


def test_private_golden_audit_tracks_drafts_and_rejects_split_leakage() -> None:
    groups = {
        "real_dev": [
            {
                "id": "dev_001",
                "query": "开发查询",
                "annotation_status": "human_verified",
                "relevant_paths": ["dev.md"],
            }
        ],
        "real_test": [
            {
                "id": "test_001",
                "query": "测试查询",
                "annotation_status": "assistant_draft_requires_owner_review",
                "category": "semantic_paraphrase",
                "relevant_paths": ["test.md"],
            }
        ],
    }
    audit = audit_retrieval_golden_groups(groups, {"dev.md", "test.md"})
    assert audit["group_counts"] == {"real_dev": 1, "real_test": 1}
    assert audit["draft_case_count"] == 1
    assert audit["draft_case_ids"] == ["test_001"]
    assert audit["dev_test_relevant_path_overlap"] == 0
    assert audit["private_text_included"] is False

    groups["real_test"][0]["relevant_paths"] = ["dev.md"]
    with pytest.raises(ValueError, match="overlap count=1"):
        audit_retrieval_golden_groups(groups, {"dev.md"})


def test_private_golden_without_explicit_status_is_not_treated_as_verified() -> None:
    audit = audit_retrieval_golden_groups(
        {
            "real_dev": [
                {
                    "id": "legacy_001",
                    "query": "旧查询",
                    "relevant_paths": ["legacy.md"],
                }
            ]
        },
        {"legacy.md"},
    )
    assert audit["status_counts"] == {"unverified_legacy": 1}
    assert audit["draft_case_count"] == 1
    assert audit["draft_case_ids"] == ["legacy_001"]


def test_mcp_server_registers_all_tools(settings, database, retriever) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from memory_garden.mcp_server import build_mcp_server

    server = build_mcp_server(settings)

    async def collect():
        return [tool.name for tool in await server.list_tools()]

    tools = asyncio.run(collect())
    for expected in (
        "search_sources", "read_source", "get_topic_timeline", "find_change_candidates",
        "discover_cognitive_shifts", "find_interval_events", "search_hypothesis_evidence",
        "get_user_verdicts", "ask_garden",
    ):
        assert expected in tools


def test_web_full_loop(settings, database, retriever) -> None:
    settings_copy = Settings(
        vault_path=settings.vault_path, database_path=settings.database_path
    )
    app = create_app(settings_copy)
    # 注入已同步的库与检索器（create_app 默认新建，测试环境复用 fixture 数据）
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200

    index = client.get("/")
    assert "MEMORY GARDEN" in index.text and "知微" in index.text

    settings_page = client.get("/settings")
    assert settings_page.status_code == 200 and "API Key" in settings_page.text
    assert "明确允许云端 Embedding" in settings_page.text
    assert "明确允许云端 cross-encoder Rerank" in settings_page.text

    saved = client.post(
        "/api/settings",
        json={
            "assistant_name": "知微",
            "backend": "local",
            "embedding_backend": "mock",
            "embedding_provider": "siliconflow",
            "embedding_base_url": "https://api.siliconflow.cn/v1",
            "retrieval_mode": "embedding",
            "reranker_backend": "local_heuristic",
            "reranker_provider": "siliconflow",
            "reranker_base_url": "https://api.siliconflow.cn/v1",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_candidate_limit": 24,
            "reranker_fusion": "rank_fusion",
            "llm_embedding_model": "mock-semantic-v1",
            "allow_cloud_embedding": False,
            "allow_cloud_rerank": False,
        },
    )
    assert saved.status_code == 200 and saved.json()["saved"] is True
    runtime = json.loads((settings.database_path.parent / "settings.json").read_text(encoding="utf-8"))
    assert runtime["embedding_backend"] == "mock"
    assert runtime["embedding_provider"] == "siliconflow"
    assert runtime["retrieval_mode"] == "embedding"
    assert runtime["reranker_backend"] == "local_heuristic"
    assert runtime["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert runtime["rerank_candidate_limit"] == 24
    assert runtime["reranker_fusion"] == "rank_fusion"
    assert runtime["allow_cloud_embedding"] is False
    assert runtime["allow_cloud_rerank"] is False

    ask = client.post("/api/ask", json={"question": "自主判断这个主题有没有变化？"})
    assert ask.status_code == 200
    payload = ask.json()
    assert payload["answer"]["answer_type"] == "traced_change"
    assert payload["message_id"] > 0

    verdict = client.post(
        "/api/verdict",
        json={"message_id": payload["message_id"], "verdict": "accurate"},
    )
    assert verdict.status_code == 200

    discover = client.get("/api/discover?limit=2")
    assert discover.status_code == 200
    candidates = discover.json()["candidates"]
    assert isinstance(candidates, list)

    runs = client.get(f"/api/runs/{payload['message_id']}")
    assert runs.status_code == 200
    assert runs.json()[0]["backend"] in {"local", "scripted", "openai_compatible"}

    threads = client.get("/api/threads").json()
    assert threads, "应至少有一个线程"
    thread_id = threads[0]["thread_id"]
    detail = client.get(f"/api/threads/{thread_id}").json()
    assert detail and detail[0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in detail)
