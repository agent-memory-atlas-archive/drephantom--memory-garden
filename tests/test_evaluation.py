"""评测与接口层测试：对比评测可复现、MCP 工具注册、Web API 全链路。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memory_garden.config import Settings
from memory_garden.evaluation import run_comparative_eval
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

    saved = client.post(
        "/api/settings",
        json={"assistant_name": "知微", "backend": "local"},
    )
    assert saved.status_code == 200 and saved.json()["saved"] is True

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
