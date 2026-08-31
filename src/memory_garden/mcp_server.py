"""MCP Server：把七个只读认知工具 + 完整回溯入口暴露为 Model Context Protocol 工具。

同一 ToolRegistry 同时服务：
1. 内部 Agent 循环（web/CLI）；
2. 外部 MCP 客户端（Claude Desktop、任何支持 MCP 的宿主）。
这意味着工具的边界规则（只读、发现工具的访问范围）在两个世界完全一致。
运行：``memory-garden mcp``（stdio transport）。
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import Settings
from .db import Database
from .retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from .tools import CognitiveTools


def build_retriever(database: Database, settings: Settings) -> HybridRetriever:
    embedding_client = None
    vector = VectorRetriever(database, embedding_client)
    return HybridRetriever(BM25Retriever(database), vector)


def build_mcp_server(settings: Settings) -> MCPServer:
    database = Database(settings.database_path)
    database.initialize()
    retriever = build_retriever(database, settings)
    retriever.vector.ensure_vectors()
    # 单用户本地进程：一次会话内保持"已发现来源"状态，read_source 的
    # 最小必要访问约束在 MCP 场景同样生效
    tools = CognitiveTools(database, retriever)

    mcp: MCPServer = MCPServer(
        "memory-garden",
        instructions=(
            "Memory Garden 认知回溯工具集。全部只读；只能读取 search_sources/"
            "get_topic_timeline/find_change_candidates 本轮返回过的 atom_id。"
            "较近记录只是 recent 候选；时间相邻不等于因果；"
            "对原因假设必须分别检索 support 与 challenge 两侧。"
        ),
    )

    def _render(observation: Any) -> str:
        return observation.render()

    @mcp.tool()
    def search_sources(query: str, date_from: str = "", date_to: str = "", limit: int = 6) -> str:
        """按关键词与时间窗检索用户本人的记录候选，返回 [A{id}] 编号。"""
        return _render(
            tools.search_sources(
                {"query": query, "date_from": date_from or None, "date_to": date_to or None,
                 "limit": limit}
            )
        )

    @mcp.tool()
    def read_source(atom_id: int) -> str:
        """读取本轮已发现来源的原文与定位信息（引用前必须先读）。"""
        return _render(tools.read_source({"atom_id": atom_id}))

    @mcp.tool()
    def get_topic_timeline(topic: str, limit: int = 10) -> str:
        """按时间整理同一主题的历次表达；最近一条只是 recent 候选。"""
        return _render(tools.get_topic_timeline({"topic": topic, "limit": limit}))

    @mcp.tool()
    def find_change_candidates(topic: str, limit: int = 5) -> str:
        """围绕明确主题提出变化候选端点对（最早端点 × 最近端点）；措辞差异不等于变化。"""
        return _render(tools.find_change_candidates({"topic": topic, "limit": limit}))

    @mcp.tool()
    def discover_cognitive_shifts(limit: int = 5) -> str:
        """全库发现可能被忽视的变化候选；较近记录只是 recent 候选，需用户确认。"""
        return _render(tools.discover_cognitive_shifts({"limit": limit}))

    @mcp.tool()
    def find_interval_events(date_from: str, date_to: str, query: str = "", limit: int = 6) -> str:
        """在两个日期端点构成的区间内搜索经历/决定/事件候选。时间相邻不等于因果。"""
        return _render(
            tools.find_interval_events(
                {"date_from": date_from, "date_to": date_to, "query": query, "limit": limit}
            )
        )

    @mcp.tool()
    def search_hypothesis_evidence(hypothesis: str, stance: str, query: str = "") -> str:
        """对原因假设分侧检索：stance=support 或 challenge；两侧都必须调用。"""
        return _render(
            tools.search_hypothesis_evidence(
                {"hypothesis": hypothesis, "stance": stance, "query": query}
            )
        )

    @mcp.tool()
    def get_user_verdicts(topic: str) -> str:
        """读取用户对该主题此前的确认/否认判定；被否认的解释不得复用。"""
        return _render(tools.get_user_verdicts({"topic": topic}))

    @mcp.tool()
    def ask_garden(question: str) -> str:
        """运行完整认知回溯 Agent（本地确定性路径），返回带引用的结构化结论。"""
        from .agent import AgentHarness

        harness = AgentHarness(database, retriever, settings)
        result = harness.run(question)
        return json.dumps(
            {
                "reply": result.reply,
                "answer_type": result.answer.answer_type,
                "unknowns": result.answer.unknowns,
                "question_to_user": result.answer.question_to_user,
                "backend": result.backend,
            },
            ensure_ascii=False,
        )

    return mcp


def run_mcp(settings: Settings) -> None:
    server = build_mcp_server(settings)
    server.run()
