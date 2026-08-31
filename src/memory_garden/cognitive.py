"""认知服务：主动发现扫描 + 用户判定记忆。

协议要点：
- 候选发现与变化确认严格分离：Agent 只产 candidate，确认只能来自用户判定；
- 用户判定沉淀为主题级记忆，下一轮回溯自动加载；被否认的解释不得复用；
- 发现评审与对话判定共用同一张 verdicts 主题记忆（origin 不同，语义一致）。
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .db import Database, utc_now
from .retrieval import HybridRetriever
from .tools import CognitiveTools, fallback_topic_key

VALID_VERDICTS = {
    "accurate", "partly_accurate", "no_change", "not_my_view",
    "insufficient_evidence", "defer",
}


class DiscoveryCandidate(BaseModel):
    discovery_id: int
    topic: str
    topic_key: str
    early_atom_id: int
    recent_atom_id: int
    early_date: str | None
    recent_date: str | None
    early_excerpt: str
    recent_excerpt: str
    diff_terms: list[str]
    score: float
    question_to_user: str
    change_type: str = ""
    signal_type: str = "stance_pair"


class VerdictService:
    def __init__(self, database: Database):
        self.database = database

    def save_verdict(
        self,
        *,
        message_id: int,
        verdict: str,
        topic_key: str | None = None,
        user_revision: str = "",
        confirmed_interpretation: str = "",
        missing_event: str = "",
        accepted_atom_ids: list[int] | None = None,
        rejected_atom_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"不支持的用户判定: {verdict}")
        message = self.database.fetchone(
            "SELECT id FROM messages WHERE id=? AND role='assistant'", (message_id,)
        )
        if message is None:
            raise KeyError("Agent 回答不存在")
        if topic_key is None:
            user_row = self.database.fetchone(
                "SELECT content FROM messages WHERE thread_id=(SELECT thread_id FROM messages"
                " WHERE id=?) AND role='user' AND id<? ORDER BY id DESC LIMIT 1",
                (message_id, message_id),
            )
            topic_key = fallback_topic_key(str(user_row["content"]) if user_row else "")
        now = utc_now()
        verdict_id = self.database.execute(
            """
            INSERT INTO verdicts(message_id, topic_key, verdict, user_revision,
                confirmed_interpretation, missing_event, accepted_atom_ids_json,
                rejected_atom_ids_json, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                message_id, topic_key, verdict, user_revision.strip() or None,
                confirmed_interpretation.strip() or None, missing_event.strip() or None,
                json.dumps(sorted(set(accepted_atom_ids or []))),
                json.dumps(sorted(set(rejected_atom_ids or []))),
                now,
            ),
        )
        row = self.database.fetchone("SELECT * FROM verdicts WHERE id=?", (verdict_id,))
        if row is None:
            return {}
        saved = dict(row)
        for field in ("accepted_atom_ids_json", "rejected_atom_ids_json"):
            saved[field.removesuffix("_json")] = json.loads(saved.pop(field) or "[]")
        return saved

    def for_topic(self, topic_key: str, limit: int = 6) -> list[dict[str, Any]]:
        rows = self.database.fetchall(
            "SELECT * FROM verdicts WHERE topic_key=? ORDER BY id DESC LIMIT ?", (topic_key, limit)
        )
        return [dict(row) for row in rows]

    def rejected_atom_ids(self, topic_key: str) -> set[int]:
        rejected: set[int] = set()
        for row in self.for_topic(topic_key):
            if row["verdict"] in {"no_change", "not_my_view"}:
                rejected.update(json.loads(row["rejected_atom_ids_json"] or "[]"))
        return rejected


class DiscoveryService:
    """主动发现：快照引擎（或启发式回退）→ 持久化候选 → 用户反应/评审沉淀为判定记忆。"""

    def __init__(
        self, database: Database, retriever: HybridRetriever, classifier: Any | None = None
    ):
        self.database = database
        self.retriever = retriever
        self.classifier = classifier  # None = 确定性分类；可注入 LLMChangeClassifier

    def _engine_candidates(self, limit: int) -> tuple[list[dict[str, Any]], str]:

        tools = CognitiveTools(self.database, self.retriever)
        if tools._snapshot_count():
            from .snapshots import DiscoveryEngine, candidate_to_dict

            engine = DiscoveryEngine(self.database, classifier=self.classifier)
            return [candidate_to_dict(item) for item in engine.run(limit=limit)], "stance_snapshots"
        raw = tools.discover_cognitive_shifts({"limit": limit}).data["candidates"]
        return raw, "heuristic"

    def run_scan(self, limit: int = 5) -> tuple[int, list[DiscoveryCandidate]]:
        raw, source = self._engine_candidates(limit)
        scan_id = self.database.execute(
            "INSERT INTO discovery_scans(candidates_found, created_at) VALUES(?,?)",
            (len(raw), utc_now()),
        )
        candidates: list[DiscoveryCandidate] = []
        now = utc_now()
        for item in raw:
            early, recent = item["early"], item["recent"]
            topic = str(item.get("topic") or "")
            discovery_id = self.database.execute(
                """
                INSERT INTO discoveries(scan_id, topic_key, early_atom_id, recent_atom_id,
                    early_date, recent_date, early_excerpt, recent_excerpt, diff_terms_json,
                    score, status, change_type, signal_type, change_confidence, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,'candidate',?,?,?,?)
                """,
                (
                    scan_id, str(item.get("topic_key") or fallback_topic_key(topic)),
                    int(early["atom_id"]), int(recent["atom_id"]),
                    str(early.get("date") or "") or None, str(recent.get("date") or "") or None,
                    str(early.get("excerpt") or "")[:300], str(recent.get("excerpt") or "")[:300],
                    json.dumps(item.get("diff_terms", []), ensure_ascii=False),
                    float(item.get("score") or 0),
                    str(item.get("change_type") or "true_change"),
                    str(item.get("signal_type") or "stance_pair"),
                    float(item.get("change_confidence") or 0.5),
                    now,
                ),
            )
            is_contrast = item.get("signal_type") == "explicit_contrast"
            if is_contrast:
                question = (
                    f"你自己写过：「{str(early.get('excerpt') or '')[:80]}」。"
                    "这句话里藏着一次你亲口承认的转变——还成立吗？"
                )
            else:
                question = (
                    f"关于「{topic}」：{str(early.get('date') or '')} 你写「{str(early.get('excerpt') or '')[:60]}」，"
                    f"{str(recent.get('date') or '')} 你写「{str(recent.get('excerpt') or '')[:60]}」。"
                    "这是表达更具体了，还是你的想法真的变了？"
                )
            candidates.append(
                DiscoveryCandidate(
                    discovery_id=discovery_id,
                    topic=topic,
                    topic_key=str(item.get("topic_key") or ""),
                    early_atom_id=int(early["atom_id"]),
                    recent_atom_id=int(recent["atom_id"]),
                    early_date=early.get("date"),
                    recent_date=recent.get("date"),
                    early_excerpt=str(early.get("excerpt") or "")[:300],
                    recent_excerpt=str(recent.get("excerpt") or "")[:300],
                    diff_terms=list(item.get("diff_terms", [])),
                    score=float(item.get("score") or 0),
                    question_to_user=question,
                    change_type=str(item.get("change_type") or ""),
                    signal_type=str(item.get("signal_type") or "stance_pair"),
                )
            )
        return scan_id, candidates

    def mark_shown(self, discovery_ids: list[int]) -> None:
        """呈现即计数：排序器的新颖度因子依赖展示历史。"""
        now = utc_now()
        with self.database.transaction() as connection:
            for discovery_id in discovery_ids:
                connection.execute(
                    "UPDATE discoveries SET shown_count = shown_count + 1, last_shown_at = ?"
                    " WHERE id = ?",
                    (now, discovery_id),
                )

    def react(self, discovery_id: int, reaction: str) -> dict[str, Any]:
        """一键反应（accurate/wrong/boring）：直接写回该主题的排序权重。"""
        if reaction not in {"accurate", "wrong", "boring"}:
            raise ValueError(f"不支持的反应: {reaction}")
        row = self.database.fetchone("SELECT id FROM discoveries WHERE id=?", (discovery_id,))
        if row is None:
            raise KeyError("发现候选不存在")
        reaction_id = self.database.execute(
            "INSERT INTO candidate_reactions(discovery_id, reaction, created_at) VALUES(?,?,?)",
            (discovery_id, reaction, utc_now()),
        )
        if reaction in {"wrong", "boring"}:
            # 负反应的候选直接沉底，避免同一伤口被反复揭开
            self.database.execute(
                "UPDATE discoveries SET status='reviewed', review_verdict='defer' WHERE id=?",
                (discovery_id,),
            )
        return {"discovery_id": discovery_id, "reaction_id": reaction_id, "reaction": reaction}

    def review(
        self,
        discovery_id: int,
        verdict: str,
        user_revision: str = "",
        missing_event: str = "",
    ) -> dict[str, Any]:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"不支持的用户判定: {verdict}")
        row = self.database.fetchone("SELECT * FROM discoveries WHERE id=?", (discovery_id,))
        if row is None:
            raise KeyError("发现候选不存在")
        self.database.execute(
            "UPDATE discoveries SET status='reviewed', review_verdict=?, review_revision=?"
            " WHERE id=?",
            (verdict, user_revision.strip() or None, discovery_id),
        )
        # 评审结果同时沉淀为主题级判定记忆，供下一轮回溯自动遵守
        verdict_id = self.database.execute(
            """
            INSERT INTO verdicts(message_id, topic_key, verdict, user_revision, missing_event,
                accepted_atom_ids_json, rejected_atom_ids_json, created_at)
            VALUES(NULL, ?, ?, ?, ?, '[]', '[]', ?)
            """,
            (
                str(row["topic_key"]), verdict, user_revision.strip() or None,
                missing_event.strip() or None, utc_now(),
            ),
        )
        return {"discovery_id": discovery_id, "verdict_id": verdict_id, "verdict": verdict}

    def pending_candidates(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.database.fetchall(
            "SELECT * FROM discoveries WHERE status='candidate' ORDER BY score DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]
