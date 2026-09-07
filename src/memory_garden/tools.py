"""七个只读认知工具：一次认知回溯闭环所需的全部动作，每个都带明确边界。

设计原则（对应《AI Agents in Depth》第 4 章工具设计）：
- 全部只读：没有"确认变化"这个工具——确认只能由用户本人做出；
- 最小必要访问：主题已明确时不注册全库发现工具（运行时收缩，不只是提示词约束）；
- 观察自带边界说明：工具返回值里明示"时间相邻≠因果"等告诫，模型每次都能看到。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .db import Database
from .retrieval import HybridRetriever, RetrievalQuery

_TOPIC_STOP = {
    "为什么", "怎么", "如何", "现在", "以前", "过去", "变化", "改变", "观点", "想法",
    "认为", "觉得", "看看", "我的", "这件事", "演化", "转变", "有没有", "主题", "自己",
    "这种", "那种", "关于", "曾经", "如今", "从未", "专门", "本来", "其实",
    "一个", "从未有过", "有过", "看着", "周围",
}

# 全库发现专用的结构性停用词：笔记标题里的"早期/近期/整理"等元信息不构成主题
THEME_STOP = {
    "早期", "近期", "表达", "记录", "片段", "想法", "经历", "反例", "整理", "索引",
    "待确认", "建议", "方案", "问题", "待整理", "灵感", "样例", "清单", "地图",
    "审校", "覆盖", "新增", "规范", "开局", "种子", "场景", "版本", "融合", "第一卷",
    "综合", "社会", "角色", "剧情线", "主轴", "世界观", "日常",
}


def topic_terms(text: str) -> list[str]:
    """极简主题抽取：长度≥2 的连续词片段，过滤疑问/泛指词。"""
    import jieba

    jieba.setLogLevel(60)
    terms: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip()
        if len(token) < 2 or token.isdigit() or token in _TOPIC_STOP:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:4]


def fallback_topic_key(text: str) -> str:
    import hashlib

    compact = re.sub(r"\s+", "", text).lower()[:80]
    terms = topic_terms(text)
    if terms:
        return "terms:" + "|".join(sorted(terms))
    return "text:" + hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]


class ToolObservation(BaseModel):
    """工具返回：数据 + 边界说明，二者都进入模型上下文。"""

    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    boundary: str = ""

    def render(self) -> str:
        parts = [f"[{self.tool}]"]
        if self.data:
            parts.append(json.dumps(self.data, ensure_ascii=False, default=str)[:6000])
        if self.boundary:
            parts.append(f"边界: {self.boundary}")
        return "\n".join(parts)


@dataclass
class ToolSpec:
    """一个工具 = 名称 + 描述 + 参数 schema + 实现。同一注册表服务内部循环与 MCP。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolObservation]
    discovery_only: bool = False  # 仅在"用户未给出主题"的发现场景开放


class CognitiveTools:
    """工具实现。全部只读；引用以 [A{atom_id}] 编号返回。"""

    def __init__(self, database: Database, retriever: HybridRetriever):
        self.database = database
        self.retriever = retriever
        self._seen_atom_ids: set[int] = set()

    # ── 1. 检索候选来源 ────────────────────────────────────────────────
    def search_sources(self, args: dict[str, Any]) -> ToolObservation:
        query = RetrievalQuery(
            text=str(args.get("query") or ""),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            authorship="user" if args.get("user_only", True) else None,
            limit=min(int(args.get("limit") or 6), 12),
        )
        hits = self.retriever.search(query)
        self._seen_atom_ids.update(hit.atom_id for hit in hits)
        return ToolObservation(
            tool="search_sources",
            data={
                "query": query.text,
                "hits": [self._hit_view(index, hit) for index, hit in enumerate(hits)],
            },
            boundary="只返回用户本人来源的候选；候选仍需 read_source 核对原文。",
        )

    # ── 2. 读取已发现来源的原文 ─────────────────────────────────────────
    def read_source(self, args: dict[str, Any]) -> ToolObservation:
        atom_id = int(args.get("atom_id") or 0)
        if atom_id not in self._seen_atom_ids:
            return ToolObservation(
                tool="read_source",
                data={"error": f"atom {atom_id} 不在本轮已发现列表中，必须先经 search_sources 发现"},
                boundary="只能读取本轮已发现的来源，防止越过检索直接翻库。",
            )
        row = self.database.fetchone(
            """
            SELECT a.*, s.title, s.rel_path, s.uid AS source_uid FROM source_atoms a
            JOIN sources s ON s.id = a.source_id WHERE a.id=?
            """,
            (atom_id,),
        )
        if row is None:
            return ToolObservation(tool="read_source", data={"error": "来源不存在"})
        return ToolObservation(
            tool="read_source",
            data={
                "atom_id": int(row["id"]),
                "title": row["title"],
                "path": row["rel_path"],
                "heading": row["heading"],
                "text": row["text"][:2000],
                "recorded_at": row["recorded_at"],
                "event_time": row["event_time"],
                "authorship": row["authorship"],
                "citation": f"[A{row['id']}]",
            },
            boundary="引用必须指向本轮真实返回的 [A{id}]。",
        )

    # ── 3. 主题时间线 ──────────────────────────────────────────────────
    def get_topic_timeline(self, args: dict[str, Any]) -> ToolObservation:
        topic = str(args.get("topic") or "")
        limit = min(int(args.get("limit") or 10), 20)
        query = RetrievalQuery(text=topic, authorship="user", limit=max(limit, 12))
        hits = self.retriever.search(query)
        self._seen_atom_ids.update(hit.atom_id for hit in hits)
        timeline = sorted(
            (
                {
                    "atom_id": hit.atom_id,
                    "date": hit.fields.get("event_time") or hit.fields.get("recorded_at"),
                    "excerpt": hit.fields.get("excerpt"),
                    "title": hit.fields.get("title"),
                    "tags": hit.fields.get("tags") or [],
                }
                for hit in hits
            ),
            key=lambda item: str(item["date"] or ""),
        )
        return ToolObservation(
            tool="get_topic_timeline",
            data={"topic": topic, "timeline": timeline[:limit]},
            boundary="时间线按记录时间排序；最近一条只是 recent 候选，不自动代表当前观点。",
        )

    def _snapshot_count(self) -> int:
        row = self.database.fetchone(
            "SELECT COUNT(*) AS n FROM stance_snapshots v JOIN source_atoms a ON a.id=v.atom_id "
            "JOIN sources s ON s.id=a.source_id WHERE v.has_stance=1 AND a.is_current=1 AND s.is_present=1"
        )
        return int(row["n"]) if row else 0

    # ── 4. 变化候选（定向配对：快照查询优先，启发式回退）────────────────
    def find_change_candidates(self, args: dict[str, Any]) -> ToolObservation:
        topic = str(args.get("topic") or "")
        limit = min(int(args.get("limit") or 5), 10)
        if not topic:
            return ToolObservation(
                tool="find_change_candidates",
                data={"error": "必须提供 topic；全库发现请用 discover_cognitive_shifts"},
                boundary="定向配对需要明确主题。",
            )
        candidates: list[dict[str, Any]] = []
        source = "heuristic"
        if self._snapshot_count():
            from .snapshots import DiscoveryEngine, candidate_to_dict

            engine = DiscoveryEngine(self.database)
            topics = engine.resolve_topics(topic)
            candidates = [
                candidate_to_dict(item)
                for item in engine.snapshot_candidates(topics=topics, limit=limit)
            ]
            source = "stance_snapshots"
        if not candidates:
            candidates = self._candidates_for_topic(topic, limit)
        return ToolObservation(
            tool="find_change_candidates",
            data={"candidates": candidates, "source": source},
            boundary="候选只说明'值得核对'；措辞差异不等于观点变化。",
        )

    # ── 4b. 全库主动发现（仅在用户未给出主题时开放）──────────────────────
    def discover_cognitive_shifts(self, args: dict[str, Any]) -> ToolObservation:
        limit = min(int(args.get("limit") or 5), 10)
        candidates: list[dict[str, Any]] = []
        source = "heuristic"
        if self._snapshot_count():
            from .snapshots import DiscoveryEngine, candidate_to_dict

            engine = DiscoveryEngine(self.database)
            candidates = [candidate_to_dict(item) for item in engine.run(limit=limit)]
            source = "stance_snapshots"
        if not candidates:
            candidates = self._discover_broad(limit)
        return ToolObservation(
            tool="discover_cognitive_shifts",
            data={"candidates": candidates, "source": source},
            boundary="只给候选；较近记录不作当前观点；差异信号只用于排序，不是变化概率。",
        )

    def _candidates_for_topic(self, topic: str, limit: int) -> list[dict[str, Any]]:
        observation = self.get_topic_timeline({"topic": topic, "limit": 12})
        timeline = observation.data.get("timeline", [])
        # 主题过滤：只保留标题或正文与主题词相关的条目，防止跨主题噪声参与配对
        terms = topic_terms(topic)
        if terms:
            def on_topic(item: dict[str, Any]) -> bool:
                haystack = f"{item.get('title') or ''} {item.get('excerpt') or ''}"
                return any(term in haystack for term in terms)

            timeline = [item for item in timeline if on_topic(item)]
        return self._pair_endpoints(topic, timeline, limit)

    def _discover_broad(self, limit: int) -> list[dict[str, Any]]:
        """全库发现：对高频共现主题生成端点对。只读启发式，排序信号非概率。"""
        rows = self.database.fetchall(
            """
            SELECT a.id, a.text, a.heading, COALESCE(a.event_time, a.recorded_at) AS moment,
                   s.title, s.tags_json FROM source_atoms a JOIN sources s ON s.id=a.source_id
            WHERE s.is_present=1 AND s.searchable=1 AND a.is_current=1 AND a.authorship='user'
              AND COALESCE(a.event_time, a.recorded_at) IS NOT NULL
            """
        )
        themes: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tags = " ".join(json.loads(str(row["tags_json"] or "[]")))
            for term in topic_terms(
                str(row["heading"] or "") + " " + str(row["title"] or "") + " " + tags
            ):
                if term in THEME_STOP:
                    continue
                themes.setdefault(term, []).append(
                    {
                        "atom_id": int(row["id"]),
                        "date": row["moment"],
                        "excerpt": str(row["text"])[:200],
                        "title": row["title"],
                    }
                )
        candidates: list[dict[str, Any]] = []
        for term, items in themes.items():
            if len(items) < 2:
                continue
            for pair in self._pair_endpoints(term, sorted(items, key=lambda x: str(x["date"])), 1):
                # 高词汇重叠 = 表达深化，不作为"变化候选"打扰用户
                if float(pair.get("lexical_overlap") or 0) < 0.35:
                    candidates.append(pair)
        # 同一端点对可能命中多个主题词（如"自主/判断"），只保留最高分主题
        best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        for pair in candidates:
            key = (int(pair["early"]["atom_id"]), int(pair["recent"]["atom_id"]))
            if key not in best_by_pair or float(pair.get("score") or 0) > float(
                best_by_pair[key].get("score") or 0
            ):
                best_by_pair[key] = pair
        candidates = sorted(
            best_by_pair.values(), key=lambda item: -float(item.get("score") or 0)
        )
        return candidates[:limit]

    def _pair_endpoints(
        self, topic: str, timeline: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        from .retrieval import tokenize

        candidates: list[dict[str, Any]] = []
        dated = sorted(
            (item for item in timeline if item.get("date")),
            key=lambda item: str(item["date"]),
        )
        if len(dated) >= 2 and self._shared_identity(dated[0], dated[-1]):
            # 回溯语义：最早端点 × 最近端点构成唯一对照对；中间节点属于区间
            pairs = [(dated[0], dated[-1])]
        else:
            pairs = []
        for early, recent in pairs:
            days = self._days_between(str(early["date"]), str(recent["date"]))
            if days is None or days < 7:
                continue
            early_terms = set(tokenize(str(early["excerpt"] or "")))
            recent_terms = set(tokenize(str(recent["excerpt"] or "")))
            diff = sorted(early_terms.symmetric_difference(recent_terms))[:8]
            union = early_terms | recent_terms
            overlap = (len(early_terms & recent_terms) / len(union)) if union else 1.0
            score = min(len(diff) / 10.0, 1.0) * min(days / 365.0, 1.0)
            candidates.append(
                {
                    "topic": topic,
                    "topic_key": fallback_topic_key(topic),
                    "early": early,
                    "recent": recent,
                    "recent_status": "latest_memory_candidate",
                    "diff_terms": diff,
                    "lexical_overlap": round(overlap, 3),
                    "score": round(score, 4),
                }
            )
        candidates.sort(key=lambda item: -float(item.get("score") or 0))
        for pair in candidates[:limit]:
            self._seen_atom_ids.update(
                {int(pair["early"]["atom_id"]), int(pair["recent"]["atom_id"])}
            )
        return candidates[:limit]

    @staticmethod
    def _shared_identity(early: dict[str, Any], recent: dict[str, Any]) -> bool:
        """端点配对的身份要求：共享标签，或共享非通用标题词。

        正文顺带提及不算身份——"没有更早的本人表达可供比较"这类元信息
        不能把两个不同主题的笔记配成对照。
        """
        early_tags = {str(tag) for tag in early.get("tags") or []}
        recent_tags = {str(tag) for tag in recent.get("tags") or []}
        if early_tags & recent_tags:
            return True
        early_terms = {
            term for term in topic_terms(str(early.get("title") or ""))
            if term not in THEME_STOP
        }
        recent_terms = {
            term for term in topic_terms(str(recent.get("title") or ""))
            if term not in THEME_STOP
        }
        return bool(early_terms & recent_terms)

    @staticmethod
    def _days_between(start: str, end: str) -> int | None:
        from datetime import date

        def parse(value: str) -> date | None:
            match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", value)
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None

        start_date, end_date = parse(start), parse(end)
        if not start_date or not end_date:
            return None
        return (end_date - start_date).days

    # ── 5. 区间经历 ────────────────────────────────────────────────────
    def find_interval_events(self, args: dict[str, Any]) -> ToolObservation:
        date_from = str(args.get("date_from") or "")
        date_to = str(args.get("date_to") or "")
        query_text = str(args.get("query") or args.get("topic") or "")
        if not date_from or not date_to:
            return ToolObservation(
                tool="find_interval_events",
                data={"error": "必须提供区间两端日期"},
                boundary="先建立时间区间，再搜索区间内经历。",
            )
        query = RetrievalQuery(
            text=query_text or "经历 决定 事件",
            date_from=date_from,
            date_to=date_to,
            authorship="user",
            limit=min(int(args.get("limit") or 6), 10),
        )
        hits = self.retriever.search(query)
        self._seen_atom_ids.update(hit.atom_id for hit in hits)
        return ToolObservation(
            tool="find_interval_events",
            data={
                "interval": [date_from, date_to],
                "events": [self._hit_view(index, hit) for index, hit in enumerate(hits)],
            },
            boundary="区间内的经历与变化只是时间相邻，不等于因果。",
        )

    # ── 6. 假设的正反证据 ───────────────────────────────────────────────
    def search_hypothesis_evidence(self, args: dict[str, Any]) -> ToolObservation:
        stance = str(args.get("stance") or "support")
        if stance not in {"support", "challenge"}:
            stance = "support"
        hypothesis = str(args.get("hypothesis") or "")
        base_query = str(args.get("query") or hypothesis)
        # stance 不能只是一个写进返回 JSON、却完全不影响检索的装饰字段。
        # challenge 路线加入反例词并以显式反例标签/转折表达作确定性重排；它仍只产生
        # “待核对的挑战候选”，不把词面信号冒充为真正反证。
        query_text = base_query
        if stance == "challenge":
            query_text = f"{base_query} 反例 例外 仍会 但是 不等于"
        query = RetrievalQuery(
            text=query_text,
            authorship=None,  # 反例可能藏在引用/草稿里，本工具不限作者归属
            limit=10,
        )
        hits = self.retriever.search(query)
        if stance == "challenge":
            ranked_hits = sorted(
                enumerate(hits),
                key=lambda item: (-self._challenge_signal_score(item[1]), item[0]),
            )
            hits = [hit for _, hit in ranked_hits]
        hits = hits[: min(int(args.get("limit") or 6), 10)]
        self._seen_atom_ids.update(hit.atom_id for hit in hits)
        return ToolObservation(
            tool="search_hypothesis_evidence",
            data={
                "hypothesis": hypothesis,
                "stance": stance,
                "hits": [self._hit_view(index, hit) for index, hit in enumerate(hits)],
            },
            boundary="单侧命中不能证明假设；必须两侧都查，挑战证据不得隐藏。",
        )

    @staticmethod
    def _challenge_signal_score(hit: Any) -> int:
        """对显式反例信号作稳定重排；仅用于候选排序，不是语义判决器。"""
        fields = hit.fields
        tags = {str(tag) for tag in fields.get("tags") or []}
        title = str(fields.get("title") or "")
        excerpt = str(fields.get("excerpt") or "")
        text = f"{title}\n{excerpt}"
        score = 8 if "反例" in tags else 0
        score += 5 if "反例" in title else 0
        for marker in ("仍会", "但是", "但", "不等于", "并非", "例外", "相反"):
            if marker in text:
                score += 1
        return score

    # ── 7. 用户历史判定 ────────────────────────────────────────────────
    def get_user_verdicts(self, args: dict[str, Any]) -> ToolObservation:
        text = str(args.get("topic") or "")
        topic_key = fallback_topic_key(text)
        rows = self.database.fetchall(
            "SELECT * FROM verdicts WHERE topic_key=? ORDER BY id DESC LIMIT 6", (topic_key,)
        )
        return ToolObservation(
            tool="get_user_verdicts",
            data={
                "topic": text,
                "verdicts": [
                    {
                        "verdict": row["verdict"],
                        "user_revision": row["user_revision"],
                        "confirmed_interpretation": row["confirmed_interpretation"],
                        "missing_event": row["missing_event"],
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ],
            },
            boundary="用户判定高于模型此前的结论；被否认的解释不得再次提出。",
        )

    @staticmethod
    def _hit_view(index: int, hit: Any) -> dict[str, Any]:
        view = dict(hit.fields)
        view["ref"] = f"[A{hit.atom_id}]"
        view["rank"] = index + 1
        return view


TOOL_BOUNDARIES = {
    "search_sources": "候选仍需核对原文",
    "read_source": "只能读取本轮已发现来源",
    "get_topic_timeline": "最近记录只是 recent 候选",
    "find_change_candidates": "措辞差异不等于变化",
    "discover_cognitive_shifts": "仅限未给出主题的发现场景",
    "find_interval_events": "时间相邻不等于因果",
    "search_hypothesis_evidence": "单侧命中不能证明假设",
    "get_user_verdicts": "用户判定优先于模型结论",
}


def build_tool_registry(
    database: Database,
    retriever: HybridRetriever,
    allow_discovery: bool,
    tools: CognitiveTools | None = None,
) -> dict[str, ToolSpec]:
    """构建本轮可用工具。allow_discovery=False 时移除全库发现能力（运行时收缩）。

    tools 实例由调用方注入以共享"本轮已发现来源"状态；缺省时自建。
    """
    tools = tools or CognitiveTools(database, retriever)
    registry: dict[str, ToolSpec] = {
        "search_sources": ToolSpec(
            name="search_sources",
            description="按关键词与时间窗检索用户本人的记录候选，返回 [A{id}] 编号",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=tools.search_sources,
        ),
        "read_source": ToolSpec(
            name="read_source",
            description="读取本轮已发现来源的原文与定位信息（引用前必须先读）",
            parameters={
                "type": "object",
                "properties": {"atom_id": {"type": "integer"}},
                "required": ["atom_id"],
            },
            handler=tools.read_source,
        ),
        "get_topic_timeline": ToolSpec(
            name="get_topic_timeline",
            description="按时间整理同一主题的历次表达，形成端点候选时间线",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["topic"],
            },
            handler=tools.get_topic_timeline,
        ),
        "find_change_candidates": ToolSpec(
            name="find_change_candidates",
            description="围绕明确主题提出变化候选端点对（最早端点 × 最近端点，recent 标记为候选）",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["topic"],
            },
            handler=tools.find_change_candidates,
        ),
        "discover_cognitive_shifts": ToolSpec(
            name="discover_cognitive_shifts",
            description=(
                "用户未给出主题时执行全库发现，提出可能被忽视的变化候选；"
                "仅在用户明确要求寻找未察觉变化时可用"
            ),
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
            handler=tools.discover_cognitive_shifts,
            discovery_only=True,
        ),
        "find_interval_events": ToolSpec(
            name="find_interval_events",
            description="在两个日期端点构成的区间内搜索经历/决定/事件候选",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["date_from", "date_to"],
            },
            handler=tools.find_interval_events,
        ),
        "search_hypothesis_evidence": ToolSpec(
            name="search_hypothesis_evidence",
            description="对原因假设分侧检索：stance=support 或 challenge，两侧都必须调用",
            parameters={
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "query": {"type": "string"},
                    "stance": {"type": "string", "enum": ["support", "challenge"]},
                },
                "required": ["hypothesis", "stance"],
            },
            handler=tools.search_hypothesis_evidence,
        ),
        "get_user_verdicts": ToolSpec(
            name="get_user_verdicts",
            description="读取用户对该主题此前的确认/否认判定，被否认的解释不得复用",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
            handler=tools.get_user_verdicts,
        ),
    }
    if not allow_discovery:
        registry = {
            name: spec for name, spec in registry.items() if not spec.discovery_only
        }
    return registry
