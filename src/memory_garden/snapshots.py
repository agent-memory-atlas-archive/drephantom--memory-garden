"""立场快照层：把"发现"从运行时扫描变成对离线结构化数据的查询。

管道（对应"记忆花园 v2 发现架构"）：
1. 抽取（离线，导入后）：每条带日期的用户记录 → {主题, 立场, 原文引用, 语气强度, 是否本人观点}。
   "无立场"是合法输出；快照带 schema 版本，提示词迭代后可整版重抽。
2. 对比（离线/低频）：同主题按时间相邻两两比较 + 首尾对照（防慢漂移漏检），
   用变化分类学（措辞漂移/深化/真变化/并列新立场/语境性立场）而非词面差异；
   只有真变化与并列新立场成为候选。
3. 补充信号：显式对比句（"以前我以为…现在…"）精度极高，正则直取。
4. 评分（在线）：变化置信度 × 主题重要度 × 新颖度 × 用户反应权重，限额呈现。

诚实边界：主题归并的完整聚类评测、语气漂移、主题消失信号是 roadmap；
LLM 不可用时全部降级为确定性启发式（词面重叠映射到分类学）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .db import Database, utc_now
from .llm import LLMError
from .retrieval import tokenize
from .tools import THEME_STOP, topic_terms

SNAPSHOT_SCHEMA_VERSION = 2  # v2: 抽取语义变更——创作设定/世界观内容不算个人立场

CHANGE_TYPES = {
    "wording_drift": "措辞漂移（同义重述）",
    "deepening": "深化（同方向加细节）",
    "true_change": "真变化（方向反转）",
    "parallel_stance": "并列新立场（两套并存）",
    "contextual_stance": "语境性立场（特定人/场合）",
}
CANDIDATE_TYPES = {"true_change", "parallel_stance"}


def _calendar_days(start: str, end: str) -> int | None:
    """两个 ISO 日期之间的天数；解析失败返回 None。"""
    import re as _re
    from datetime import date

    def parse(value: str) -> date | None:
        match = _re.search(r"(20\d{2})-(\d{2})-(\d{2})", value)
        if not match:
            return None
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    start_date, end_date = parse(start), parse(end)
    if not start_date or not end_date:
        return None
    return (end_date - start_date).days


class StanceSnapshot(BaseModel):
    atom_id: int
    topic: str
    stance: str = ""
    has_stance: bool = True
    quote: str = ""
    tone_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    is_own_view: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SnapshotExtractor(Protocol):
    name: str

    def extract(self, items: list[dict[str, Any]]) -> list[StanceSnapshot]: ...


# ── 确定性抽取器：无 LLM 时的降级路线（也是 CI 的稳定基线）──────────────────


_META_RECORD_RE = re.compile(r"补记|混为一谈|没有因果证据|仅供测试|用于测试|待核对")


class DeterministicExtractor:
    """启发式快照：主题取标签/标题词，立场取原文首句。

    它不理解语义，只为离线测试与降级保持管道可运行；
    真实精度依赖 LLM 抽取器。
    """

    name = "deterministic"

    def extract(self, items: list[dict[str, Any]]) -> list[StanceSnapshot]:
        results: list[StanceSnapshot] = []
        for item in items:
            tags = [str(tag) for tag in item.get("tags") or [] if str(tag) not in THEME_STOP]
            text = str(item.get("text") or "")
            title = str(item.get("title") or "")
            topic_terms_found = [term for term in topic_terms(f"{title} {' '.join(tags)}") if term not in THEME_STOP]
            topic = (tags or topic_terms_found or ["未命名主题"])[0]
            first_sentence = re.split(r"[。；;！!？?\n]", text.strip())[0]
            is_meta = bool(_META_RECORD_RE.search(text))
            has_stance = (
                bool(text.strip()) and not is_meta
                and str(item.get("authorship") or "user") == "user"
            )
            results.append(
                StanceSnapshot(
                    atom_id=int(item["atom_id"]),
                    topic=topic,
                    stance=(first_sentence[:120] if has_stance else ""),
                    has_stance=has_stance,
                    quote=text.strip()[:160],
                    tone_strength=0.5,
                    is_own_view=str(item.get("authorship") or "user") == "user",
                    confidence=0.4,
                )
            )
        return results


class LLMBatchExtractor:
    """LLM 批量抽取：每批一次调用，附带已知主题表促进别名收敛。

    主题归并的完整聚类评测是 roadmap；这里用"优先沿用已知主题"实现轻量版本。
    """

    name = "llm"

    BATCH_SIZE = 30

    def __init__(self, client: Any, model: str, known_topics: list[str] | None = None):
        self.client = client
        self.model = model
        self.known_topics = known_topics or []

    @staticmethod
    def _prompt(items: list[dict[str, Any]], known_topics: list[str]) -> str:
        payload = [
            {
                "id": item["atom_id"],
                "date": item.get("date") or "",
                "title": item.get("title") or "",
                "tags": item.get("tags") or [],
                "text": str(item.get("text") or "")[:600],
            }
            for item in items
        ]
        return (
            "你是个人记忆分析师。对下面每条带日期的个人记录，抽取一个立场快照。\n"
            '输出一个 JSON 对象，顶层键为 "snapshots"、值为数组；数组每个元素对应一条记录，\n'
            '元素字段：{"id": 原样返回, "topic": "稳定主题(2-6字, 优先沿用已知主题表; 确无合适主题才新建)",'
            ' "stance": "该记录表达的观点立场(一句话)", "has_stance": true/false,'
            ' "quote": "最能代表立场的原文(<=60字)", "tone_strength": 0.0-1.0(0犹疑-1笃定),'
            ' "is_own_view": true/false(引用他人/书摘为false), "confidence": 0.0-1.0}\n'
            "规则：记录不是个人观点时 has_stance=false——包括引用/书摘、AI 草稿、纯事实记录，"
            "以及小说设定/世界观构建/剧情设计等创作内容（那是作品设定，不是你对现实生活的立场）；"
            "不要编造记录里没有的立场。\n\n"
            f"已知主题表: {json.dumps(known_topics, ensure_ascii=False)}\n\n"
            f"记录列表:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def extract(self, items: list[dict[str, Any]]) -> list[StanceSnapshot]:
        if not items:
            return []
        response = self.client.chat_json(
            "你是严谨的信息抽取器，只输出 JSON。",
            self._prompt(items, self.known_topics),
        )
        raw = response.get("snapshots") if isinstance(response.get("snapshots"), list) else response.get("data")
        if not isinstance(raw, list):
            raise LLMError("快照抽取响应缺少 snapshots 数组")
        by_id = {int(item["atom_id"]): item for item in items}
        results: list[StanceSnapshot] = []
        for entry in raw:
            try:
                atom_id = int(entry["id"])
                source = by_id.get(atom_id)
                if source is None:
                    continue
                has_stance = bool(entry.get("has_stance", True))
                results.append(
                    StanceSnapshot(
                        atom_id=atom_id,
                        topic=str(entry.get("topic") or "未命名主题")[:24],
                        stance=str(entry.get("stance") or "")[:200],
                        has_stance=has_stance,
                        quote=str(entry.get("quote") or "")[:160],
                        tone_strength=float(entry.get("tone_strength") or 0.5),
                        is_own_view=bool(entry.get("is_own_view", True)),
                        confidence=float(entry.get("confidence") or 0.5),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return results


def ensure_snapshots(database: Database, extractor: SnapshotExtractor) -> dict[str, int]:
    """为缺失快照的原子补齐（按 schema 版本）。内容变化的原子 id 本身会变，
    历史快照保留以供追溯，但不再参加当前候选发现。"""
    missing = database.fetchall(
        """
        SELECT a.id AS atom_id, a.text, a.authorship, a.heading,
               COALESCE(a.event_time, a.recorded_at) AS date,
               s.title, s.tags_json
        FROM source_atoms a JOIN sources s ON s.id = a.source_id
        LEFT JOIN stance_snapshots v
               ON v.atom_id = a.id AND v.snapshot_schema_version = ?
        WHERE s.is_present = 1 AND s.searchable = 1 AND a.is_current = 1 AND v.atom_id IS NULL
        """,
        (SNAPSHOT_SCHEMA_VERSION,),
    )
    if not missing:
        return {"extracted": 0, "skipped": 0}
    items = [
        {
            "atom_id": int(row["atom_id"]),
            "text": row["text"],
            "authorship": row["authorship"],
            "title": row["title"],
            "tags": json.loads(str(row["tags_json"] or "[]")),
            "date": row["date"],
        }
        for row in missing
    ]
    extracted_total = 0
    skipped = 0
    for start in range(0, len(items), getattr(extractor, "BATCH_SIZE", 200)):
        batch = items[start : start + getattr(extractor, "BATCH_SIZE", 200)]
        try:
            snapshots = extractor.extract(batch)
        except LLMError:
            skipped += len(batch)
            continue
        with database.transaction() as connection:
            for snapshot in snapshots:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO stance_snapshots(atom_id, snapshot_schema_version,
                        topic, stance, has_stance, quote, tone_strength, is_own_view,
                        confidence, extractor, extracted_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        snapshot.atom_id, SNAPSHOT_SCHEMA_VERSION, snapshot.topic,
                        snapshot.stance, int(snapshot.has_stance), snapshot.quote,
                        snapshot.tone_strength, int(snapshot.is_own_view),
                        snapshot.confidence, extractor.name, utc_now(),
                    ),
                )
        extracted_total += len(snapshots)
        skipped += len(batch) - len(snapshots)
    return {"extracted": extracted_total, "skipped": skipped}


# ── 变化分类学：相邻对 + 首尾对照 ───────────────────────────────────────────


class ChangeClassifier(Protocol):
    name: str

    def classify(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class DeterministicChangeClassifier:
    """词面重叠 → 分类学的确定性映射（降级路线，语义判断仍属 LLM/用户）。"""

    name = "deterministic"

    def classify(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pair in pairs:
            overlap = float(pair.get("lexical_overlap") or 0.0)
            tone_delta = abs(
                float(pair.get("early_tone") or 0.5) - float(pair.get("recent_tone") or 0.5)
            )
            if overlap >= 0.35:
                change_type, confidence = "wording_drift", min(0.6 + tone_delta / 4, 0.85)
            elif overlap >= 0.18:
                change_type, confidence = "deepening", 0.5
            else:
                change_type, confidence = "true_change", min(0.5 + tone_delta / 3, 0.8)
            results.append({"change_type": change_type, "confidence": round(confidence, 3)})
        return results


class LLMChangeClassifier:
    name = "llm"

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _prompt(pairs: list[dict[str, Any]]) -> str:
        return (
            "你是认知变化分析员。对每对同一主题、不同时间的立场快照，判断变化类型。\n"
            f"分类学: {json.dumps(CHANGE_TYPES, ensure_ascii=False)}\n"
            "输出 JSON: {\"pairs\":[{\"id\": 原样返回, \"change_type\": 类别, "
            "\"confidence\": 0.0-1.0}]}\n"
            "注意：只有方向反转或立场被替换才是 true_change；同一方向补充细节是 deepening；"
            "同义换说法是 wording_drift；两套立场并存是 parallel_stance；"
            "仅对特定人/场合成立的立场是 contextual_stance。\n\n"
            f"快照对:\n{json.dumps(pairs, ensure_ascii=False)}"
        )

    def classify(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not pairs:
            return []
        payload = [
            {
                "id": index,
                "topic": pair.get("topic"),
                "early_date": pair.get("early_date"),
                "early_stance": pair.get("early_stance"),
                "recent_date": pair.get("recent_date"),
                "recent_stance": pair.get("recent_stance"),
            }
            for index, pair in enumerate(pairs)
        ]
        try:
            response = self.client.chat_json(
                "你是严谨的分析员，只输出 JSON。", self._prompt(payload)
            )
            entries = response.get("pairs")
            if not isinstance(entries, list):
                raise LLMError("缺少 pairs 数组")
        except LLMError:
            return DeterministicChangeClassifier().classify(pairs)
        fallback = DeterministicChangeClassifier().classify(pairs)
        results: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or int(entry.get("id", -1)) != index:
                results.append(fallback[index])
                continue
            change_type = str(entry.get("change_type") or "")
            if change_type not in CHANGE_TYPES:
                results.append(fallback[index])
                continue
            results.append(
                {
                    "change_type": change_type,
                    "confidence": float(entry.get("confidence") or 0.5),
                }
            )
        return results


# ── 显式对比句：高精度补充信号 ─────────────────────────────────────────────

_CONTRAST_RE = re.compile(
    r"(以前|过去|曾经|原本|当初)[^。？！\n]{0,36}?[，,][^。？！\n]{0,10}?(现在|如今|这两年|最近)"
)


def contains_explicit_contrast(text: str) -> bool:
    """只标识文字中出现的前后对照，不据此确认当前立场或变化原因。"""
    return _CONTRAST_RE.search(text) is not None


def find_contrast_sentences(database: Database, limit: int = 5) -> list[dict[str, Any]]:
    """显式对比句检测：正则直取 + 与检索索引相同的来源过滤。

    精度极高（作者本人在原句里承认了变化），因此直接成为候选，置信度 0.85。
    """
    rows = database.fetchall(
        """
        SELECT a.id AS atom_id, a.text, a.authorship,
               COALESCE(a.event_time, a.recorded_at) AS moment,
               s.title, s.rel_path
        FROM source_atoms a JOIN sources s ON s.id = a.source_id
        WHERE s.is_present = 1 AND s.searchable = 1 AND a.is_current = 1 AND a.authorship = 'user'
        """
    )
    hits: list[dict[str, Any]] = []
    for row in rows:
        text = str(row["text"] or "")
        match = _CONTRAST_RE.search(text)
        if not match:
            continue
        sentence = match.group(0)
        start = max(match.start() - 20, 0)
        end = min(match.end() + 40, len(text))
        hits.append(
            {
                "atom_id": int(row["atom_id"]),
                "sentence": sentence.strip(),
                "context": text[start:end].strip(),
                "date": row["moment"],
                "title": row["title"],
                "path": row["rel_path"],
            }
        )
        if len(hits) >= limit * 3:
            break
    return hits[:limit]


# ── 发现引擎：快照对 + 显式对比句 → 评分 → 限额 ─────────────────────────────


def candidate_to_dict(candidate: DiscoveryCandidate) -> dict[str, Any]:
    return {
        "topic": candidate.topic,
        "topic_key": candidate.topic_key,
        "early": candidate.early,
        "recent": candidate.recent,
        "change_type": candidate.change_type,
        "change_confidence": candidate.change_confidence,
        "signal_type": candidate.signal_type,
        "diff_terms": candidate.diff_terms,
        "lexical_overlap": candidate.lexical_overlap,
        "score": candidate.score,
    }


@dataclass
class DiscoveryCandidate:
    topic: str
    topic_key: str
    early: dict[str, Any]
    recent: dict[str, Any]
    change_type: str
    change_confidence: float
    signal_type: str
    diff_terms: list[str]
    lexical_overlap: float
    score: float


class DiscoveryEngine:
    """对快照表做发现查询。快照不存在时由调用方回退到旧的启发式扫描。"""

    def __init__(
        self,
        database: Database,
        classifier: ChangeClassifier | None = None,
    ):
        self.database = database
        self.classifier = classifier or DeterministicChangeClassifier()

    def _topic_snapshots(self, topics: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
        sql = (
            "SELECT v.atom_id, v.topic, v.stance, v.quote, v.tone_strength, v.is_own_view,"
            " v.confidence, COALESCE(a.event_time, a.recorded_at) AS moment, s.title,"
            " s.rel_path, s.tags_json FROM stance_snapshots v"
            " JOIN source_atoms a ON a.id = v.atom_id JOIN sources s ON s.id = a.source_id"
            " WHERE v.has_stance = 1 AND v.is_own_view = 1"
            " AND s.is_present = 1 AND s.searchable=1 AND a.is_current=1"
            " AND COALESCE(a.event_time, a.recorded_at) IS NOT NULL"
        )
        params: list[Any] = []
        if topics:
            sql += f" AND v.topic IN ({','.join('?' * len(topics))})"
            params.extend(topics)
        sql += " ORDER BY moment"
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self.database.fetchall(sql, params):
            groups.setdefault(str(row["topic"]), []).append(dict(row))
        return groups

    def _build_pairs(self, groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for topic, snaps in groups.items():
            if len(snaps) < 2:
                continue
            # 相邻对（定位转折点）+ 首尾对照（防慢漂移漏检），去重
            candidates_idx: list[tuple[int, int]] = []
            for i in range(len(snaps) - 1):
                candidates_idx.append((i, i + 1))
            if snaps[0] is not snaps[-1]:
                candidates_idx.append((0, len(snaps) - 1))
            seen: set[tuple[int, int]] = set()
            for i, j in candidates_idx:
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                early, recent = snaps[i], snaps[j]
                gap_days = _calendar_days(str(early["moment"]), str(recent["moment"]))
                if gap_days is None or gap_days < 7:
                    continue  # 同日/邻日条目是文档结构或密集灵感，不是时间演化
                early_terms = set(tokenize(str(early["stance"] or "")))
                recent_terms = set(tokenize(str(recent["stance"] or "")))
                union = early_terms | recent_terms
                overlap = (len(early_terms & recent_terms) / len(union)) if union else 1.0
                pairs.append(
                    {
                        "topic": topic,
                        "early_atom_id": int(early["atom_id"]),
                        "recent_atom_id": int(recent["atom_id"]),
                        "early_date": early["moment"],
                        "recent_date": recent["moment"],
                        "early_stance": early["stance"],
                        "recent_stance": recent["stance"],
                        "early_quote": early["quote"],
                        "recent_quote": recent["quote"],
                        "early_title": early["title"],
                        "recent_title": recent["title"],
                        "early_tone": early["tone_strength"],
                        "recent_tone": recent["tone_strength"],
                        "lexical_overlap": round(overlap, 3),
                        "diff_terms": sorted(early_terms.symmetric_difference(recent_terms))[:8],
                    }
                )
        return pairs

    def resolve_topics(self, text: str) -> list[str]:
        """把用户问题里的主题词解析到快照主题实体（词面包含即匹配，别名归并是 roadmap）。"""
        terms = topic_terms(text)
        if not terms:
            return []
        rows = self.database.fetchall(
            "SELECT DISTINCT v.topic FROM stance_snapshots v JOIN source_atoms a ON a.id=v.atom_id "
            "JOIN sources s ON s.id=a.source_id WHERE a.is_current=1 AND s.is_present=1 AND s.searchable=1"
        )
        return [
            str(row["topic"])
            for row in rows
            if any(term in str(row["topic"]) or str(row["topic"]) in term for term in terms)
        ]

    def snapshot_candidates(
        self, topics: list[str] | None = None, limit: int = 5
    ) -> list[DiscoveryCandidate]:
        groups = self._topic_snapshots(topics)
        if topics is not None:
            # 作用域查询：解析不到主题时必须返回空，绝不放行全库
            groups = {key: value for key, value in groups.items() if key in topics}
        pairs = self._build_pairs(groups)
        classified = self.classifier.classify(pairs)
        importance = self._topic_importance(groups)
        candidates: list[DiscoveryCandidate] = []
        for pair, verdict in zip(pairs, classified, strict=False):
            if verdict["change_type"] not in CANDIDATE_TYPES:
                continue
            confidence = float(verdict["confidence"])
            score = confidence * importance.get(pair["topic"], 0.5) * self._novelty(pair) \
                * self._reaction_weight(pair["topic"])
            candidates.append(
                DiscoveryCandidate(
                    topic=str(pair["topic"]),
                    topic_key="snap:" + str(pair["topic"]),
                    early={
                        "atom_id": pair["early_atom_id"], "date": pair["early_date"],
                        "excerpt": pair["early_stance"] or pair["early_quote"],
                        "title": pair["early_title"], "quote": pair["early_quote"],
                    },
                    recent={
                        "atom_id": pair["recent_atom_id"], "date": pair["recent_date"],
                        "excerpt": pair["recent_stance"] or pair["recent_quote"],
                        "title": pair["recent_title"], "quote": pair["recent_quote"],
                    },
                    change_type=str(verdict["change_type"]),
                    change_confidence=confidence,
                    signal_type="stance_pair",
                    diff_terms=pair["diff_terms"],
                    lexical_overlap=float(pair["lexical_overlap"]),
                    score=round(score, 4),
                )
            )
        candidates.sort(key=lambda item: -item.score)
        return candidates[:limit]

    def contrast_candidates(self, limit: int = 3) -> list[DiscoveryCandidate]:
        results: list[DiscoveryCandidate] = []
        for hit in find_contrast_sentences(self.database, limit=limit):
            source_row = self.database.fetchone(
                """
                SELECT s.tags_json FROM source_atoms a JOIN sources s ON s.id = a.source_id
                WHERE a.id = ?
                """,
                (int(hit["atom_id"]),),
            )
            tags = [
                str(tag) for tag in json.loads(str(source_row["tags_json"] or "[]"))
                if str(tag) not in THEME_STOP
            ] if source_row else []
            title_terms = [
                term for term in topic_terms(str(hit.get("title") or ""))
                if term not in THEME_STOP
            ]
            sentence_terms = [
                term for term in topic_terms(str(hit.get("sentence") or ""))
                if term not in THEME_STOP
            ]
            topic = (tags or title_terms or sentence_terms or ["显式对比"])[0]
            results.append(
                DiscoveryCandidate(
                    topic=topic,
                    topic_key="contrast:" + str(hit["atom_id"]),
                    early={
                        "atom_id": int(hit["atom_id"]), "date": hit.get("date"),
                        "excerpt": str(hit.get("sentence") or ""),
                        "title": hit.get("title"), "quote": str(hit.get("sentence") or ""),
                    },
                    recent={
                        "atom_id": int(hit["atom_id"]), "date": hit.get("date"),
                        "excerpt": str(hit.get("sentence") or ""),
                        "title": hit.get("title"), "quote": str(hit.get("sentence") or ""),
                    },
                    change_type="true_change",
                    change_confidence=0.85,
                    signal_type="explicit_contrast",
                    diff_terms=[],
                    lexical_overlap=0.0,
                    score=0.85 * 0.9 * self._novelty(hit) * self._reaction_weight(topic),
                )
            )
        return results[:limit]

    # ── 评分因子 ────────────────────────────────────────────────────────
    @staticmethod
    def _topic_importance(groups: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
        """主题重要度：快照数量归一（出现越多对你越重要），上限 1。"""
        total = sum(len(snaps) for snaps in groups.values()) or 1
        return {
            topic: min(len(snaps) / max(total / 4, 1), 1.0)
            for topic, snaps in groups.items()
        }

    def _novelty(self, pair: dict[str, Any]) -> float:
        """新颖度：最近 7 天内已呈现过的主题降权，避免重复打扰。"""
        row = self.database.fetchone(
            """
            SELECT last_shown_at FROM discoveries
            WHERE topic_key = ? OR topic_key = ?
            ORDER BY last_shown_at DESC LIMIT 1
            """,
            ("snap:" + str(pair.get("topic") or ""), "contrast:" + str(pair.get("atom_id") or "")),
        )
        if row and row["last_shown_at"]:
            from datetime import datetime, timedelta

            try:
                shown = datetime.fromisoformat(str(row["last_shown_at"]).replace("Z", "+00:00"))
                if datetime.now(UTC) - shown < timedelta(days=7):
                    return 0.3
            except ValueError:
                pass
        return 1.0

    def _reaction_weight(self, topic: str) -> float:
        """用户反应写回：属实加权，不是/无聊降权。这是第五层个性化排序。"""
        rows = self.database.fetchall(
            """
            SELECT r.reaction FROM candidate_reactions r
            JOIN discoveries d ON d.id = r.discovery_id
            WHERE d.topic_key = ?
            ORDER BY r.id DESC LIMIT 10
            """,
            ("snap:" + topic,),
        )
        weight = 1.0
        for row in rows:
            if row["reaction"] == "accurate":
                weight *= 1.15
            elif row["reaction"] == "wrong":
                weight *= 0.6
            elif row["reaction"] == "boring":
                weight *= 0.8
        return min(weight, 2.0)

    def run(self, limit: int = 5) -> list[DiscoveryCandidate]:
        """合并两路信号，按分排序，限额呈现。每周只打扰用户 3-5 条。"""
        candidates = self.snapshot_candidates(limit=limit * 2)
        candidates += self.contrast_candidates(limit=max(1, limit // 2))
        candidates.sort(key=lambda item: -item.score)
        return candidates[:limit]
