"""混合检索：BM25(trigram) + 向量(字符 n-gram / API embedding) → RRF 融合。

设计动机（对应《AI Agents in Depth》3.2 混合检索）：
- FTS5 trigram 免分词、对中文召回稳，但 <3 字的短词无法命中；
- 字符 2/3-gram 哈希向量是确定性的离线路线，覆盖短词与轻度同义改写；
- 两路用 Reciprocal Rank Fusion 融合，互为召回兜底；
- 配置了 embedding API 时自动升级为稠密向量，同一接口不变。
所有路线都是只读的，向量缓存存放在派生库 atom_vectors。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import jieba

from .db import Database

jieba.setLogLevel(60)

STOPWORDS = {
    "为什么", "怎么", "如何", "现在", "以前", "过去", "变化", "改变", "观点", "想法",
    "认为", "觉得", "看看", "我的", "这件事", "什么", "还是", "而且", "但是", "自己",
    "一个", "一些", "没有", "就是", "可以", "这个", "那个", "我们", "你们", "他们",
    "有没有", "是不是", "主题",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip().lower()
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens[:12]


@dataclass
class RetrievalQuery:
    text: str
    date_from: str | None = None
    date_to: str | None = None
    authorship: str | None = None  # None=全部；"user"=排除 quoted/ai_generated
    limit: int = 8


@dataclass
class RetrievalHit:
    atom_id: int
    source_id: int
    score: float
    route_scores: dict[str, float] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query: RetrievalQuery) -> list[RetrievalHit]: ...


def _parse_tags(raw: Any) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
        return [str(item) for item in value] if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


_ATOM_SELECT = """
SELECT a.id AS atom_id, a.source_id AS source_id, a.heading, a.text, a.authorship,
       a.recorded_at, a.event_time, a.seq,
       s.title, s.rel_path, s.uid AS source_uid, s.tags_json
FROM source_atoms a JOIN sources s ON s.id = a.source_id
WHERE s.is_present = 1 AND s.searchable = 1 AND a.authorship != 'derived'
"""

_ATOM_SELECT_FTS = """
SELECT a.id AS atom_id, a.source_id AS source_id, a.heading, a.text, a.authorship,
       a.recorded_at, a.event_time, a.seq,
       s.title, s.rel_path, s.uid AS source_uid, s.tags_json,
       bm25(source_atoms_fts) AS bm25_rank
FROM source_atoms_fts
JOIN source_atoms a ON a.id = source_atoms_fts.rowid
JOIN sources s ON s.id = a.source_id
WHERE s.is_present = 1 AND s.searchable = 1 AND a.authorship != 'derived'
"""


def _row_to_hit(row: Any, score: float, route: str) -> RetrievalHit:
    return RetrievalHit(
        atom_id=int(row["atom_id"]),
        source_id=int(row["source_id"]),
        score=round(score, 6),
        route_scores={route: round(score, 6)},
        fields={
            "atom_id": int(row["atom_id"]),
            "source_id": int(row["source_id"]),
            "source_uid": row["source_uid"],
            "title": row["title"],
            "path": row["rel_path"],
            "heading": row["heading"],
            "excerpt": row["text"][:400],
            "recorded_at": row["recorded_at"],
            "event_time": row["event_time"],
            "authorship": row["authorship"],
            "tags": _parse_tags(row["tags_json"]),
        },
    )


def _apply_common_filters(sql: list[str], params: list[Any], query: RetrievalQuery) -> None:
    if query.authorship == "user":
        sql.append("AND a.authorship = 'user'")
    if query.date_from:
        sql.append("AND COALESCE(a.event_time, a.recorded_at, '') >= ?")
        params.append(query.date_from)
    if query.date_to:
        sql.append("AND COALESCE(a.event_time, a.recorded_at, '9999') <= ?")
        params.append(query.date_to)


class BM25Retriever:
    """FTS5 trigram 关键词路线：短语精确匹配 + 短词 LIKE 兜底。"""

    def __init__(self, database: Database):
        self.database = database

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        runs = _CJK_RE.findall(query.text) + re.findall(r"[A-Za-z0-9]{2,}", query.text)
        phrases = [run for run in runs if len(run) >= 3]
        # trigram 分词器无法命中 <3 字 token；2 字词经 jieba 切出后走 LIKE 兜底
        short_terms: list[str] = []
        for token in tokenize(query.text):
            if len(token) == 2 and token not in short_terms:
                short_terms.append(token)
        if not phrases and not short_terms:
            return []
        hits: dict[int, RetrievalHit] = {}
        if phrases:
            match_expr = " OR ".join(f'"{p}"' for p in phrases[:6])
            sql = [_ATOM_SELECT_FTS, "AND source_atoms_fts MATCH ?"]
            params: list[Any] = [match_expr]
            _apply_common_filters(sql, params, query)
            sql.append("ORDER BY bm25_rank LIMIT ?")
            params.append(query.limit * 4)
            rows = self.database.fetchall("\n".join(sql), params)
            for rank, row in enumerate(rows):
                score = 1.0 / (1.0 + rank)
                hit = _row_to_hit(row, score, "bm25")
                hits[hit.atom_id] = hit
        for term in short_terms[:4]:
            like = f"%{term}%"
            sql = [_ATOM_SELECT, "AND (a.text LIKE ? OR s.title LIKE ? OR a.heading LIKE ?"
                   " OR s.tags_json LIKE ?)"]
            like_params: list[Any] = [like, like, like, like]
            _apply_common_filters(sql, like_params, query)
            sql.append("LIMIT ?")
            like_params.append(query.limit)
            for row in self.database.fetchall("\n".join(sql), like_params):
                if int(row["atom_id"]) in hits:
                    continue
                hit = _row_to_hit(row, 0.3, "bm25_like")
                hits[hit.atom_id] = hit
        return sorted(hits.values(), key=lambda h: h.score, reverse=True)[: query.limit]


class HashedNgramVectorizer:
    """确定性字符 2/3-gram 哈希向量（512 维，L2 归一化）。

    不依赖任何外部服务与随机初始化，同文本必得同向量；这是离线评测可复现的基础。
    """

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (2, 3)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _grams(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", "", text.lower())
        grams: list[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(normalized) < n:
                if normalized:
                    grams.append(normalized)
                continue
            grams.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
        return grams[:1200]

    def transform(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in self._grams(text):
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=False))


class VectorRetriever:
    """向量路线：优先 API 稠密 embedding，未配置时降级为哈希 n-gram。"""

    def __init__(self, database: Database, embedding_client: Any | None = None, dim: int = 512):
        self.database = database
        self.embedding_client = embedding_client
        self.local_vectorizer = HashedNgramVectorizer(dim=dim)

    def ensure_vectors(self, batch_size: int = 64) -> int:
        """为缺失向量的原子补齐缓存；返回补齐数量。幂等。嵌入文本含标题与标签。"""
        missing = self.database.fetchall(
            """
            SELECT a.id, a.text, s.title, s.tags_json FROM source_atoms a
            JOIN sources s ON s.id = a.source_id
            LEFT JOIN atom_vectors v ON v.atom_id = a.id
            WHERE v.atom_id IS NULL
            """
        )
        if not missing:
            return 0

        def embed_text(row: Any) -> str:
            tags = " ".join(json.loads(str(row["tags_json"] or "[]")))
            return f"{row['title']}\n{tags}\n{row['text']}"

        filled = 0
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            if self.embedding_client is not None:
                vectors = self.embedding_client.embed([embed_text(row) for row in batch])
            else:
                vectors = [self.local_vectorizer.transform(embed_text(row)) for row in batch]
            with self.database.transaction() as connection:
                for row, vector in zip(batch, vectors, strict=False):
                    connection.execute(
                        "INSERT OR REPLACE INTO atom_vectors(atom_id, dim, vector_json)"
                        " VALUES(?,?,?)",
                        (int(row["id"]), len(vector), json.dumps(vector)),
                    )
            filled += len(batch)
        return filled

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        query_vector = (
            self.embedding_client.embed_one(query.text)
            if self.embedding_client is not None
            else self.local_vectorizer.transform(query.text)
        )
        rows = self.database.fetchall(
            _ATOM_SELECT + " AND a.id IN (SELECT atom_id FROM atom_vectors)",
            (),
        )
        scored: list[tuple[float, Any]] = []
        for row in rows:
            vector_row = self.database.fetchone(
                "SELECT vector_json FROM atom_vectors WHERE atom_id=?", (int(row["atom_id"]),)
            )
            if vector_row is None:
                continue
            vector = json.loads(vector_row["vector_json"])
            if len(vector) != len(query_vector):
                continue
            scored.append((VectorRetriever._cosine(query_vector, vector), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, row in scored[: query.limit * 4]:
            if score <= 0.02:
                continue
            hits.append(_row_to_hit(row, score, "vector"))
        return self._filtered(hits, query)[: query.limit]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=False))

    @staticmethod
    def _filtered(hits: list[RetrievalHit], query: RetrievalQuery) -> list[RetrievalHit]:
        def keep(hit: RetrievalHit) -> bool:
            if query.authorship == "user" and hit.fields.get("authorship") != "user":
                return False
            moment = str(hit.fields.get("event_time") or hit.fields.get("recorded_at") or "")
            in_window = not (query.date_from and moment and moment < query.date_from) and not (
                query.date_to and moment and moment > query.date_to
            )
            return in_window

        return [hit for hit in hits if keep(hit)]


class HybridRetriever:
    """双路召回 + RRF 融合；route_scores 保留每路名次，评测可归因。"""

    RRF_K = 60

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        routes: Sequence[str] = ("bm25", "vector"),
    ):
        self.bm25 = bm25
        self.vector = vector
        self.routes = tuple(routes)

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        per_route: dict[str, list[RetrievalHit]] = {}
        if "bm25" in self.routes:
            per_route["bm25"] = self.bm25.search(query)
        if "vector" in self.routes:
            per_route["vector"] = self.vector.search(query)
        fused: dict[int, RetrievalHit] = {}
        for route, hits in per_route.items():
            for rank, hit in enumerate(hits):
                contribution = 1.0 / (self.RRF_K + rank + 1)
                if hit.atom_id in fused:
                    fused[hit.atom_id].score += contribution
                    fused[hit.atom_id].route_scores[route] = round(contribution, 6)
                else:
                    merged = RetrievalHit(
                        atom_id=hit.atom_id,
                        source_id=hit.source_id,
                        score=contribution,
                        route_scores={route: round(contribution, 6)},
                        fields=hit.fields,
                    )
                    fused[hit.atom_id] = merged
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)[: query.limit]
