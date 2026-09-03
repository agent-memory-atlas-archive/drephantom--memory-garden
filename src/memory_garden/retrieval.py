"""可切换检索：BM25、字符 n-gram、Embedding 与 RRF 混合。

设计动机（对应《AI Agents in Depth》3.2 混合检索）：
- FTS5 trigram 免分词、对中文召回稳，但 <3 字的短词无法命中；
- 字符 2/3-gram 哈希向量是确定性的离线路线，覆盖短词与轻度同义改写；
- 两路用 Reciprocal Rank Fusion 融合，互为召回兜底；
- API Embedding 只有在显式选择并允许云端发送后才会启用；
- mock 只用于公开 CI/合成评测，不代表真实语义模型效果。
所有路线都是只读的，向量缓存存放在派生库 atom_vectors。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import jieba

from .config import Settings
from .db import Database, utc_now

jieba.setLogLevel(60)

EMBEDDING_TEXT_VERSION = "v2-heading"
RETRIEVAL_MODES = {"bm25", "hash_vector", "embedding", "hybrid"}
EMBEDDING_BACKENDS = {"local_hash", "mock", "api"}
RERANKER_BACKENDS = {"none", "local_heuristic", "api"}
RERANKER_FUSIONS = {"replace", "rank_fusion"}
DEFAULT_RRF_K = 60

STOPWORDS = {
    "为什么", "怎么", "如何", "现在", "以前", "过去", "变化", "改变", "观点", "想法",
    "认为", "觉得", "看看", "我的", "这件事", "什么", "还是", "而且", "但是", "自己",
    "一个", "一些", "没有", "就是", "可以", "这个", "那个", "我们", "你们", "他们",
    "有没有", "是不是", "主题",
}

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


class EmbeddingBackend(Protocol):
    provider: str
    model: str
    dimension: int | None
    is_cloud: bool

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    name: str
    candidate_limit: int

    def rerank(
        self, query: RetrievalQuery, candidates: list[RetrievalHit]
    ) -> list[RetrievalHit]: ...


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


def embedding_text(row: Any) -> str:
    """缓存与 rerank 的唯一规范：标题、标题层级、标签、正文。"""
    tags = " ".join(_parse_tags(row["tags_json"]))
    return f"{row['title']}\n{row['heading'] or ''}\n{tags}\n{row['text']}"


def embedding_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_vector(
    raw: Any, *, expected_dimension: int | None = None, context: str = "embedding"
) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{context} 必须是非空数值数组")
    try:
        vector = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} 包含非数值元素") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{context} 包含非有限数值")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(
            f"{context} 维度 {len(vector)} 与期望维度 {expected_dimension} 不一致"
        )
    return vector


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
        # 不能把整段连续中文问句当作一个 FTS phrase：那会要求正文包含整句原文，
        # 对自然语言问题几乎等同于精确字符串检索。这里按 jieba 词项拆分，
        # 3 字及以上走 trigram FTS，2 字词及标题/标签命中走 LIKE 兜底。
        terms = tokenize(query.text)
        for token in re.findall(r"[A-Za-z0-9_]{2,}", query.text.lower()):
            if token not in terms:
                terms.append(token)
        fts_terms = [term for term in terms if len(term) >= 3][:8]
        like_terms = terms[:8]
        if not fts_terms and not like_terms:
            return []
        hits: dict[int, RetrievalHit] = {}
        if fts_terms:
            match_expr = " OR ".join(f'"{term}"' for term in fts_terms)
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
        # FTS 表只索引 atom 正文；LIKE 同时覆盖标题、标题层级与标签，
        # 也补齐 trigram 无法处理的 2 字中文词。
        for term in like_terms:
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


class LocalHashEmbeddingBackend:
    """默认离线后端：沿用字符 2/3-gram 哈希向量。"""

    provider = "local_hash"
    model = "char-ngram-2-3-v1"
    is_cloud = False

    def __init__(self, dimension: int = 512):
        self.dimension: int | None = dimension
        self.vectorizer = HashedNgramVectorizer(dim=dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectorizer.transform(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.vectorizer.transform(text)


class MockEmbeddingBackend:
    """公开测试专用的固定模拟 Embedding；它不是模型效果替代品。"""

    provider = "mock"
    is_cloud = False
    _CONCEPTS: tuple[tuple[str, ...], ...] = (
        ("自主判断", "自己的判断", "相信自己", "决定权", "判断权", "外界认可", "别人认可"),
        ("听取别人意见", "参考他人", "朋友求证", "询问朋友", "检查盲点", "不等于拒绝"),
        ("从不寻求支持", "永远不需要任何人", "拒绝任何意见"),
        ("独处", "一个人", "安静时间", "恢复精力", "注意力充电", "重新充电", "避免消息打扰"),
        ("手机", "系统设置", "恢复出厂", "消息通知", "通知设置"),
        ("先完成", "可用版本", "可用成果", "原型", "再修正", "逐步打磨", "迭代"),
        ("完美主义", "完全没有漏洞", "一次完美"),
        ("谨慎决策", "重要决定", "重要选择", "收集信息", "了解情况", "一时冲动"),
        ("数据库", "缓存", "字段", "保留期限", "数据项目"),
        ("咖啡", "咖啡豆"),
        ("书摘", "引用", "书中"),
        ("AI草稿", "AI 生成", "生成的草稿"),
        ("写作", "读者诚实", "打动自己"),
        ("边界", "说出边界"),
    )

    def __init__(self, dimension: int = 64, model: str = "mock-semantic-v1"):
        if dimension <= len(self._CONCEPTS):
            raise ValueError(f"mock embedding 维度必须大于 {len(self._CONCEPTS)}")
        self.dimension: int | None = dimension
        self.model = model

    def _transform(self, text: str) -> list[float]:
        assert self.dimension is not None
        normalized = re.sub(r"\s+", "", text.lower())
        vector = [0.0] * self.dimension
        for index, phrases in enumerate(self._CONCEPTS):
            if any(re.sub(r"\s+", "", phrase.lower()) in normalized for phrase in phrases):
                vector[index] = 3.0
        # 小权重字符特征让未列入词典的精确匹配仍可检索；语义概念占主导。
        tail_start = len(self._CONCEPTS)
        tail_dim = self.dimension - tail_start
        for n in (2, 3):
            for offset in range(max(0, len(normalized) - n + 1)):
                gram = normalized[offset : offset + n]
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                index = tail_start + int.from_bytes(digest[:4], "little") % tail_dim
                vector[index] += 0.08 if digest[4] % 2 == 0 else -0.08
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 8) for value in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._transform(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._transform(text)


class VectorRetriever:
    """单一明确 backend 的向量路线，缓存身份与失效规则集中在这里。"""

    def __init__(
        self,
        database: Database,
        embedding_backend: EmbeddingBackend | None = None,
        dim: int = 512,
        text_version: str = EMBEDDING_TEXT_VERSION,
    ):
        self.database = database
        self.backend: EmbeddingBackend = embedding_backend or LocalHashEmbeddingBackend(dim)
        self.embedding_client = self.backend  # 兼容旧调用方的只读属性
        self.text_version = text_version
        self.route_name = "hash_vector" if self.backend.provider == "local_hash" else "embedding"
        # 只累计发送过的 Vault 原子文本；查询文本不计入 private_vault_sent。
        self.private_payload_sent_count = 0

    @property
    def is_cloud(self) -> bool:
        return self.backend.is_cloud

    def _cache_rows(self) -> dict[int, Any]:
        rows = self.database.fetchall(
            """
            SELECT atom_id, embedding_dimension, embedding_text_hash, vector_json
            FROM atom_vectors
            WHERE embedding_provider=? AND embedding_model=? AND embedding_text_version=?
            """,
            (self.backend.provider, self.backend.model, self.text_version),
        )
        return {int(row["atom_id"]): row for row in rows}

    @staticmethod
    def _cached_vector(row: Any, text_hash: str) -> tuple[list[float], int] | None:
        if row is None or str(row["embedding_text_hash"]) != text_hash:
            return None
        try:
            dimension = int(row["embedding_dimension"])
            vector = _validated_vector(
                json.loads(str(row["vector_json"])),
                expected_dimension=dimension,
                context="缓存 embedding",
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return vector, dimension

    def ensure_vectors(self, batch_size: int = 64, expected_dimension: int | None = None) -> int:
        """补齐或重建当前身份的缓存；文本、模型、维度、版本任一变化都会失效。"""
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        atoms = self.database.fetchall(_ATOM_SELECT)
        cache = self._cache_rows()
        prepared: list[tuple[Any, str, str, tuple[list[float], int] | None]] = []
        cached_dimensions: set[int] = set()
        for row in atoms:
            text = embedding_text(row)
            text_hash = embedding_text_hash(text)
            cached = self._cached_vector(cache.get(int(row["atom_id"])), text_hash)
            if cached is not None:
                cached_dimensions.add(cached[1])
            prepared.append((row, text, text_hash, cached))

        configured_dimension = self.backend.dimension
        if expected_dimension is not None and configured_dimension not in {None, expected_dimension}:
            raise ValueError(
                f"查询向量维度 {expected_dimension} 与后端声明维度 {configured_dimension} 不一致"
            )
        resolved_dimension = expected_dimension or configured_dimension
        if resolved_dimension is None and len(cached_dimensions) == 1:
            resolved_dimension = next(iter(cached_dimensions))
        pending = [
            item for item in prepared
            if item[3] is None or resolved_dimension is None or item[3][1] != resolved_dimension
        ]
        if not pending:
            return 0

        filled = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            if self.is_cloud:
                # 即使请求随后失败，文本也已经离开本机，因此在发起调用前记账。
                self.private_payload_sent_count += len(batch)
            raw_vectors = self.backend.embed([item[1] for item in batch])
            if len(raw_vectors) != len(batch):
                raise ValueError(
                    f"embedding 返回 {len(raw_vectors)} 条，期望 {len(batch)} 条"
                )
            vectors: list[list[float]] = []
            for raw_vector in raw_vectors:
                vector = _validated_vector(
                    raw_vector,
                    expected_dimension=resolved_dimension,
                    context="provider embedding",
                )
                if resolved_dimension is None:
                    resolved_dimension = len(vector)
                vectors.append(vector)
            now = utc_now()
            with self.database.transaction() as connection:
                for (row, _text, text_hash, _cached), vector in zip(
                    batch, vectors, strict=True
                ):
                    connection.execute(
                        """
                        INSERT INTO atom_vectors(
                            atom_id, embedding_provider, embedding_model,
                            embedding_dimension, embedding_text_version,
                            embedding_text_hash, vector_json, created_at, updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(
                            atom_id, embedding_provider, embedding_model, embedding_text_version
                        ) DO UPDATE SET
                            embedding_dimension=excluded.embedding_dimension,
                            embedding_text_hash=excluded.embedding_text_hash,
                            vector_json=excluded.vector_json,
                            updated_at=excluded.updated_at
                        """,
                        (
                            int(row["atom_id"]), self.backend.provider, self.backend.model,
                            len(vector), self.text_version, text_hash,
                            json.dumps(vector, separators=(",", ":")), now, now,
                        ),
                    )
            filled += len(batch)
        return filled

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        query_vector = _validated_vector(
            self.backend.embed_one(query.text),
            expected_dimension=self.backend.dimension,
            context="查询 embedding",
        )
        query_dimension = len(query_vector)
        self.ensure_vectors(expected_dimension=query_dimension)
        sql = [_ATOM_SELECT]
        params: list[Any] = []
        _apply_common_filters(sql, params, query)
        rows = self.database.fetchall("\n".join(sql), params)
        scored: list[tuple[float, Any]] = []
        for row in rows:
            vector_row = self.database.fetchone(
                """
                SELECT embedding_dimension, embedding_text_hash, vector_json
                FROM atom_vectors
                WHERE atom_id=? AND embedding_provider=? AND embedding_model=?
                  AND embedding_text_version=? AND embedding_dimension=?
                """,
                (
                    int(row["atom_id"]), self.backend.provider, self.backend.model,
                    self.text_version, query_dimension,
                ),
            )
            current_hash = embedding_text_hash(embedding_text(row))
            cached = self._cached_vector(vector_row, current_hash)
            if cached is None:
                continue
            vector = cached[0]
            scored.append((VectorRetriever._cosine(query_vector, vector), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, row in scored[: query.limit * 4]:
            if score <= 0.02:
                continue
            hits.append(_row_to_hit(row, score, self.route_name))
        return hits[: query.limit]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=False))

class LocalHeuristicReranker:
    """离线、确定性的第二阶段相关性重排，不冒充 cross-encoder。"""

    name = "local_heuristic"

    def __init__(self, candidate_limit: int = 30):
        if candidate_limit < 1:
            raise ValueError("rerank candidate_limit 必须大于 0")
        self.candidate_limit = candidate_limit

    @staticmethod
    def _normalized(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def rerank(
        self, query: RetrievalQuery, candidates: list[RetrievalHit]
    ) -> list[RetrievalHit]:
        if not candidates:
            return []
        query_tokens = tokenize(query.text)
        if not query_tokens:
            compact = self._normalized(query.text)
            query_tokens = list(dict.fromkeys(compact[i : i + 2] for i in range(len(compact) - 1)))
        query_compact = self._normalized(query.text)
        max_initial = max(candidate.score for candidate in candidates) or 1.0
        reranked: list[RetrievalHit] = []
        for candidate in candidates:
            fields = candidate.fields
            title_area = " ".join(
                [
                    str(fields.get("title") or ""),
                    str(fields.get("heading") or ""),
                    " ".join(str(tag) for tag in fields.get("tags") or []),
                ]
            )
            document = f"{title_area}\n{fields.get('excerpt') or ''}"
            document_compact = self._normalized(document)
            title_compact = self._normalized(title_area)
            coverage = (
                sum(1 for token in query_tokens if self._normalized(token) in document_compact)
                / len(query_tokens)
                if query_tokens
                else 0.0
            )
            title_coverage = (
                sum(1 for token in query_tokens if self._normalized(token) in title_compact)
                / len(query_tokens)
                if query_tokens
                else 0.0
            )
            phrase_match = 1.0 if query_compact and query_compact in document_compact else 0.0
            route_count = len(
                {route for route in candidate.route_scores if route != "rerank"}
            )
            route_agreement = min(route_count / 2.0, 1.0)
            initial = candidate.score / max_initial
            score = (
                0.55 * initial
                + 0.25 * coverage
                + 0.10 * title_coverage
                + 0.05 * phrase_match
                + 0.05 * route_agreement
            )
            route_scores = dict(candidate.route_scores)
            route_scores["rerank"] = round(score, 6)
            hit_fields = dict(fields)
            hit_fields["pre_rerank_score"] = round(candidate.score, 6)
            reranked.append(
                RetrievalHit(
                    atom_id=candidate.atom_id,
                    source_id=candidate.source_id,
                    score=round(score, 6),
                    route_scores=route_scores,
                    fields=hit_fields,
                )
            )
        return sorted(reranked, key=lambda hit: (-hit.score, hit.atom_id))


class APICrossEncoderReranker:
    """Second-stage neural reranker using a SiliconFlow-compatible /rerank API."""

    name = "api"

    def __init__(
        self,
        database: Database,
        client: Any,
        *,
        provider: str,
        model: str,
        candidate_limit: int = 30,
        fusion_strategy: str = "rank_fusion",
        score_cache: dict[str, list[tuple[int, float]]] | None = None,
    ):
        if candidate_limit < 1:
            raise ValueError("rerank candidate_limit 必须大于 0")
        if fusion_strategy not in RERANKER_FUSIONS:
            raise ValueError("reranker fusion_strategy 必须是 replace 或 rank_fusion")
        self.database = database
        self.client = client
        self.provider = provider
        self.model = model
        self.candidate_limit = candidate_limit
        self.fusion_strategy = fusion_strategy
        self.score_cache = score_cache if score_cache is not None else {}
        self.is_cloud = True
        self.private_payload_sent_count = 0

    def _document(self, atom_id: int) -> str:
        row = self.database.fetchone(
            """
            SELECT a.heading, a.text, s.title, s.tags_json
            FROM source_atoms a JOIN sources s ON s.id=a.source_id
            WHERE a.id=? AND s.is_present=1 AND s.searchable=1
            """,
            (atom_id,),
        )
        if row is None:
            raise ValueError(f"rerank 候选 atom_id={atom_id} 不存在")
        return embedding_text(row)

    def rerank(
        self, query: RetrievalQuery, candidates: list[RetrievalHit]
    ) -> list[RetrievalHit]:
        if not candidates:
            return []
        documents = [self._document(candidate.atom_id) for candidate in candidates]
        cache_payload = json.dumps(
            [self.model, query.text, documents], ensure_ascii=False, separators=(",", ":")
        )
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        ranked = self.score_cache.get(cache_key)
        if ranked is None:
            # documents 来自 Vault 原子；请求失败也应记录为已经尝试发送。
            self.private_payload_sent_count += len(documents)
            ranked = self.client.rerank(query.text, documents, model=self.model)
            self.score_cache[cache_key] = ranked
        output: list[RetrievalHit] = []
        prior_ranks = {
            candidate.atom_id: rank
            for rank, candidate in enumerate(candidates, start=1)
        }
        for cross_rank, (index, score) in enumerate(ranked, start=1):
            candidate = candidates[index]
            route_scores = dict(candidate.route_scores)
            route_scores["rerank"] = round(score, 6)
            hit_fields = dict(candidate.fields)
            hit_fields["pre_rerank_score"] = round(candidate.score, 6)
            hit_fields["reranker_provider"] = self.provider
            hit_fields["reranker_model"] = self.model
            hit_fields["reranker_fusion_strategy"] = self.fusion_strategy
            final_score = score
            if self.fusion_strategy == "rank_fusion":
                prior_rank = prior_ranks[candidate.atom_id]
                final_score = (
                    1.0 / (DEFAULT_RRF_K + prior_rank)
                    + 1.0 / (DEFAULT_RRF_K + cross_rank)
                )
                route_scores["rerank_rank_fusion"] = round(final_score, 6)
            output.append(
                RetrievalHit(
                    atom_id=candidate.atom_id,
                    source_id=candidate.source_id,
                    score=round(final_score, 6),
                    route_scores=route_scores,
                    fields=hit_fields,
                )
            )
        return sorted(output, key=lambda hit: (-hit.score, hit.atom_id))


class HybridRetriever:
    """按 routes 运行单路或双路检索；双路时用 RRF 融合。"""

    RRF_K = DEFAULT_RRF_K

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        routes: Sequence[str] = ("bm25", "vector"),
        mode: str = "hybrid",
        reranker: Reranker | None = None,
        candidate_limit: int = 30,
    ):
        if candidate_limit < 1:
            raise ValueError("hybrid candidate_limit 必须大于 0")
        self.bm25 = bm25
        self.vector = vector
        self.routes = tuple(routes)
        self.mode = mode
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    @property
    def private_payload_sent_count(self) -> int:
        """本进程中由检索组件发送到云端的 Vault 文本条数。"""
        vector_count = int(getattr(self.vector, "private_payload_sent_count", 0))
        reranker_count = int(getattr(self.reranker, "private_payload_sent_count", 0))
        return vector_count + reranker_count

    def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
        route_query = query
        if len(self.routes) > 1:
            route_query = replace(
                query, limit=max(query.limit, self.candidate_limit)
            )
        per_route: dict[str, list[RetrievalHit]] = {}
        if "bm25" in self.routes:
            per_route["bm25"] = self.bm25.search(route_query)
        if "vector" in self.routes:
            per_route[self.vector.route_name] = self.vector.search(route_query)
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
        ranked = sorted(fused.values(), key=lambda h: (-h.score, h.atom_id))
        if self.reranker is not None and len(self.routes) > 1:
            ranked = self.reranker.rerank(query, ranked[: self.candidate_limit])
        return ranked[: query.limit]


def build_embedding_backend(settings: Settings, backend_name: str | None = None) -> EmbeddingBackend:
    """唯一 Embedding provider 工厂；这里只构造客户端，不发起网络请求。"""
    name = (backend_name or settings.embedding_backend).strip().lower()
    if name not in EMBEDDING_BACKENDS:
        raise ValueError(
            f"未知 MG_EMBEDDING_BACKEND={name!r}；可选 local_hash/mock/api"
        )
    if name == "local_hash":
        return LocalHashEmbeddingBackend(settings.local_hash_dimension)
    if name == "mock":
        return MockEmbeddingBackend(settings.mock_embedding_dimension)
    if not settings.allow_cloud_embedding:
        raise ValueError(
            "API Embedding 已被隐私边界阻止；只有显式设置 "
            "MG_ALLOW_CLOUD_EMBEDDING=true 后才允许发送标题、标题层级、标签、正文和查询文本"
        )
    base_url = settings.embedding_base_url
    api_key = settings.embedding_api_key
    missing = []
    if not base_url:
        missing.append("MG_EMBEDDING_BASE_URL")
    if not api_key:
        missing.append("MG_EMBEDDING_API_KEY")
    if not settings.llm_embedding_model:
        missing.append("MG_LLM_EMBEDDING_MODEL")
    if missing:
        raise ValueError(f"API Embedding 缺少配置：{', '.join(missing)}")
    from .llm import EmbeddingClientAdapter, OpenAICompatibleClient

    client = OpenAICompatibleClient(
        settings,
        base_url=base_url,
        api_key=api_key,
        embedding_model=settings.llm_embedding_model,
        embedding_dimension=settings.llm_embedding_dimension,
    )
    return EmbeddingClientAdapter(client, provider=settings.embedding_provider)


def build_reranker(
    database: Database,
    settings: Settings,
    backend_name: str | None = None,
    fusion_strategy: str | None = None,
    score_cache: dict[str, list[tuple[int, float]]] | None = None,
) -> Reranker | None:
    """唯一 reranker 工厂；构造 API 客户端时不发送任何文本。"""
    name = (backend_name or settings.reranker_backend).strip().lower()
    if name not in RERANKER_BACKENDS:
        raise ValueError(
            f"未知 MG_RERANKER_BACKEND={name!r}；可选 none/local_heuristic/api"
        )
    if name == "none":
        return None
    if name == "local_heuristic":
        return LocalHeuristicReranker(settings.rerank_candidate_limit)
    selected_fusion = (fusion_strategy or settings.reranker_fusion).strip().lower()
    if selected_fusion not in RERANKER_FUSIONS:
        raise ValueError(
            f"未知 MG_RERANKER_FUSION={selected_fusion!r}；可选 replace/rank_fusion"
        )
    if not settings.allow_cloud_rerank:
        raise ValueError(
            "API Rerank 已被隐私边界阻止；只有显式设置 "
            "MG_ALLOW_CLOUD_RERANK=true 后才允许发送查询和 RRF 候选文本"
        )
    base_url = settings.reranker_base_url
    api_key = settings.reranker_api_key
    missing = []
    if not base_url:
        missing.append("MG_RERANKER_BASE_URL")
    if not api_key:
        missing.append("MG_RERANKER_API_KEY")
    if not settings.reranker_model:
        missing.append("MG_RERANKER_MODEL")
    if missing:
        raise ValueError(f"API Rerank 缺少配置：{', '.join(missing)}")
    from .llm import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        settings,
        base_url=base_url,
        api_key=api_key,
    )
    return APICrossEncoderReranker(
        database,
        client,
        provider=settings.reranker_provider,
        model=settings.reranker_model,
        candidate_limit=settings.rerank_candidate_limit,
        fusion_strategy=selected_fusion,
        score_cache=score_cache,
    )


def build_retriever(
    database: Database,
    settings: Settings,
    mode: str | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    reranker_backend: str | None = None,
    reranker_fusion: str | None = None,
    reranker_score_cache: dict[str, list[tuple[int, float]]] | None = None,
) -> HybridRetriever:
    """CLI、Web、MCP 共用的唯一 retriever/provider 工厂。"""
    selected_mode = (mode or settings.retrieval_mode).strip().lower()
    if selected_mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"未知检索模式 {selected_mode!r}；可选 bm25/hash_vector/embedding/hybrid"
        )
    backend: EmbeddingBackend
    if selected_mode in {"bm25", "hash_vector"}:
        backend = LocalHashEmbeddingBackend(settings.local_hash_dimension)
    else:
        backend = embedding_backend or build_embedding_backend(settings)
        if selected_mode == "embedding" and backend.provider == "local_hash":
            raise ValueError(
                "retrieval_mode=embedding 需要 MG_EMBEDDING_BACKEND=mock 或 api；"
                "字符哈希向量请使用 hash_vector"
            )
    vector = VectorRetriever(database, backend)
    routes = {
        "bm25": ("bm25",),
        "hash_vector": ("vector",),
        "embedding": ("vector",),
        "hybrid": ("bm25", "vector"),
    }[selected_mode]
    reranker = (
        build_reranker(
            database,
            settings,
            reranker_backend,
            reranker_fusion,
            reranker_score_cache,
        )
        if selected_mode == "hybrid"
        else None
    )
    return HybridRetriever(
        BM25Retriever(database),
        vector,
        routes=routes,
        mode=selected_mode,
        reranker=reranker,
        candidate_limit=settings.rerank_candidate_limit,
    )


def build_public_evaluation_retrievers(
    database: Database, settings: Settings
) -> dict[str, HybridRetriever]:
    """公开评估固定使用 local_hash + mock，并保留重排前后对照，绝不调用 API。"""
    bm25 = build_retriever(database, settings, mode="bm25")
    hash_vector = build_retriever(database, settings, mode="hash_vector")
    mock_backend = MockEmbeddingBackend(settings.mock_embedding_dimension)
    embedding = build_retriever(
        database, settings, mode="embedding", embedding_backend=mock_backend
    )
    hybrid = build_retriever(
        database, settings, mode="hybrid", embedding_backend=mock_backend,
        reranker_backend="none",
    )
    hybrid_rerank = build_retriever(
        database, settings, mode="hybrid", embedding_backend=mock_backend,
        reranker_backend="local_heuristic",
    )
    hash_vector.vector.ensure_vectors()
    embedding.vector.ensure_vectors()
    return {
        "bm25": bm25,
        "hash_vector": hash_vector,
        "embedding": embedding,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
    }


def build_private_local_evaluation_retrievers(
    database: Database, settings: Settings
) -> dict[str, HybridRetriever]:
    """真实私有库评估只使用本地哈希向量，不调用 mock 或云端 API。"""
    return {
        "bm25": build_retriever(database, settings, mode="bm25"),
        "hash_vector": build_retriever(database, settings, mode="hash_vector"),
        "hybrid": build_retriever(
            database, settings, mode="hybrid",
            embedding_backend=LocalHashEmbeddingBackend(settings.local_hash_dimension),
            reranker_backend="none",
        ),
        "hybrid_rerank": build_retriever(
            database, settings, mode="hybrid",
            embedding_backend=LocalHashEmbeddingBackend(settings.local_hash_dimension),
            reranker_backend="local_heuristic",
        ),
    }


def build_private_api_evaluation_retrievers(
    database: Database, settings: Settings
) -> dict[str, HybridRetriever]:
    """Opt-in real-model routes for a private golden set; caller must use a derived DB."""
    api_backend = build_embedding_backend(settings, "api")
    rerank_score_cache: dict[str, list[tuple[int, float]]] = {}
    embedding = build_retriever(
        database, settings, mode="embedding", embedding_backend=api_backend
    )
    hybrid = build_retriever(
        database,
        settings,
        mode="hybrid",
        embedding_backend=api_backend,
        reranker_backend="none",
    )
    hybrid_rerank = build_retriever(
        database,
        settings,
        mode="hybrid",
        embedding_backend=api_backend,
        reranker_backend="api",
        reranker_fusion="replace",
        reranker_score_cache=rerank_score_cache,
    )
    hybrid_rerank_fused = build_retriever(
        database,
        settings,
        mode="hybrid",
        embedding_backend=api_backend,
        reranker_backend="api",
        reranker_fusion="rank_fusion",
        reranker_score_cache=rerank_score_cache,
    )
    return {
        "bm25": build_retriever(database, settings, mode="bm25"),
        "hash_vector": build_retriever(database, settings, mode="hash_vector"),
        "embedding": embedding,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
        "hybrid_rerank_fused": hybrid_rerank_fused,
    }
