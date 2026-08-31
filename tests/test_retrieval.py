"""检索层测试：短词 LIKE 兜底、向量确定性、RRF 融合、标题/标签可检索。"""
from __future__ import annotations

from memory_garden.retrieval import (
    BM25Retriever,
    HashedNgramVectorizer,
    HybridRetriever,
    RetrievalHit,
    RetrievalQuery,
    VectorRetriever,
    build_retriever,
)


def test_bm25_finds_by_title_and_tags(database) -> None:
    retriever = BM25Retriever(database)
    hits = retriever.search(RetrievalQuery(text="自主判断", limit=8))
    paths = {hit.fields["path"] for hit in hits}
    assert "自主判断-2019.md" in paths
    assert "自主判断-2024.md" in paths


def test_bm25_two_char_term_like_fallback(database) -> None:
    retriever = BM25Retriever(database)
    hits = retriever.search(RetrievalQuery(text="独处", limit=8))
    assert any(hit.fields["path"] == "独处-2020.md" for hit in hits)


def test_bm25_splits_natural_language_question(database) -> None:
    retriever = BM25Retriever(database)
    hits = retriever.search(
        RetrievalQuery(text="关于自主判断这件事，我想看看以前和现在的变化", limit=8)
    )
    paths = {hit.fields["path"] for hit in hits}
    assert "自主判断-2019.md" in paths
    assert "自主判断-2024.md" in paths


def test_vector_route_covers_synonym_gap(database, retriever) -> None:
    """同义缺口：'一个人恢复精力' 通过向量/短词路命中两代记录。"""
    hybrid_hits = retriever.search(RetrievalQuery(text="一个人恢复精力", limit=8))
    paths = {hit.fields["path"] for hit in hybrid_hits}
    assert "独处-2020.md" in paths
    assert "一个人-2024.md" in paths


def test_vectorizer_deterministic() -> None:
    vectorizer = HashedNgramVectorizer(dim=256)
    a = vectorizer.transform("认知回溯与记忆的花园")
    b = vectorizer.transform("认知回溯与记忆的花园")
    c = vectorizer.transform("完全不同的另外一段文本内容")
    assert a == b
    assert HashedNgramVectorizer.cosine(a, c) < HashedNgramVectorizer.cosine(a, b)


def test_hybrid_merges_routes_with_rrf(database, retriever) -> None:
    hybrid = HybridRetriever(BM25Retriever(database), retriever.vector)
    hits = hybrid.search(RetrievalQuery(text="完美主义 准备", limit=5))
    assert hits, "混合检索应返回结果"
    top = hits[0]
    assert top.score > 0
    assert top.route_scores, "应保留每路的归因分数"


def test_authorship_filter_excludes_quoted_and_ai(database) -> None:
    retriever = BM25Retriever(database)
    hits = retriever.search(RetrievalQuery(text="独立", authorship="user", limit=10))
    authorships = {hit.fields["authorship"] for hit in hits}
    assert authorships <= {"user"}


def test_vector_filters_before_top_k_truncation(database) -> None:
    class FilterBackend:
        provider = "test"
        model = "filter-order-v1"
        dimension = 2
        is_cloud = False

        @staticmethod
        def _vector(text: str) -> list[float]:
            if "EXCLUDED" in text:
                return [1.0, 0.0]
            if "INCLUDED" in text:
                return [0.8, 0.6]
            return [0.0, 1.0]

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [self._vector(text) for text in texts]

        def embed_one(self, _text: str) -> list[float]:
            return [1.0, 0.0]

    rows = database.fetchall("SELECT id FROM source_atoms ORDER BY id LIMIT 6")
    excluded_ids = [int(row["id"]) for row in rows[:5]]
    included_id = int(rows[5]["id"])
    for atom_id in excluded_ids:
        database.execute(
            "UPDATE source_atoms SET text='EXCLUDED', authorship='quoted' WHERE id=?",
            (atom_id,),
        )
    database.execute(
        "UPDATE source_atoms SET text='INCLUDED', authorship='user' WHERE id=?",
        (included_id,),
    )

    retriever = VectorRetriever(database, FilterBackend())
    hits = retriever.search(
        RetrievalQuery(text="query", authorship="user", limit=1)
    )
    assert [hit.atom_id for hit in hits] == [included_id]


def test_local_reranker_is_deterministic_and_auditable(settings, database) -> None:
    first = build_retriever(database, settings, mode="hybrid")
    first.vector.ensure_vectors()
    query = RetrievalQuery(text="先完成可用版本再根据反馈修正", limit=5)
    hits_a = first.search(query)
    hits_b = first.search(query)
    assert [hit.atom_id for hit in hits_a] == [hit.atom_id for hit in hits_b]
    assert hits_a
    assert all("rerank" in hit.route_scores for hit in hits_a)
    assert all("pre_rerank_score" in hit.fields for hit in hits_a)


def test_hybrid_reranker_can_be_disabled(settings, database) -> None:
    retriever = build_retriever(
        database, settings, mode="hybrid", reranker_backend="none"
    )
    retriever.vector.ensure_vectors()
    hits = retriever.search(RetrievalQuery(text="自主判断", limit=5))
    assert hits
    assert all("rerank" not in hit.route_scores for hit in hits)


def test_hybrid_and_rerank_use_the_same_route_candidate_depth() -> None:
    class RecordingRoute:
        def __init__(self, route_name: str):
            self.route_name = route_name
            self.limits: list[int] = []

        def search(self, query: RetrievalQuery) -> list[RetrievalHit]:
            self.limits.append(query.limit)
            return [
                RetrievalHit(
                    atom_id=index,
                    source_id=index,
                    score=1.0 / index,
                    fields={"path": f"{index}.md"},
                )
                for index in range(1, query.limit + 1)
            ]

    class RecordingReranker:
        name = "recording"
        candidate_limit = 30

        def __init__(self):
            self.seen = 0

        def rerank(self, _query, candidates):
            self.seen = len(candidates)
            return candidates

    baseline_bm25 = RecordingRoute("bm25")
    baseline_vector = RecordingRoute("hash_vector")
    rerank_bm25 = RecordingRoute("bm25")
    rerank_vector = RecordingRoute("hash_vector")
    reranker = RecordingReranker()
    baseline = HybridRetriever(
        baseline_bm25, baseline_vector, candidate_limit=30
    )
    reranked = HybridRetriever(
        rerank_bm25,
        rerank_vector,
        reranker=reranker,
        candidate_limit=30,
    )
    query = RetrievalQuery(text="同一查询", limit=5)
    baseline.search(query)
    reranked.search(query)
    assert baseline_bm25.limits == rerank_bm25.limits == [30]
    assert baseline_vector.limits == rerank_vector.limits == [30]
    assert reranker.seen == 30
