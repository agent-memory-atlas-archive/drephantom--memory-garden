"""检索层测试：短词 LIKE 兜底、向量确定性、RRF 融合、标题/标签可检索。"""
from __future__ import annotations

from memory_garden.retrieval import (
    BM25Retriever,
    HashedNgramVectorizer,
    HybridRetriever,
    RetrievalQuery,
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
