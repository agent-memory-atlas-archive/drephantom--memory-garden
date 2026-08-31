"""Embedding provider、缓存失效、隐私边界与三入口工厂测试。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.importer import VaultSyncService
from memory_garden.llm import LLMError, OpenAICompatibleClient
from memory_garden.retrieval import (
    APICrossEncoderReranker,
    MockEmbeddingBackend,
    RetrievalHit,
    RetrievalQuery,
    VectorRetriever,
    build_embedding_backend,
    build_private_api_evaluation_retrievers,
    build_private_local_evaluation_retrievers,
    build_public_evaluation_retrievers,
    build_reranker,
    build_retriever,
    embedding_text,
)
from memory_garden.web import create_app


class CountingBackend:
    provider = "mock"
    is_cloud = False

    def __init__(self, model: str, dimension: int):
        self.model = model
        self.dimension: int | None = dimension
        self.texts_embedded: list[str] = []

    def _vector(self, text: str) -> list[float]:
        assert self.dimension is not None
        vector = [0.0] * self.dimension
        index = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
        vector[index % self.dimension] = 1.0
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vector(text)


def test_mock_embedding_is_reproducible() -> None:
    first = MockEmbeddingBackend(64).embed_one("别把决定权交给别人")
    second = MockEmbeddingBackend(64).embed_one("别把决定权交给别人")
    assert first == second
    assert len(first) == 64


def test_canonical_embedding_text_includes_heading() -> None:
    text = embedding_text(
        {
            "title": "笔记标题",
            "heading": "二级标题",
            "tags_json": '["标签甲", "标签乙"]',
            "text": "正文内容",
        }
    )
    assert text == "笔记标题\n二级标题\n标签甲 标签乙\n正文内容"


def test_default_mode_never_constructs_cloud_client(settings, database, monkeypatch) -> None:
    def fail_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("默认模式不应创建网络客户端")

    monkeypatch.setattr("memory_garden.llm.httpx.Client", fail_client)
    retriever = build_retriever(database, settings)
    retriever.vector.ensure_vectors()
    assert retriever.vector.backend.provider == "local_hash"
    assert retriever.search(RetrievalQuery(text="自主判断", limit=3))


def test_web_api_backend_does_not_upload_during_startup(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    def fail_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Web 启动不应发起 Embedding API 请求")

    monkeypatch.setattr("memory_garden.llm.httpx.Client", fail_client)
    cloud = Settings(
        vault_path=settings.vault_path,
        database_path=tmp_path / "web-cloud.db",
        embedding_backend="api",
        retrieval_mode="hybrid",
        allow_cloud_embedding=True,
        llm_base_url="https://example.invalid/v1",
        llm_api_key="startup-secret",
        llm_embedding_model="embed-model",
        reranker_backend="api",
        reranker_base_url="https://example.invalid/v1",
        reranker_api_key="startup-rerank-secret",
        reranker_model="rerank-model",
        allow_cloud_rerank=True,
    )
    client = TestClient(create_app(cloud))
    page = client.get("/settings")
    assert page.status_code == 200
    assert "标题、标题层级、标签、正文" in page.text
    assert "Web 启动本身不会批量发送" in page.text
    assert "cross-encoder" in page.text


def test_api_backend_requires_explicit_cloud_consent(settings: Settings) -> None:
    guarded = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        embedding_backend="api",
        llm_base_url="https://example.invalid/v1",
        llm_api_key="secret-not-printed",
        llm_embedding_model="embed-model",
        allow_cloud_embedding=False,
    )
    with pytest.raises(ValueError, match="显式设置 MG_ALLOW_CLOUD_EMBEDDING"):
        build_embedding_backend(guarded)
    assert "secret-not-printed" not in repr(guarded)


def test_api_provider_request_format_and_dimension_validation(
    settings: Settings, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            }

    class Client:
        def __init__(self, **kwargs: Any):
            calls.append({"client": kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "json": json})
            return Response()

    monkeypatch.setattr("memory_garden.llm.httpx.Client", Client)
    api_settings = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        llm_base_url="https://embedding.example/v1",
        llm_api_key="test-secret-key",
        llm_embedding_model="embed-v1",
        llm_embedding_dimension=3,
        allow_cloud_embedding=True,
        embedding_backend="api",
    )
    vectors = OpenAICompatibleClient(api_settings).embed(["甲", "乙"])
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    request = calls[-1]
    assert request["url"] == "https://embedding.example/v1/embeddings"
    assert request["json"] == {"model": "embed-v1", "input": ["甲", "乙"]}

    api_settings.llm_embedding_dimension = 4
    with pytest.raises(LLMError, match="返回维度 3"):
        OpenAICompatibleClient(api_settings).embed(["甲", "乙"])


def test_embedding_retries_transient_network_error(settings: Settings, monkeypatch) -> None:
    attempts = 0

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    class Client:
        def __init__(self, **_kwargs: Any):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def close(self) -> None:
            return None

        def post(self, *_args: Any, **_kwargs: Any):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("transient TLS close")
            return Response()

    monkeypatch.setattr("memory_garden.llm.httpx.Client", Client)
    monkeypatch.setattr("memory_garden.llm.time.sleep", lambda _seconds: None)
    retrying = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        llm_base_url="https://embedding.example/v1",
        llm_api_key="fake-key",
        llm_embedding_model="embed-v1",
        llm_embedding_dimension=2,
        llm_max_retries=1,
    )
    assert OpenAICompatibleClient(retrying).embed_one("合成文本") == [1.0, 0.0]
    assert attempts == 2


def test_dedicated_embedding_connection_does_not_reuse_chat_provider(
    settings: Settings, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}

    class Client:
        def __init__(self, **kwargs: Any):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "json": json})
            return Response()

    monkeypatch.setattr("memory_garden.llm.httpx.Client", Client)
    dedicated = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        llm_base_url="https://chat.example/v1",
        llm_api_key="chat-fake-key",
        embedding_backend="api",
        embedding_provider="siliconflow",
        embedding_base_url="https://embedding.example/v1",
        embedding_api_key="embedding-fake-key",
        llm_embedding_model="BAAI/bge-m3",
        llm_embedding_dimension=3,
        allow_cloud_embedding=True,
    )
    backend = build_embedding_backend(dedicated)
    assert backend.embed_one("合成文本") == [1.0, 0.0, 0.0]
    assert backend.provider == "siliconflow"
    assert calls[0]["url"] == "https://embedding.example/v1/embeddings"
    assert calls[0]["headers"]["Authorization"] == "Bearer embedding-fake-key"


def test_api_cross_encoder_request_format_and_result_mapping(
    settings: Settings, database, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            count = len(calls[-1]["json"]["documents"])
            return {
                "results": [
                    {"index": index, "relevance_score": float(count - index)}
                    for index in range(count)
                ]
            }

    class Client:
        def __init__(self, **kwargs: Any):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "json": json})
            return Response()

    monkeypatch.setattr("memory_garden.llm.httpx.Client", Client)
    api_settings = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        reranker_backend="api",
        reranker_provider="siliconflow",
        reranker_base_url="https://rerank.example/v1",
        reranker_api_key="rerank-fake-key",
        reranker_model="BAAI/bge-reranker-v2-m3",
        rerank_candidate_limit=3,
        allow_cloud_rerank=True,
    )
    reranker = build_reranker(database, api_settings)
    assert reranker is not None
    candidates = build_retriever(database, settings, mode="bm25").search(
        RetrievalQuery(text="自主判断", limit=3)
    )
    ranked = reranker.rerank(RetrievalQuery(text="换一种说法", limit=3), candidates)
    request = calls[0]
    assert request["url"] == "https://rerank.example/v1/rerank"
    assert request["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert request["json"]["query"] == "换一种说法"
    assert request["json"]["top_n"] == len(candidates)
    assert request["json"]["return_documents"] is False
    assert "自主判断" in request["json"]["documents"][0]
    assert ranked[0].fields["reranker_provider"] == "siliconflow"
    assert ranked[0].fields["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert reranker.private_payload_sent_count == len(candidates)


def test_api_reranker_requires_separate_cloud_consent(settings: Settings, database) -> None:
    guarded = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        reranker_backend="api",
        reranker_base_url="https://example.invalid/v1",
        reranker_api_key="fake-key",
        reranker_model="rerank-model",
        allow_cloud_rerank=False,
    )
    with pytest.raises(ValueError, match="MG_ALLOW_CLOUD_RERANK"):
        build_reranker(database, guarded)
    assert "fake-key" not in repr(guarded)


def test_api_reranker_can_compare_replace_and_rank_fusion(database) -> None:
    class Client:
        def rerank(self, _query: str, _documents: list[str], *, model: str):
            assert model == "rerank-model"
            return [(2, 3.0), (0, 2.0), (1, 1.0)]

    rows = database.fetchall("SELECT id, source_id FROM source_atoms ORDER BY id LIMIT 3")
    candidates = [
        RetrievalHit(
            atom_id=int(row["id"]),
            source_id=int(row["source_id"]),
            score=1.0 / rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]
    replace = APICrossEncoderReranker(
        database,
        Client(),
        provider="test",
        model="rerank-model",
        candidate_limit=3,
        fusion_strategy="replace",
    )
    fused = APICrossEncoderReranker(
        database,
        Client(),
        provider="test",
        model="rerank-model",
        candidate_limit=3,
        fusion_strategy="rank_fusion",
    )
    query = RetrievalQuery(text="测试查询", limit=3)
    replace_ids = [hit.atom_id for hit in replace.rerank(query, candidates)]
    fused_hits = fused.rerank(query, candidates)
    assert replace_ids == [candidates[2].atom_id, candidates[0].atom_id, candidates[1].atom_id]
    assert [hit.atom_id for hit in fused_hits] == [
        candidates[0].atom_id,
        candidates[2].atom_id,
        candidates[1].atom_id,
    ]
    assert all("rerank_rank_fusion" in hit.route_scores for hit in fused_hits)


def test_separate_key_files_are_loaded_without_secret_repr(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    embedding_key = tmp_path / "embedding.key"
    reranker_key = tmp_path / "reranker.key"
    embedding_key.write_text("embedding-file-secret", encoding="utf-8")
    reranker_key.write_text("reranker-file-secret", encoding="utf-8")
    (project / ".env").write_text(
        "MG_EMBEDDING_API_KEY_FILE=" + str(embedding_key) + "\n"
        "MG_RERANKER_API_KEY_FILE=" + str(reranker_key) + "\n",
        encoding="utf-8",
    )
    loaded = Settings.load(project)
    assert loaded.embedding_api_key == "embedding-file-secret"
    assert loaded.reranker_api_key == "reranker-file-secret"
    assert "embedding-file-secret" not in repr(loaded)
    assert "reranker-file-secret" not in repr(loaded)


def test_model_change_builds_separate_cache(database) -> None:
    first = VectorRetriever(database, CountingBackend("model-a", 32))
    second = VectorRetriever(database, CountingBackend("model-b", 32))
    atom_count = int(database.fetchone("SELECT COUNT(*) AS n FROM source_atoms")["n"])
    assert first.ensure_vectors() == atom_count
    assert second.ensure_vectors() == atom_count
    models = {
        str(row["embedding_model"])
        for row in database.fetchall("SELECT DISTINCT embedding_model FROM atom_vectors")
    }
    assert {"model-a", "model-b"} <= models


def test_text_hash_change_rebuilds_only_changed_atom(database) -> None:
    backend = CountingBackend("stable-model", 32)
    retriever = VectorRetriever(database, backend)
    retriever.ensure_vectors()
    atom_id = int(database.fetchone("SELECT id FROM source_atoms ORDER BY id LIMIT 1")["id"])
    before = database.fetchone(
        "SELECT embedding_text_hash FROM atom_vectors WHERE atom_id=? AND embedding_model=?",
        (atom_id, backend.model),
    )["embedding_text_hash"]
    database.execute("UPDATE source_atoms SET text=text || ' 新增内容' WHERE id=?", (atom_id,))
    assert retriever.ensure_vectors() == 1
    after = database.fetchone(
        "SELECT embedding_text_hash FROM atom_vectors WHERE atom_id=? AND embedding_model=?",
        (atom_id, backend.model),
    )["embedding_text_hash"]
    assert before != after


def test_text_version_change_builds_new_identity(database) -> None:
    backend = CountingBackend("versioned-model", 32)
    v1 = VectorRetriever(database, backend, text_version="v1")
    v2 = VectorRetriever(database, backend, text_version="v2")
    atom_count = int(database.fetchone("SELECT COUNT(*) AS n FROM source_atoms")["n"])
    assert v1.ensure_vectors() == atom_count
    assert v2.ensure_vectors() == atom_count
    versions = {
        str(row["embedding_text_version"])
        for row in database.fetchall(
            "SELECT DISTINCT embedding_text_version FROM atom_vectors WHERE embedding_model=?",
            (backend.model,),
        )
    }
    assert versions == {"v1", "v2"}


def test_dimension_change_rebuilds_and_old_dimension_is_not_used(database) -> None:
    model = "dimension-changing-model"
    old = VectorRetriever(database, MockEmbeddingBackend(32, model=model))
    new = VectorRetriever(database, MockEmbeddingBackend(48, model=model))
    atom_count = int(database.fetchone("SELECT COUNT(*) AS n FROM source_atoms")["n"])
    assert old.ensure_vectors() == atom_count
    assert new.ensure_vectors() == atom_count
    dimensions = {
        int(row["embedding_dimension"])
        for row in database.fetchall(
            "SELECT DISTINCT embedding_dimension FROM atom_vectors WHERE embedding_model=?",
            (model,),
        )
    }
    assert dimensions == {48}
    assert new.search(RetrievalQuery(text="自主判断", limit=3))


def test_public_eval_has_rerank_route_and_uses_mock(settings, database) -> None:
    routes = build_public_evaluation_retrievers(database, settings)
    assert set(routes) == {
        "bm25", "hash_vector", "embedding", "hybrid", "hybrid_rerank",
    }
    assert routes["hash_vector"].vector.backend.provider == "local_hash"
    assert routes["embedding"].vector.backend.provider == "mock"
    assert routes["hybrid"].vector.backend.provider == "mock"
    assert routes["hybrid"].reranker is None
    assert routes["hybrid_rerank"].reranker is not None
    private_routes = build_private_local_evaluation_retrievers(database, settings)
    assert set(private_routes) == {"bm25", "hash_vector", "hybrid", "hybrid_rerank"}
    assert all(
        route.vector.backend.provider == "local_hash"
        for route in private_routes.values()
    )


def test_private_api_eval_compares_rerank_strategies_with_shared_scores(
    settings: Settings, database
) -> None:
    api_settings = Settings(
        vault_path=settings.vault_path,
        database_path=settings.database_path,
        embedding_backend="api",
        embedding_provider="test",
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="fake-embedding-key",
        llm_embedding_model="embed-model",
        allow_cloud_embedding=True,
        reranker_backend="api",
        reranker_provider="test",
        reranker_base_url="https://example.invalid/v1",
        reranker_api_key="fake-rerank-key",
        reranker_model="rerank-model",
        allow_cloud_rerank=True,
    )
    routes = build_private_api_evaluation_retrievers(database, api_settings)
    assert set(routes) == {
        "bm25",
        "hash_vector",
        "embedding",
        "hybrid",
        "hybrid_rerank",
        "hybrid_rerank_fused",
    }
    replace_reranker = routes["hybrid_rerank"].reranker
    fused_reranker = routes["hybrid_rerank_fused"].reranker
    assert isinstance(replace_reranker, APICrossEncoderReranker)
    assert isinstance(fused_reranker, APICrossEncoderReranker)
    assert replace_reranker.fusion_strategy == "replace"
    assert fused_reranker.fusion_strategy == "rank_fusion"
    assert replace_reranker.score_cache is fused_reranker.score_cache


def test_cli_web_mcp_share_provider_factory() -> None:
    pytest.importorskip("mcp")
    from memory_garden import cli, mcp_server, retrieval, web

    assert cli.build_retriever is retrieval.build_retriever
    assert web.build_retriever is retrieval.build_retriever
    assert mcp_server.build_retriever is retrieval.build_retriever


def test_v2_vector_table_migrates_losslessly(tmp_path: Path) -> None:
    db_path = tmp_path / "migration.db"
    database = Database(db_path)
    database.initialize()
    vault = Path(__file__).resolve().parents[1] / "evals" / "cognitive_mvp_vault"
    VaultSyncService(database, vault).sync()
    atom_id = int(database.fetchone("SELECT id FROM source_atoms ORDER BY id LIMIT 1")["id"])
    database.close()

    legacy_vector = json.dumps([0.25, -0.5, 0.75])
    connection = sqlite3.connect(db_path)
    connection.execute("DROP TABLE atom_vectors")
    connection.execute(
        "CREATE TABLE atom_vectors(atom_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, "
        "vector_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO atom_vectors(atom_id, dim, vector_json) VALUES(?,?,?)",
        (atom_id, 3, legacy_vector),
    )
    connection.execute("UPDATE schema_meta SET value='2' WHERE key='schema_version'")
    connection.commit()
    connection.close()

    migrated = Database(db_path)
    migrated.initialize()
    row = migrated.fetchone("SELECT * FROM atom_vectors WHERE atom_id=?", (atom_id,))
    assert row is not None
    assert row["embedding_provider"] == "legacy"
    assert row["embedding_model"] == "unknown-v2"
    assert row["embedding_dimension"] == 3
    assert row["vector_json"] == legacy_vector
    assert migrated.fetchone(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    )["value"] == "3"
