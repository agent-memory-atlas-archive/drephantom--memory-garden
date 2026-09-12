"""Run one opt-in private RAG smoke against real embedding/rerank/generation APIs.

The Vault is read-only. A temporary derived database is deleted after the run,
and stdout contains aggregate metadata only (no query, title, path, excerpt, or answer).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory_garden.agent import AgentHarness, OpenAIProvider
from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.importer import VaultSyncService, vault_markdown_hash
from memory_garden.llm import OpenAICompatibleClient
from memory_garden.retrieval import BM25Retriever, RetrievalQuery, build_retriever


def run(case_index: int, source_limit: int, group: str) -> dict[str, Any]:
    if case_index < 0:
        raise ValueError("case_index must be non-negative")
    if source_limit < 1:
        raise ValueError("source_limit must be positive")
    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "evals" / "retrieval_goldens.real.json").read_text(encoding="utf-8")
    )
    groups = document.get("groups", {})
    if group not in groups:
        raise ValueError(f"private golden group {group!r} is unavailable")
    cases = groups[group]
    if case_index >= len(cases):
        raise ValueError(f"case_index={case_index} is outside {group} (size={len(cases)})")
    case = cases[case_index]
    query = str(case["query"])
    relevant_paths = [str(path) for path in case["relevant_paths"]]

    loaded = Settings.load(root)
    settings = replace(
        loaded,
        backend="deepseek",
        embedding_backend="api",
        retrieval_mode="hybrid",
        reranker_backend="api",
    )
    if not settings.llm_ready:
        raise ValueError("LLM generation connection is not ready")
    if not settings.embedding_api_ready:
        raise ValueError("Embedding API connection is not ready")
    if not settings.reranker_api_ready:
        raise ValueError("Rerank API connection is not ready")

    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(settings, database_path=Path(tmp) / "rag-smoke.db")
        database = Database(settings.database_path)
        database.initialize()
        before_hash = vault_markdown_hash(settings.vault_path)
        try:
            VaultSyncService(database, settings.vault_path).sync()
            lexical = BM25Retriever(database).search(
                RetrievalQuery(text=query, limit=source_limit)
            )
            keep_source_ids = {hit.source_id for hit in lexical}
            placeholders = ",".join("?" for _ in relevant_paths)
            relevant_rows = database.fetchall(
                f"SELECT id FROM sources WHERE rel_path IN ({placeholders})",
                tuple(relevant_paths),
            )
            keep_source_ids.update(int(row["id"]) for row in relevant_rows)
            extras = database.fetchall(
                "SELECT id FROM sources WHERE is_present=1 AND searchable=1 "
                "ORDER BY id LIMIT ?",
                (source_limit,),
            )
            keep_source_ids.update(int(row["id"]) for row in extras)
            source_ids = sorted(keep_source_ids)
            database.execute("UPDATE sources SET searchable=0")
            selected_placeholders = ",".join("?" for _ in source_ids)
            database.execute(
                f"UPDATE sources SET searchable=1 WHERE id IN ({selected_placeholders})",
                tuple(source_ids),
            )
            count_row = database.fetchone(
                "SELECT COUNT(*) AS n FROM source_atoms a "
                "JOIN sources s ON s.id=a.source_id "
                "WHERE s.searchable=1 AND s.is_present=1 AND a.authorship!='derived'"
            )
            assert count_row is not None
            atom_count = int(count_row["n"])
            retriever = build_retriever(database, settings)
            vectors_built = retriever.vector.ensure_vectors()
            provider = OpenAIProvider(OpenAICompatibleClient(settings))
            result = AgentHarness(
                database, retriever, settings, provider=provider
            ).run(query)
            run_row = database.fetchone(
                "SELECT private_vault_sent FROM agent_runs WHERE message_id=?",
                (result.message_id,),
            )
            assert run_row is not None
            after_hash = vault_markdown_hash(settings.vault_path)
            return {
                "selected_sources": len(source_ids),
                "selected_atoms": atom_count,
                "vectors_built": vectors_built,
                "backend": result.backend,
                "steps": result.steps,
                "tool_calls": result.tool_calls,
                "stop_reason": result.stop_reason,
                "answer_type": result.answer.answer_type,
                "citations": len(result.answer.citations),
                "latency_ms": result.latency_ms,
                "private_vault_sent": bool(run_row["private_vault_sent"]),
                "vault_hash_unchanged": before_hash == after_hash,
                "error_present": bool(result.error),
            }
        finally:
            database.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--source-limit", type=int, default=60)
    parser.add_argument("--group", choices=["real_dev", "real_test"], default="real_dev")
    args = parser.parse_args()
    print(
        json.dumps(
            run(case_index=args.case_index, source_limit=args.source_limit, group=args.group),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
