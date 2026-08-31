"""命令行入口：init / sync / ask / discover / review / eval-* / serve / mcp / verify-vault。"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .db import Database
from .retrieval import (
    build_private_api_evaluation_retrievers,
    build_private_local_evaluation_retrievers,
    build_public_evaluation_retrievers,
    build_retriever,
)


def build_database(settings: Settings) -> Database:
    database = Database(settings.database_path)
    database.initialize()
    return database


def _print_candidates(candidates) -> None:
    if not candidates:
        print("本次扫描没有发现值得核对的候选。")
        return
    from .snapshots import CHANGE_TYPES

    for item in candidates:
        change_type = getattr(item, "change_type", "") or ""
        label = f" [{CHANGE_TYPES.get(change_type, change_type)}]" if change_type else ""
        print(f"\n#{item.discovery_id} 主题「{item.topic}」{label} score={item.score:.2f}")
        print(f"  早期 {item.early_date}: {item.early_excerpt[:60]}")
        print(f"  近期 {item.recent_date}: {item.recent_excerpt[:60]}")
        print(f"  差异词: {'、'.join(item.diff_terms[:6])}")
        print(f"  {item.question_to_user}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-garden", description="认知回溯 Agent")
    parser.add_argument(
        "--retrieval-mode",
        choices=["bm25", "hash_vector", "embedding", "hybrid"],
        default=None,
        help="覆盖 MG_RETRIEVAL_MODE（需放在子命令前）",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["local_hash", "mock", "api"],
        default=None,
        help="覆盖 MG_EMBEDDING_BACKEND（embedding/hybrid 使用）",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "local_heuristic", "api"],
        default=None,
        help="覆盖 MG_RERANKER_BACKEND（只对 hybrid 生效）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化数据库并同步 Vault")
    sub.add_parser("sync", help="重新同步 Vault（幂等）")
    ask = sub.add_parser("ask", help="提出一个认知回溯问题")
    ask.add_argument("question")
    ask.add_argument("--thread", type=int, default=None)
    discover = sub.add_parser("discover", help="扫描未察觉的变化候选")
    discover.add_argument("--limit", type=int, default=5)
    review = sub.add_parser("review", help="评审一条发现候选")
    review.add_argument("discovery_id", type=int)
    review.add_argument("verdict", choices=["accurate", "partly_accurate", "no_change", "not_my_view", "insufficient_evidence", "defer"])
    review.add_argument("--revision", default="")
    review.add_argument("--missing-event", default="")
    eval_agent = sub.add_parser("eval-agent", help="运行三配置对比评测（离线确定性）")
    eval_agent.add_argument("--output", default="artifacts/evals/agent_comparative")
    eval_retr = sub.add_parser("eval-retrieval", help="运行 BM25/哈希/Embedding/Hybrid 四路指标")
    eval_retr.add_argument(
        "--group",
        choices=["synthetic", "real_dev", "real_test", "real", "all"],
        default="synthetic",
    )
    eval_retr.add_argument(
        "--real-models",
        action="store_true",
        help="真实组显式调用已配置的 API Embedding + API cross-encoder rerank",
    )
    eval_retr.add_argument(
        "--include-draft-goldens",
        action="store_true",
        help="允许运行尚未经 Vault 所有者确认的草稿标注；结果不得作为正式 test 成绩",
    )
    eval_retr.add_argument("--output", default="artifacts/evals/retrieval")
    sub.add_parser("extract-snapshots", help="离线抽取立场快照（自动选择 LLM/确定性）")
    eval_disc = sub.add_parser("eval-discovery", help="合成库发现精度评测")
    eval_disc.add_argument("--output", default="artifacts/evals/discovery")
    sub.add_parser("serve", help="启动本地 Web UI")
    sub.add_parser("mcp", help="启动 MCP stdio 服务")
    sub.add_parser("verify-vault", help="校验同步幂等且 Vault 只读（哈希不变）")

    args = parser.parse_args(argv)
    settings = Settings.load()
    if args.retrieval_mode:
        settings = replace(settings, retrieval_mode=args.retrieval_mode)
    if args.embedding_backend:
        settings = replace(settings, embedding_backend=args.embedding_backend)
    if args.reranker:
        settings = replace(settings, reranker_backend=args.reranker)

    if args.command == "eval-retrieval":
        from .evaluation import audit_retrieval_golden_groups, eval_retrieval_detailed
        from .importer import VaultSyncService

        repo_root = Path(__file__).resolve().parents[2]
        goldens_doc = json.loads(
            (repo_root / "evals" / "retrieval_goldens.json").read_text(encoding="utf-8")
        )
        real_goldens_path = repo_root / "evals" / "retrieval_goldens.real.json"
        private_groups: dict[str, list[dict[str, object]]] = {}
        private_audit: dict[str, object] | None = None
        if real_goldens_path.exists():
            real_doc = json.loads(real_goldens_path.read_text(encoding="utf-8"))
            private_groups = real_doc.get("groups", {})
            private_audit = audit_retrieval_golden_groups(private_groups)
            goldens_doc["groups"].update(private_groups)
            if "real_dev" in private_groups and "real_test" in private_groups:
                goldens_doc["groups"]["real"] = [
                    *private_groups["real_dev"],
                    *private_groups["real_test"],
                ]
        if args.group == "all":
            wanted = ["synthetic"]
            if "real_dev" in goldens_doc["groups"] and "real_test" in goldens_doc["groups"]:
                wanted.extend(["real_dev", "real_test"])
            else:
                wanted.append("real")
        else:
            wanted = [args.group]
        selected_private = {
            group: goldens_doc["groups"][group]
            for group in wanted
            if group != "synthetic" and group in goldens_doc["groups"]
        }
        if selected_private:
            selected_audit = audit_retrieval_golden_groups(selected_private)
            if selected_audit["draft_case_count"] and not args.include_draft_goldens:
                raise SystemExit(
                    "所选私有 golden 含未经 Vault 所有者确认的草稿标注；默认拒绝把它们当正式成绩。"
                    "请先复核 annotation_status，或仅为草稿诊断显式传入 "
                    "--include-draft-goldens。"
                )
            if selected_audit["draft_case_count"]:
                print(
                    "注意：本次包含草稿标注，输出只能用于标注审计，不是正式 test 成绩。",
                    file=sys.stderr,
                )
        retrieval_report: dict[str, object] = {}
        diagnostics_report: dict[str, object] = {}
        for group in wanted:
            if group not in goldens_doc["groups"]:
                raise SystemExit(
                    f"golden 组 {group!r} 不可用。真实组含私人笔记标题，不入公开仓库；"
                    "如需评测，请在本地放置 evals/retrieval_goldens.real.json 后重试。"
                )
            if group == "synthetic":
                with tempfile.TemporaryDirectory() as tmp:
                    eval_settings = Settings(
                        vault_path=repo_root / "evals" / "cognitive_mvp_vault",
                        database_path=Path(tmp) / "retrieval-eval.db",
                    )
                    eval_db = build_database(eval_settings)
                    try:
                        VaultSyncService(eval_db, eval_settings.vault_path).sync()
                        routes = build_public_evaluation_retrievers(eval_db, eval_settings)
                        detailed = eval_retrieval_detailed(
                            routes, goldens_doc["groups"][group]
                        )
                        retrieval_report[group] = detailed["summary"]
                        diagnostics_report[group] = {
                            "cases": detailed["cases"],
                            "pairwise": detailed["pairwise"],
                        }
                    finally:
                        eval_db.close()
            else:
                # 真实组始终复制到一次性派生库；只有 --real-models 才允许走云端。
                with tempfile.TemporaryDirectory() as tmp:
                    if args.real_models:
                        eval_settings = replace(
                            settings,
                            database_path=Path(tmp) / "retrieval-real-api-eval.db",
                            embedding_backend="api",
                            retrieval_mode="hybrid",
                            reranker_backend="api",
                        )
                        print(
                            "注意：--real-models 会发送查询、标题、标题层级、标签、正文到已配置的 "
                            "Embedding/Rerank API；Vault 只读，向量仅写临时派生库。",
                            file=sys.stderr,
                        )
                    else:
                        eval_settings = replace(
                            settings,
                            database_path=Path(tmp) / "retrieval-real-eval.db",
                            embedding_backend="local_hash",
                            retrieval_mode="hybrid",
                            allow_cloud_embedding=False,
                            allow_cloud_rerank=False,
                        )
                    eval_db = build_database(eval_settings)
                    try:
                        VaultSyncService(eval_db, eval_settings.vault_path).sync()
                        available_paths = {
                            str(row["rel_path"])
                            for row in eval_db.fetchall(
                                "SELECT rel_path FROM sources "
                                "WHERE is_present=1 AND searchable=1"
                            )
                        }
                        audit_retrieval_golden_groups(
                            {group: goldens_doc["groups"][group]}, available_paths
                        )
                        routes = (
                            build_private_api_evaluation_retrievers(eval_db, eval_settings)
                            if args.real_models
                            else build_private_local_evaluation_retrievers(
                                eval_db, eval_settings
                            )
                        )
                        detailed = eval_retrieval_detailed(
                            routes, goldens_doc["groups"][group]
                        )
                        retrieval_report[group] = detailed["summary"]
                        diagnostics_report[group] = {
                            "cases": detailed["cases"],
                            "pairwise": detailed["pairwise"],
                        }
                    finally:
                        eval_db.close()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "retrieval_metrics.json").write_text(
            json.dumps(retrieval_report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (output / "retrieval_diagnostics.json").write_text(
            json.dumps(diagnostics_report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if private_audit is not None:
            (output / "retrieval_dataset_audit.json").write_text(
                json.dumps(private_audit, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        print(json.dumps(retrieval_report, ensure_ascii=False, indent=1))
        return 0

    database = build_database(settings)

    if args.command in {"init", "sync"}:
        from .importer import VaultSyncService

        service = VaultSyncService(database, settings.vault_path)
        report = service.sync()
        retriever = build_retriever(database, settings)
        if retriever.vector.is_cloud:
            print(
                "注意：已显式允许云端 Embedding；将发送每条原子的标题、标题层级、标签、正文。",
                file=sys.stderr,
            )
        filled = retriever.vector.ensure_vectors() if "vector" in retriever.routes else 0
        print(json.dumps({"sync": report, "vectors_filled": filled}, ensure_ascii=False, indent=1))
        return 0

    if args.command == "ask":
        from .agent import AgentHarness

        retriever = build_retriever(database, settings)
        settings_llm = settings
        provider = None
        if settings_llm.backend != "local" and settings_llm.llm_ready:
            from .agent import OpenAIProvider
            from .llm import OpenAICompatibleClient

            provider = OpenAIProvider(OpenAICompatibleClient(settings_llm))
        harness = AgentHarness(database, retriever, settings, provider=provider)
        result = harness.run(args.question, thread_id=args.thread)
        print(result.reply)
        print(f"\n--- backend={result.backend} steps={result.steps} tool_calls={result.tool_calls}"
              f" stop={result.stop_reason} type={result.answer.answer_type} ---")
        return 0

    if args.command == "discover":
        from .cognitive import DiscoveryService

        retriever = build_retriever(database, settings)
        classifier = None
        if settings.llm_ready:
            from .llm import OpenAICompatibleClient
            from .snapshots import LLMChangeClassifier

            classifier = LLMChangeClassifier(OpenAICompatibleClient(settings))
        _, candidates = DiscoveryService(database, retriever, classifier).run_scan(limit=args.limit)
        _print_candidates(candidates)
        return 0

    if args.command == "review":
        from .cognitive import DiscoveryService

        outcome = DiscoveryService(database, build_retriever(database, settings)).review(
            args.discovery_id, args.verdict, args.revision, args.missing_event
        )
        print(json.dumps(outcome, ensure_ascii=False))
        return 0

    if args.command == "eval-agent":
        from .evaluation import run_comparative_eval
        from .importer import VaultSyncService

        # 协议评测必须与开发库隔离：一次性临时库 + 强制合成 Vault
        cases = json.loads(
            (Path(__file__).resolve().parents[2] / "evals" / "agent_cases.json").read_text(encoding="utf-8")
        )["cases"]
        with tempfile.TemporaryDirectory() as tmp:
            eval_settings = Settings(
                vault_path=Path(__file__).resolve().parents[2] / "evals" / "cognitive_mvp_vault",
                database_path=Path(tmp) / "eval.db",
                llm_chat_model="",
            )
            eval_db = Database(eval_settings.database_path)
            eval_db.initialize()
            VaultSyncService(eval_db, eval_settings.vault_path).sync()
            eval_retriever = build_retriever(eval_db, eval_settings)
            eval_retriever.vector.ensure_vectors()
            summary = run_comparative_eval(eval_db, eval_retriever, cases, Path(args.output))
            eval_db.close()
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    if args.command == "extract-snapshots":
        from .llm import OpenAICompatibleClient
        from .snapshots import DeterministicExtractor, LLMBatchExtractor, ensure_snapshots

        retriever = build_retriever(database, settings)
        if settings.llm_ready:
            known = [
                row["topic"]
                for row in database.fetchall(
                    "SELECT DISTINCT topic FROM stance_snapshots LIMIT 50"
                )
            ]
            chosen_extractor: DeterministicExtractor | LLMBatchExtractor = LLMBatchExtractor(
                OpenAICompatibleClient(settings), settings.llm_chat_model, known
            )
        else:
            chosen_extractor = DeterministicExtractor()
        extract_report = ensure_snapshots(database, chosen_extractor)
        topics = database.fetchall(
            "SELECT topic, COUNT(*) AS n FROM stance_snapshots WHERE has_stance=1"
            " GROUP BY topic ORDER BY n DESC LIMIT 15"
        )
        print(json.dumps({
            "report": extract_report,
            "extractor": chosen_extractor.name,
            "topics": {row["topic"]: row["n"] for row in topics},
        }, ensure_ascii=False, indent=1))
        return 0

    if args.command == "eval-discovery":
        from .evaluation import eval_discovery

        retriever = build_retriever(database, settings)
        summary = eval_discovery(database, retriever, Path(args.output))
        print(json.dumps(
            {k: v for k, v in summary.items() if k != "candidates"},
            ensure_ascii=False, indent=1,
        ))
        return 0

    if args.command == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(settings), host="127.0.0.1", port=8766)
        return 0

    if args.command == "mcp":
        from .mcp_server import run_mcp

        run_mcp(settings)
        return 0

    if args.command == "verify-vault":
        from .importer import VaultSyncService, vault_markdown_hash

        before = vault_markdown_hash(settings.vault_path)
        service = VaultSyncService(database, settings.vault_path)
        first = service.sync()
        second = service.sync()
        after = vault_markdown_hash(settings.vault_path)
        ok = before == after and not second["changed"]
        print(json.dumps({
            "vault_read_only": before == after,
            "second_sync_idempotent": not second["changed"],
            "first_sync": first,
        }, ensure_ascii=False, indent=1))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
