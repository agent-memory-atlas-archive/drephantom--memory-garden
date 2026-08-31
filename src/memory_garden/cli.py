"""命令行入口：init / sync / ask / discover / review / eval-* / serve / mcp / verify-vault。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .db import Database
from .retrieval import BM25Retriever, HybridRetriever, VectorRetriever


def build_database(settings: Settings) -> Database:
    database = Database(settings.database_path)
    database.initialize()
    return database


def build_retriever(database: Database, settings: Settings) -> HybridRetriever:
    embedding_client = None
    vector = VectorRetriever(database, embedding_client)
    return HybridRetriever(BM25Retriever(database), vector)


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
    eval_retr = sub.add_parser("eval-retrieval", help="运行检索双路指标")
    eval_retr.add_argument("--group", choices=["synthetic", "real", "all"], default="all")
    eval_retr.add_argument("--output", default="artifacts/evals/retrieval")
    sub.add_parser("extract-snapshots", help="离线抽取立场快照（自动选择 LLM/确定性）")
    eval_disc = sub.add_parser("eval-discovery", help="合成库发现精度评测")
    eval_disc.add_argument("--output", default="artifacts/evals/discovery")
    sub.add_parser("serve", help="启动本地 Web UI")
    sub.add_parser("mcp", help="启动 MCP stdio 服务")
    sub.add_parser("verify-vault", help="校验同步幂等且 Vault 只读（哈希不变）")

    args = parser.parse_args(argv)
    settings = Settings.load()
    database = build_database(settings)

    if args.command in {"init", "sync"}:
        from .importer import VaultSyncService

        service = VaultSyncService(database, settings.vault_path)
        report = service.sync()
        retriever = build_retriever(database, settings)
        filled = retriever.vector.ensure_vectors()
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
        import tempfile

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

    if args.command == "eval-retrieval":
        from .evaluation import eval_retrieval

        goldens_doc = json.loads(
            (Path(__file__).resolve().parents[2] / "evals" / "retrieval_goldens.json").read_text(encoding="utf-8")
        )
        # 真实 Vault 的 golden 组含私人笔记标题，不入公开仓库：本地可选文件存在时合并。
        real_goldens_path = Path(__file__).resolve().parents[2] / "evals" / "retrieval_goldens.real.json"
        if real_goldens_path.exists():
            real_doc = json.loads(real_goldens_path.read_text(encoding="utf-8"))
            goldens_doc["groups"].update(real_doc.get("groups", {}))
        retriever = build_retriever(database, settings)
        bm25_only = HybridRetriever(retriever.bm25, retriever.vector, routes=("bm25",))
        vector_only = HybridRetriever(retriever.bm25, retriever.vector, routes=("vector",))
        groups = args.group
        retrieval_report: dict[str, object] = {}
        wanted = ["synthetic", "real"] if groups == "all" else [groups]
        for group in wanted:
            if group not in goldens_doc["groups"]:
                raise SystemExit(
                    f"golden 组 {group!r} 不可用。真实组含私人笔记标题，不入公开仓库；"
                    "如需评测，请在本地放置 evals/retrieval_goldens.real.json 后重试。"
                )
            retrieval_report[group] = eval_retrieval(
                {"bm25": bm25_only, "vector": vector_only, "hybrid": retriever},
                goldens_doc["groups"][group],
            )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "retrieval_metrics.json").write_text(
            json.dumps(retrieval_report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(retrieval_report, ensure_ascii=False, indent=1))
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
