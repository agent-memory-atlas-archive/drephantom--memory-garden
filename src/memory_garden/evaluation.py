"""评测：三配置消融对比（Agent 评测）+ 双路/混合检索指标（检索评测）。

诚实边界（写进报告，不写进吹嘘）：
- 对比评测使用离线确定性 ScriptedProvider，验证的是**协议约束力**（引用守卫、拒答、
  反例覆盖、判定遵守），不是真实模型的准确率；
- 检索指标在人工标注 golden 集上计算 recall@k / MRR，区分 BM25 / 哈希向量 /
  mock Embedding / RRF 混合 / RRF + 重排序，
  合成集与真实 Vault 分别标注，不混用。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import AgentHarness, AgentRunResult
from .db import Database
from .retrieval import HybridRetriever, RetrievalQuery
from .tools import fallback_topic_key

CAUSAL_MARKERS = ("因为", "导致", "原因是", "使得", "由于")


# ── 确定性脚本 Provider：按剧本回放工具调用，最终文本带真实引用占位符 ────────


class ScriptedProvider:
    name = "scripted"

    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.cursor = 0
        self.endpoint_refs: list[int] = []
        self.endpoint_dates: list[str] = []
        self.latest_refs: list[int] = []
        self.challenge_refs: list[int] = []

    def _harvest(self, messages: list[dict[str, Any]]) -> None:
        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            latest_ids = re.findall(r'"atom_id":\s*(\d+)', content)
            if latest_ids:
                self.latest_refs = [int(value) for value in latest_ids]
            try:
                rendered = content.splitlines()[1]
                payload = json.loads(rendered)
            except (IndexError, json.JSONDecodeError):
                payload = {}
            if payload.get("stance") == "challenge":
                challenge_hits = [
                    item for item in payload.get("hits", [])
                    if "反例" in str(item.get("title") or "")
                    or "反例" in " ".join(str(tag) for tag in item.get("tags") or [])
                    or "求证" in str(item.get("title") or "")
                ]
                if challenge_hits:
                    self.challenge_refs = [int(challenge_hits[0]["atom_id"])]
            # 端点引用与日期来自时间线/候选观察，而非最后一条反例命中
            if '"timeline"' in content or '"candidates"' in content:
                if latest_ids:
                    self.endpoint_refs = [int(latest_ids[0]), int(latest_ids[-1])]
                dates = re.findall(r'"date":\s*"(20\d{2}-\d{2}-\d{2})', content)
                if dates:
                    self.endpoint_dates = [dates[0], dates[-1]]
            break

    @staticmethod
    def _fill(
        text: str, refs: list[int], dates: list[str], latest_refs: list[int]
    ) -> str:
        if len(refs) >= 2:
            text = text.replace("{EARLY}", str(refs[0])).replace("{RECENT}", str(refs[-1]))
        elif len(refs) == 1:
            text = text.replace(" [A{RECENT}]", "").replace("{EARLY}", str(refs[0]))
        # 未填充的占位引用整段剔除，避免脚本回答出现伪引用
        text = re.sub(r"\s*\[A\{(EARLY|RECENT)\}\]", "", text)
        if latest_refs:
            text = text.replace("{CHALLENGE}", str(latest_refs[0]))
        else:
            text = re.sub(r"\s*\[A\{CHALLENGE\}\]", "", text)
        text = text.replace("{EARLY_DATE}", dates[0] if dates else "").replace(
            "{RECENT_DATE}", dates[-1] if dates else ""
        )
        return text

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self._harvest(messages)
        if self.cursor >= len(self.script):
            return {"content": "证据不足，暂时不能形成判断。", "tool_calls": []}
        step = self.script[self.cursor]
        self.cursor += 1
        if "tool" in step:
            call_id = f"call_{self.cursor}"
            args = {
                key: self._fill(
                    str(value), self.endpoint_refs, self.endpoint_dates,
                    self.challenge_refs or self.latest_refs,
                )
                for key, value in (step.get("args") or {}).items()
            }
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": step["tool"], "arguments": json.dumps(args, ensure_ascii=False)},
                    }
                ],
            }
        text = self._fill(
            str(step.get("final", "")), self.endpoint_refs, self.endpoint_dates,
            self.challenge_refs or self.latest_refs,
        )
        return {"content": text, "tool_calls": []}


def standard_trace_script(topic: str, hypothesis: str) -> list[dict[str, Any]]:
    """一次标准认知回溯的工具序列：检索 → 时间线 → 候选 → 区间 → 正反证据 → 作答。"""
    return [
        {"tool": "search_sources", "args": {"query": topic, "limit": 8}},
        {"tool": "get_topic_timeline", "args": {"topic": topic}},
        {"tool": "find_change_candidates", "args": {"topic": topic, "limit": 1}},
        {"tool": "find_interval_events", "args": {"topic": topic, "date_from": "{EARLY_DATE}", "date_to": "{RECENT_DATE}"}},
        {"tool": "search_hypothesis_evidence", "args": {"hypothesis": hypothesis, "query": topic, "stance": "support"}},
        {"tool": "search_hypothesis_evidence", "args": {"hypothesis": hypothesis, "query": topic, "stance": "challenge"}},
        {
            "final": (
                "基于本轮核对：早期记录 [A{EARLY}] 与近期记录 [A{RECENT}] 的表达存在差异，"
                "值得向你确认。区间内的经历与变化只是时间相邻，不等于因果；"
                "与解释不一致的记录 [A{CHALLENGE}] 也已一并列出，"
                "最近一条是否仍代表你现在的看法需要你确认。"
            )
        },
    ]


@dataclass
class CaseResult:
    case_id: str
    config: str
    answer_type: str
    checks: dict[str, bool]
    passed: bool
    reply: str
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str = ""
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    private_vault_sent: bool = False


def _run_case(
    harness: AgentHarness,
    case: dict[str, Any],
    config: str,
    provider_factory: Callable[[str], tuple[Any, bool]] | None = None,
) -> CaseResult:
    query = str(case["query"])
    # 评测用例之间判定记忆互不泄漏：每个用例前清空全部判定，再按需注入先验
    harness.database.execute("DELETE FROM verdicts")
    prior = case.get("prior_verdict")
    if prior:
        harness.database.execute(
            "INSERT INTO verdicts(message_id, topic_key, verdict, user_revision, created_at)"
            " VALUES(NULL, ?, ?, ?, '2026-01-01T00:00:00Z')",
            (fallback_topic_key(query), prior["verdict"], prior.get("user_revision", "")),
        )
    if provider_factory is not None:
        provider, load_verdicts = provider_factory(config)
    elif config == "one_shot_baseline":
        provider = ScriptedProvider(
            [{"final": f"关于「{query}」，你的想法经历了明显的演变，因为你的经历改变了你的看法。"}]
        )
        load_verdicts = False
    else:
        provider = ScriptedProvider(
            standard_trace_script(query, "这一变化与区间内的经历有关")
        )
        load_verdicts = config == "full_agent"
    # These cases already declare a retrieval topic; they do not evaluate free-
    # form intent recognition. The explicit instruction unwraps to the exact
    # original query, preserving scripted tool arguments and prior-verdict keys.
    result: AgentRunResult = harness.run(f'回看「{query}」', load_verdicts=load_verdicts, provider=provider)
    answer = result.answer

    expected_type = str(case.get("expected_answer_type") or "")
    requires_citation = bool(
        case.get(
            "requires_citation",
            expected_type in {"traced_change", "no_clear_change"},
        )
    )
    checks: dict[str, bool] = {
        "task_completion": answer.answer_type == expected_type,
        "citation_valid": _reply_refs_in_trace(
            result, require_citation=requires_citation
        ),
        "abstention_correct": bool(case.get("expect_abstain")) == answer.abstained,
    }
    expected_titles = {
        str(case[key])
        for key in ("early_title", "recent_title")
        if case.get(key)
    }
    if expected_titles:
        checks["evidence_coverage"] = expected_titles.issubset(
            _reply_citation_titles(harness.database, result)
        )
    if case.get("challenge_title"):
        challenge_title = str(case["challenge_title"])
        checks["counter_coverage"] = (
            challenge_title in result.reply
            or challenge_title in _reply_citation_titles(harness.database, result)
        )
    if case.get("prior_verdict"):
        checks["feedback_adherence"] = answer.answer_type == expected_type
    if case.get("forbid_causal_claim"):
        checks["no_causal_claim"] = not any(marker in result.reply for marker in CAUSAL_MARKERS)
    if case.get("expect_event_before_recorded"):
        checks["event_time_respected"] = "延迟" not in result.reply or "记录" in result.reply

    # 缺端点场景：即便检索到相近来源，也不得把最近记录冒充当前观点
    if expected_type == "insufficient_evidence":
        checks["task_completion"] = checks["task_completion"] and (
            not case.get("expect_abstain") or answer.abstained
        )
    passed = all(checks.values())
    return CaseResult(
        case_id=str(case["id"]),
        config=config,
        answer_type=answer.answer_type,
        checks=checks,
        passed=passed,
        reply=result.reply,
        tools_used=[str(event.get("tool")) for event in (getattr(result, "trace", None) or [])],
        stop_reason=result.stop_reason,
        error=result.error,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        private_vault_sent=result.private_vault_sent,
    )


def _reply_refs(result: AgentRunResult) -> set[int]:
    return {int(value) for value in re.findall(r"\[A(\d+)\]", result.reply)}


def _reply_refs_in_trace(
    result: AgentRunResult, *, require_citation: bool = False
) -> bool:
    trace_refs = {
        int(ref)
        for event in result.trace
        for ref in event.get("refs") or []
    }
    reply_refs = _reply_refs(result)
    if require_citation and not reply_refs:
        return False
    return reply_refs.issubset(trace_refs)


def _reply_citation_titles(database: Database, result: AgentRunResult) -> set[str]:
    titles: set[str] = set()
    for atom_id in _reply_refs(result):
        row = database.fetchone(
            "SELECT s.title FROM source_atoms a JOIN sources s ON s.id=a.source_id "
            "WHERE a.id=?",
            (atom_id,),
        )
        if row is not None:
            titles.add(str(row["title"]))
    return titles


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return int(ordered[index])


def run_comparative_eval(
    database: Database,
    retriever: HybridRetriever,
    cases: list[dict[str, Any]],
    output_dir: Path | None = None,
    provider_factory: Callable[[str], tuple[Any, bool]] | None = None,
    on_case_done: Callable[[str, str, bool], None] | None = None,
    configs: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_configs = list(
        configs or ["one_shot_baseline", "agent_without_feedback", "full_agent"]
    )
    allowed_configs = {"one_shot_baseline", "agent_without_feedback", "full_agent"}
    if not selected_configs or any(config not in allowed_configs for config in selected_configs):
        raise ValueError("configs 必须从 one_shot_baseline/agent_without_feedback/full_agent 中选择")
    results: list[CaseResult] = []
    for config in selected_configs:
        harness = AgentHarness(database, retriever, _eval_settings())
        for case in cases:
            item = _run_case(harness, case, config, provider_factory)
            results.append(item)
            if on_case_done is not None:
                on_case_done(config, str(case["id"]), item.passed)
    summary: dict[str, Any] = {
        "mode": "real_provider" if provider_factory is not None else "scripted",
        "configs": {},
        "total_cases": len(cases),
        "total_runs": len(cases) * len(selected_configs),
    }
    metric_keys = [
        "task_completion", "citation_valid", "counter_coverage",
        "abstention_correct", "feedback_adherence", "no_causal_claim",
    ]
    for config in selected_configs:
        config_results = [item for item in results if item.config == config]
        metrics: dict[str, float] = {}
        for key in metric_keys:
            relevant = [item for item in config_results if key in item.checks]
            metrics[key] = (
                round(sum(item.checks[key] for item in relevant) / len(relevant), 4)
                if relevant
                else -1.0
            )
        metrics["pass_rate"] = round(
            sum(item.passed for item in config_results) / len(config_results), 4
        )
        latencies = [item.latency_ms for item in config_results]
        metrics["provider_error_rate"] = round(
            sum(bool(item.error) for item in config_results) / len(config_results), 4
        )
        metrics["latency_p50_ms"] = _percentile(latencies, 0.50)
        metrics["latency_p95_ms"] = _percentile(latencies, 0.95)
        metrics["prompt_tokens"] = sum(item.prompt_tokens for item in config_results)
        metrics["completion_tokens"] = sum(
            item.completion_tokens for item in config_results
        )
        metrics["private_payload_rate"] = round(
            sum(item.private_vault_sent for item in config_results) / len(config_results),
            4,
        )
        summary["configs"][config] = metrics
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "summary": summary,
                    "cases": [
                        {
                            "case_id": item.case_id,
                            "config": item.config,
                            "answer_type": item.answer_type,
                            "checks": item.checks,
                            "passed": item.passed,
                            "reply": item.reply,
                            "tools_used": item.tools_used,
                            "stop_reason": item.stop_reason,
                            "error": item.error,
                            "prompt_tokens": item.prompt_tokens,
                            "completion_tokens": item.completion_tokens,
                            "latency_ms": item.latency_ms,
                            "private_vault_sent": item.private_vault_sent,
                        }
                        for item in results
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return summary


def _eval_settings():
    from .config import Settings

    return Settings(vault_path=Path("."), database_path=Path(".local/eval.db"))


# ── 发现精度评测：合成库上的候选查全/查准 ────────────────────────────────────

DISCOVERY_EXPECTATIONS: dict[str, Any] = {
    "expected_pairs": [
        ["自主判断-2019.md", "自主判断-2024.md"],
        ["完美主义-2019.md", "完美主义-2024.md"],
        ["独处-2020.md", "一个人-2024.md"],
    ],
    "expected_contrast_notes": ["显式对比-写作.md"],
    "excluded_pairs": [
        ["谨慎决策-2020.md", "谨慎决策-2024.md"],  # 措辞漂移不得成为候选
    ],
}


def _atom_path(database: Database, atom_id: int) -> str:
    row = database.fetchone(
        "SELECT s.rel_path FROM source_atoms a JOIN sources s ON s.id=a.source_id WHERE a.id=?",
        (atom_id,),
    )
    return str(row["rel_path"]) if row else ""


def eval_discovery(
    database: Database,
    retriever: Any,
    output_dir: Path | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """在合成库上度量发现质量：预期变化对的召回、措辞漂移对的排除、整体查准。"""
    from .cognitive import DiscoveryService
    from .snapshots import DeterministicExtractor, ensure_snapshots

    # 评测从干净快照重建：抽取逻辑变化时结果仍可复现
    database.execute("DELETE FROM stance_snapshots")
    stats = ensure_snapshots(database, DeterministicExtractor())
    service = DiscoveryService(database, retriever)
    _, candidates = service.run_scan(limit=limit)
    returned: list[dict[str, Any]] = []
    for item in candidates:
        early_path = _atom_path(database, item.early_atom_id)
        recent_path = _atom_path(database, item.recent_atom_id)
        record = item.model_dump()
        record.update(
            {
                "discovery_id": item.discovery_id,
                "early_path": early_path,
                "recent_path": recent_path,
                "pair": sorted([early_path, recent_path]),
            }
        )
        returned.append(record)
    expected = {tuple(sorted(pair)) for pair in DISCOVERY_EXPECTATIONS["expected_pairs"]}
    excluded = {tuple(sorted(pair)) for pair in DISCOVERY_EXPECTATIONS["excluded_pairs"]}
    contrast_notes = set(DISCOVERY_EXPECTATIONS["expected_contrast_notes"])

    hits, related, false_positives, leaked = [], [], [], []
    seen_expected: set[tuple[str, str]] = set()
    expected_endpoints = {path for pair in expected for path in pair}
    for record in returned:
        pair_key = tuple(record["pair"])
        if record["early_path"] == record["recent_path"] and record["early_path"] in contrast_notes:
            hits.append(record)  # 显式对比句候选
            continue
        if pair_key in expected:
            hits.append(record)
            seen_expected.add(pair_key)
        elif pair_key in excluded:
            leaked.append(record)
        elif expected_endpoints & set(pair_key):
            # 相邻子对（如 早期→区间经历）：转折点定位的合法产物，单独立档不记为错
            related.append(record)
        else:
            false_positives.append(record)

    judged = len(hits) + len(false_positives)
    summary = {
        "snapshots_extracted": stats.get("extracted", 0),
        "candidates_returned": len(returned),
        "expected_pair_recall": round(len(seen_expected & expected) / len(expected), 4),
        "excluded_pair_leak": len(leaked),
        "related_adjacent_pairs": len(related),
        "false_positives": len(false_positives),
        "precision_strict": round(len(hits) / len(returned), 4) if returned else 0.0,
        "precision_judged": round(len(hits) / judged, 4) if judged else 0.0,
        "candidates": returned,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "discovery_eval.json").write_text(
            json.dumps(
                {
                    "expectations": DISCOVERY_EXPECTATIONS,
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return summary


# ── 检索评测：路径级 HitRate / Recall / Precision / MRR / nDCG ─────────────

VERIFIED_GOLDEN_STATUSES = {"human_verified", "legacy_human_verified"}


def audit_retrieval_golden_groups(
    groups: dict[str, list[dict[str, Any]]],
    available_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Validate private split integrity without returning queries or source paths."""
    seen_ids: set[str] = set()
    query_to_id: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    draft_case_ids: list[str] = []
    missing_path_case_ids: set[str] = set()
    split_paths: dict[str, set[str]] = {}
    errors: list[str] = []

    for group_name, cases in groups.items():
        paths_in_group: set[str] = set()
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"{group_name}_legacy_{index:03d}")
            if case_id in seen_ids:
                errors.append(f"duplicate case id {case_id}")
            seen_ids.add(case_id)
            query = str(case.get("query") or "").strip()
            if not query:
                errors.append(f"empty query in {case_id}")
            elif query in query_to_id:
                errors.append(f"duplicate query in {query_to_id[query]} and {case_id}")
            else:
                query_to_id[query] = case_id
            raw_paths = case.get("relevant_paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                errors.append(f"empty relevant_paths in {case_id}")
                relevant_paths: list[str] = []
            else:
                relevant_paths = [str(path) for path in raw_paths if str(path)]
                if len(relevant_paths) != len(raw_paths):
                    errors.append(f"blank relevant_path in {case_id}")
                if len(set(relevant_paths)) != len(relevant_paths):
                    errors.append(f"duplicate relevant_path in {case_id}")
            paths_in_group.update(relevant_paths)
            if available_paths is not None and any(
                path not in available_paths for path in relevant_paths
            ):
                missing_path_case_ids.add(case_id)
            status = str(
                case.get("annotation_status") or "unverified_legacy"
            )
            status_counts[status] += 1
            if status not in VERIFIED_GOLDEN_STATUSES:
                draft_case_ids.append(case_id)
            category_counts[str(case.get("category") or "legacy_unspecified")] += 1
        split_paths[group_name] = paths_in_group

    dev_test_overlap = split_paths.get("real_dev", set()).intersection(
        split_paths.get("real_test", set())
    )
    if dev_test_overlap:
        errors.append(f"real_dev/real_test relevant path overlap count={len(dev_test_overlap)}")
    if missing_path_case_ids:
        errors.append(
            "relevant paths unavailable for cases="
            + ",".join(sorted(missing_path_case_ids))
        )
    if errors:
        raise ValueError("retrieval golden audit failed: " + "; ".join(errors))
    return {
        "group_counts": {name: len(cases) for name, cases in groups.items()},
        "total_cases": sum(len(cases) for cases in groups.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "draft_case_count": len(draft_case_ids),
        "draft_case_ids": draft_case_ids,
        "dev_test_relevant_path_overlap": 0,
        "private_text_included": False,
    }


def _unique_paths(paths: list[str]) -> list[str]:
    """Atom 排名映射为 source path 排名，同一笔记只保留首次出现的位置。"""
    return list(dict.fromkeys(path for path in paths if path))


def _ndcg_at_k(paths: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(paths[:k], start=1)
        if path in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _pairwise_counts(
    cases: list[dict[str, Any]], baseline: str, challenger: str, metric: str
) -> dict[str, int]:
    counts = {"challenger_wins": 0, "ties": 0, "challenger_losses": 0}
    for case in cases:
        routes = case["routes"]
        if baseline not in routes or challenger not in routes:
            continue
        baseline_value = float(routes[baseline][metric])
        challenger_value = float(routes[challenger][metric])
        if challenger_value > baseline_value:
            counts["challenger_wins"] += 1
        elif challenger_value < baseline_value:
            counts["challenger_losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def eval_retrieval_detailed(
    retrievers: dict[str, Any],
    goldens: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """计算 source-path 级指标，并返回不含查询/路径/正文的逐例诊断。"""
    if k < 1:
        raise ValueError("k 必须大于 0")
    metric_values: dict[str, dict[str, list[float]]] = {
        name: {
            "hit_rate": [],
            "recall": [],
            "precision": [],
            "mrr": [],
            "ndcg": [],
        }
        for name in retrievers
    }
    cases: list[dict[str, Any]] = []
    # 一个 source 可切成多个 atom。先取足够深的 atom 排名，再按 path 去重，
    # 避免 Top10 atom 只对应 2--4 篇笔记而伪装成 path-level Top5。
    evaluation_depth = max(k, 30)
    for case_index, golden in enumerate(goldens, start=1):
        relevant = {
            str(path) for path in golden.get("relevant_paths", []) if str(path)
        }
        if not relevant:
            continue
        case_report: dict[str, Any] = {
            "case_id": str(golden.get("id") or f"case_{case_index:03d}"),
            "relevant_total": len(relevant),
            "routes": {},
        }
        for name, retriever in retrievers.items():
            hits = retriever.search(
                RetrievalQuery(text=str(golden["query"]), limit=evaluation_depth)
            )
            unique_paths = _unique_paths(
                [str(hit.fields.get("path") or "") for hit in hits]
            )
            top_k = unique_paths[:k]
            relevant_found = len(relevant.intersection(top_k))
            found_positions = [
                index
                for index, path in enumerate(unique_paths, start=1)
                if path in relevant
            ]
            hit_rate = 1.0 if relevant_found else 0.0
            recall = relevant_found / len(relevant)
            precision = relevant_found / k
            reciprocal_rank = 1.0 / found_positions[0] if found_positions else 0.0
            ndcg = _ndcg_at_k(unique_paths, relevant, k)
            values = metric_values[name]
            values["hit_rate"].append(hit_rate)
            values["recall"].append(recall)
            values["precision"].append(precision)
            values["mrr"].append(reciprocal_rank)
            values["ndcg"].append(ndcg)
            case_report["routes"][name] = {
                "first_relevant_rank": found_positions[0] if found_positions else None,
                f"relevant_found@{k}": relevant_found,
                f"unique_results@{k}": len(top_k),
                "reciprocal_rank": round(reciprocal_rank, 6),
                f"ndcg@{k}": round(ndcg, 6),
            }
        cases.append(case_report)

    summary: dict[str, dict[str, float | int]] = {}
    for name, values in metric_values.items():
        case_count = len(values["recall"])

        def mean(
            metric: str,
            current_values: dict[str, list[float]] = values,
            current_count: int = case_count,
        ) -> float:
            return (
                round(sum(current_values[metric]) / current_count, 4)
                if current_count
                else 0.0
            )

        summary[name] = {
            f"hit_rate@{k}": mean("hit_rate"),
            f"recall@{k}": mean("recall"),
            f"precision@{k}": mean("precision"),
            "mrr": mean("mrr"),
            f"ndcg@{k}": mean("ndcg"),
            "cases": case_count,
            "evaluation_depth": evaluation_depth,
        }

    pairwise: dict[str, Any] = {}
    for baseline, challenger in (
        ("hybrid", "hybrid_rerank"),
        ("hybrid", "hybrid_rerank_fused"),
        ("hybrid_rerank", "hybrid_rerank_fused"),
    ):
        if baseline not in retrievers or challenger not in retrievers:
            continue
        label = f"{baseline}_vs_{challenger}"
        pairwise[label] = {
            "reciprocal_rank": _pairwise_counts(
                cases, baseline, challenger, "reciprocal_rank"
            ),
            f"ndcg@{k}": _pairwise_counts(cases, baseline, challenger, f"ndcg@{k}"),
        }
    return {"summary": summary, "cases": cases, "pairwise": pairwise}


def eval_retrieval(
    retrievers: dict[str, Any],
    goldens: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, dict[str, float | int]]:
    """向后兼容的摘要入口；详细诊断见 ``eval_retrieval_detailed``。"""
    return eval_retrieval_detailed(retrievers, goldens, k)["summary"]
