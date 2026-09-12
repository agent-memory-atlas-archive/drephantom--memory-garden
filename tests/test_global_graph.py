"""Whole-workspace graph boundaries, using only temporary synthetic records."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from memory_garden.chat_import import ChatImportService
from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.graph import build_map
from memory_garden.importer import VaultSyncService
from memory_garden.web import create_app


def _workspace(tmp_path: Path, notes: dict[str, str]) -> tuple[Database, Settings]:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, text in notes.items():
        path = vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    settings = Settings(vault_path=vault, database_path=tmp_path / "graph.db")
    database = Database(settings.database_path)
    database.initialize()
    VaultSyncService(database, vault).sync()
    return database, settings


def test_global_includes_more_than_local_limit_and_isolated_chats(tmp_path: Path) -> None:
    database, settings = _workspace(tmp_path, {
        **{f"n{i:03}.md": f"Record {i}. [[n{i + 1:03}]]" for i in range(313)},
        "n313.md": "End of the linked records.", "isolated.md": "A disconnected record.",
    })
    imports = ChatImportService(database, settings)
    preview = imports.preview(filename="synthetic.json", content=json.dumps([
        {"sender": "合成本人", "timestamp": "2026-01-01 12:00:00", "text": "聊天里的 [[n000]] 不应变成双链。"},
    ]))
    chat = imports.commit(preview["preview_id"], ["合成本人"], confirm_agent_access=True)
    before = database.connect().total_changes
    graph = build_map(database, scope="global")
    assert graph["scope"] == "global"
    assert len(graph["nodes"]) == graph["meta"]["total_visible"] == 316
    assert graph["meta"]["total_matching"] == graph["meta"]["total_candidates"] == 316
    assert len(graph["edges"]) == 313
    assert graph["meta"]["returned_nodes"] == 316 and not graph["meta"]["truncated"]
    assert any(node["path"] == "isolated.md" for node in graph["nodes"])
    chat_node = next(node for node in graph["nodes"] if node["uid"] == chat["source_uid"])
    assert chat_node["source_kind"] == "chat" and chat_node["date_range"]["start"] == "2026-01-01"
    assert not any(chat["source_uid"] in (edge["source"], edge["target"]) for edge in graph["edges"])
    assert len(build_map(database)["nodes"]) == 40
    assert database.connect().total_changes == before
    assert database.fetchone("SELECT COUNT(*) FROM atom_vectors")[0] == 0
    imports.deactivate(chat["source_uid"])
    assert len(build_map(database, scope="global")["nodes"]) == 315


def test_global_search_filters_records_but_local_focus_expands_neighbors(tmp_path: Path) -> None:
    database, _ = _workspace(tmp_path, {
        "root.md": "---\ntitle: 树木观察\ntags: [植物]\n---\n[[other]]",
        "other.md": "静止的石头。",
        "花园/path-match.md": "晴天。",
        "body.md": "记录一次新的经历。",
    })
    for query, expected in [("树木观察", "root.md"), ("植物", "root.md"),
                            ("花园", "花园/path-match.md"), ("新的经历", "body.md")]:
        graph = build_map(database, scope="global", query=query)
        assert [node["path"] for node in graph["nodes"]] == [expected]
        assert graph["query"] == query and graph["meta"]["total_visible"] == 4
        assert graph["meta"]["total_matching"] == 1 and graph["edges"] == []
    row = database.fetchone("SELECT uid FROM sources WHERE rel_path='root.md'")
    assert row
    assert {node["path"] for node in build_map(database, source_uid=row["uid"])["nodes"]} == {"root.md", "other.md"}
    # A stale topic/focus from the local tab must not turn 'global' into a subset.
    assert len(build_map(database, topic="树木观察", source_uid=row["uid"], scope="global")["nodes"]) == 4
    missing = build_map(database, scope="global", query="未匹配的唯一检索")
    assert missing["nodes"] == [] and missing["meta"]["total_visible"] == 4
    assert missing["meta"]["total_matching"] == 0 and not missing["meta"]["truncated"]


def test_global_hides_excluded_and_withdrawn_targets_and_their_link_labels(tmp_path: Path) -> None:
    database, _ = _workspace(tmp_path, {
        "visible.md": "[[私密标题|私密别名]] [[withdrawn]] [[derived]] [[allowed]]",
        "私密标题.md": "私密原文。", "withdrawn.md": "已撤回原文。",
        "derived.md": "模型派生原文。", "allowed.md": "仍可阅读的原文。",
    })
    database.execute("UPDATE sources SET searchable=0 WHERE rel_path='私密标题.md'")
    database.execute("UPDATE sources SET is_present=0 WHERE rel_path='withdrawn.md'")
    database.execute("UPDATE sources SET authorship='derived' WHERE rel_path='derived.md'")
    graph = build_map(database, scope="global")
    serialized = json.dumps(graph, ensure_ascii=False)
    for label in ("私密标题", "私密别名", "withdrawn", "derived", "模型派生原文"):
        assert label not in serialized
    assert graph["meta"]["total_visible"] == 2 and len(graph["edges"]) == 1
    assert graph["meta"]["unresolved_links"] == 3
    assert build_map(database, scope="global", query="私密标题")["nodes"] == []


def test_global_limits_report_all_omissions_and_never_dangle_edges(tmp_path: Path) -> None:
    database, _ = _workspace(tmp_path, {
        f"n{i:03}.md": f"[[n{(i + 1) % 505:03}]]" for i in range(505)
    })
    graph = build_map(database, scope="global", limit=999)
    assert graph["meta"]["total_visible"] == 505
    assert graph["meta"]["node_limit"] == graph["meta"]["returned_nodes"] == 500
    assert graph["meta"]["omitted_nodes"] == 5 and graph["meta"]["total_edges"] == 505
    assert graph["meta"]["omitted_edges"] == 505 - len(graph["edges"])
    assert graph["meta"]["truncated"] and not graph["meta"]["link_scan_truncated"]
    ids = {node["id"] for node in graph["nodes"]}
    assert all(edge["source"] in ids and edge["target"] in ids for edge in graph["edges"])
    limited = build_map(database, scope="global", limit=2)
    assert len(limited["nodes"]) == 2 and limited["meta"]["omitted_nodes"] == 503
    assert build_map(database, scope="global", limit=2) == limited
    # Search can reach a node omitted from the default overview.
    missing = next(f"n{i:03}.md" for i in range(505) if f"n{i:03}.md" not in {node["path"] for node in graph["nodes"]})
    assert any(node["path"] == missing for node in build_map(database, scope="global", query=missing)["nodes"])


def test_global_edge_limit_is_separate_from_node_limit(tmp_path: Path) -> None:
    database, _ = _workspace(tmp_path, {
        f"n{i:02}.md": "\n".join(f"[[n{j:02}]]" for j in range(56) if j != i)
        for i in range(56)
    })
    graph = build_map(database, scope="global")
    assert graph["meta"]["returned_nodes"] == 56 and graph["meta"]["omitted_nodes"] == 0
    assert len(graph["edges"]) == graph["meta"]["edge_limit"] == 3000
    assert graph["meta"]["total_edges"] == 56 * 55
    assert graph["meta"]["omitted_edges"] == 80 and graph["meta"]["truncated"]


def test_per_source_scan_limit_is_exposed_as_incomplete_not_a_complete_edge_count(tmp_path: Path) -> None:
    database, _ = _workspace(tmp_path, {"start.md": "\n".join(["[[end]]"] * 300), "end.md": "Destination."})
    graph = build_map(database, scope="global")
    assert graph["meta"]["total_edges"] == graph["meta"]["links_per_source_limit"] == 256
    assert graph["meta"]["link_scan_truncated"] and graph["meta"]["truncated"]
    assert graph["meta"]["omitted_nodes"] == graph["meta"]["omitted_edges"] == 0


def test_map_api_supports_global_scope_and_preserves_local_default(tmp_path: Path) -> None:
    _, settings = _workspace(tmp_path, {f"n{i:03}.md": f"Synthetic {i}." for i in range(316)})
    with TestClient(create_app(settings)) as client:
        local = client.get("/api/map").json()
        assert local["scope"] == "local" and len(local["nodes"]) == 30
        global_map = client.get("/api/map", params={"scope": "global"}).json()
        assert len(global_map["nodes"]) == 316 and global_map["meta"]["total_visible"] == 316
        searched = client.get("/api/map", params={"scope": "global", "query": "n315.md"}).json()
        assert [node["path"] for node in searched["nodes"]] == ["n315.md"]
        assert client.get("/api/map", params={"scope": "global", "limit": 500}).status_code == 200
        assert client.get("/api/map", params={"scope": "global", "limit": 501}).status_code == 422
        assert client.get("/api/map", params={"scope": "remote"}).status_code == 422
