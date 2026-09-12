"""Synthetic note maps: provenance, privacy, ambiguity and bounded neighborhoods."""
from __future__ import annotations

import json
from pathlib import Path

from memory_garden.db import Database
from memory_garden.graph import build_local_map
from memory_garden.importer import VaultSyncService


def _vault(tmp_path: Path, notes: dict[str, str]) -> tuple[Database, Path, VaultSyncService]:
    vault = tmp_path / "vault"
    for name, content in notes.items():
        path = vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    database = Database(tmp_path / "map.db")
    database.initialize()
    service = VaultSyncService(database, vault)
    service.sync()
    return database, vault, service


def _uid(database: Database, path: str) -> str:
    row = database.fetchone("SELECT uid FROM sources WHERE rel_path=?", (path,))
    assert row
    return str(row["uid"])


def test_relative_links_aliases_anchors_and_encoded_paths(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "journey/start.md": "---\ncreated: 2025-01-02\n---\n# 回看\n[[观点#^block|当时]]\n[下一条](../记录%20空间.md#新观点)\n[带井号](../观点%23备份.md)",
        "观点.md": "独立判断的记录。",
        "记录 空间.md": "---\nevent_date: 2020-03-01\n---\n第二条记录。",
        "观点#备份.md": "带井号的文件名。",
    })
    graph = build_local_map(database, source_uid=_uid(database, "journey/start.md"))
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 3
    wiki = next(edge for edge in graph["edges"] if edge["evidence"]["alias"] == "当时")
    assert wiki["type"] == "explicit_link"
    assert wiki["status"] == "source_explicit"
    assert wiki["evidence"]["quote"] == "[[观点#^block|当时]]"
    assert wiki["evidence"]["anchor"] == "^block"
    assert wiki["evidence"]["path"] == "journey/start.md"
    assert wiki["evidence"]["line_start"] == 4
    assert wiki["evidence"]["line_end"] == 7
    recorded = next(node for node in graph["nodes"] if node["path"] == "记录 空间.md")
    assert recorded["event_time"] == "2020-03-01"
    assert recorded["recorded_at"] != recorded["event_time"]


def test_ambiguous_basename_is_not_guessed(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "A/start.md": "[[重复]] 和 [[B/重复|明确路径]]",
        "A/重复.md": "A 的记录。", "B/重复.md": "B 的记录。",
    })
    graph = build_local_map(database, source_uid=_uid(database, "A/start.md"))
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["target"] == _uid(database, "B/重复.md")
    assert graph["meta"]["unresolved_links"] == 1
    assert {node["path"] for node in graph["nodes"]} == {"A/start.md", "B/重复.md"}


def test_hidden_targets_are_neither_nodes_nor_preview_labels(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "visible.md": "独立判断 [[秘密标题|别名秘密]] 和 [[allowed]]",
        "秘密标题.md": "不会出现在地图的内容。",
        "allowed.md": "允许阅读的记录。",
        "hidden-backlink.md": "不可见回链 [[visible]]",
        "derived.md": "派生解释 [[visible]]",
    })
    database.execute("UPDATE sources SET searchable=0 WHERE rel_path IN (?,?)",
                     ("秘密标题.md", "hidden-backlink.md"))
    database.execute("UPDATE sources SET authorship='derived' WHERE rel_path='derived.md'")
    graph = build_local_map(database, topic="独立判断")
    serialized = json.dumps(graph, ensure_ascii=False)
    assert "秘密标题" not in serialized
    assert "别名秘密" not in serialized
    assert "hidden-backlink" not in serialized
    assert "派生解释" not in serialized
    assert len(graph["edges"]) == 1
    assert len(graph["nodes"]) == 2
    assert graph["meta"]["unresolved_links"] == 1
    hidden_focus = build_local_map(database, source_uid=_uid(database, "秘密标题.md"))
    assert hidden_focus["nodes"] == []
    assert hidden_focus["focus_source_uid"] is None


def test_hidden_basename_collision_does_not_retarget_link(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "start.md": "[[重复]]",
        "A/重复.md": "可见的一份。", "B/重复.md": "排除的一份。",
    })
    database.execute("UPDATE sources SET searchable=0 WHERE rel_path='B/重复.md'")
    graph = build_local_map(database, source_uid=_uid(database, "start.md"))
    assert graph["edges"] == []
    assert graph["meta"]["unresolved_links"] == 1


def test_current_revision_and_rename_are_resolved_from_present_paths(tmp_path: Path) -> None:
    database, vault, service = _vault(tmp_path, {
        "start.md": "[[old]]", "old.md": "同一记录，移动时身份保持。",
        "other.md": "另一条记录。",
    })
    uid = _uid(database, "old.md")
    (vault / "old.md").rename(vault / "renamed.md")
    service.sync()
    stale = build_local_map(database, source_uid=_uid(database, "start.md"))
    assert stale["edges"] == []
    assert stale["meta"]["unresolved_links"] == 1
    (vault / "start.md").write_text("[[renamed]]", encoding="utf-8")
    service.sync()
    current = build_local_map(database, source_uid=_uid(database, "start.md"))
    assert current["edges"][0]["target"] == uid
    assert current["edges"][0]["evidence"]["quote"] == "[[renamed]]"
    atom_id = current["edges"][0]["evidence"]["atom_id"]
    assert database.fetchone("SELECT is_current FROM source_atoms WHERE id=?", (atom_id,))[0] == 1
    (vault / "renamed.md").unlink()
    service.sync()
    assert build_local_map(database, source_uid=_uid(database, "start.md"))["edges"] == []


def test_focus_is_one_hop_and_limits_keep_edges_inside_nodes(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "root.md": "[[a]] [[b]]", "a.md": "[[far]]", "b.md": "B note",
        "incoming.md": "[[root]]", "far.md": "Two-hop record",
        "unrelated.md": "Unrelated note",
    })
    uid = _uid(database, "root.md")
    graph = build_local_map(database, source_uid=uid)
    assert {node["path"] for node in graph["nodes"]} == {"root.md", "a.md", "b.md", "incoming.md"}
    limited = build_local_map(database, source_uid=uid, limit=2)
    assert len(limited["nodes"]) == 2
    assert limited["nodes"][0]["uid"] == uid
    assert limited["meta"]["omitted_nodes"] == 2
    assert limited["meta"]["truncated"] is True
    ids = {node["id"] for node in limited["nodes"]}
    assert all(edge["source"] in ids and edge["target"] in ids for edge in limited["edges"])
    assert build_local_map(database, source_uid=uid, limit=2) == limited


def test_topic_can_include_unlinked_records_without_semantic_edges(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "early.md": "---\ncreated: 2020-01-01\n---\n自主判断与别人意见。",
        "recent.md": "---\ncreated: 2025-01-01\n---\n自主判断由我自己做出。",
        "unrelated.md": "春天的植物。",
    })
    graph = build_local_map(database, topic="自主判断")
    assert len(graph["nodes"]) == 2
    assert all(node["matched_topic"] for node in graph["nodes"])
    assert graph["edges"] == []
    assert build_local_map(database, topic="不存在的主题")["nodes"] == []


def test_code_comments_external_links_attachments_and_traversal_do_not_create_edges(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {
        "start.md": "```md\n[[hidden]]\n```\n`[[hidden]]`\n<!-- [[hidden]] -->\n"
                    "[web](https://example.org/hidden.md) ![img](hidden.md) [[image.png]]\n"
                    "[escape](../../hidden.md) [malformed](//[invalid)\n[[shown]]",
        "hidden.md": "No authored note link points here.",
        "shown.md": "The one visible relationship.",
    })
    graph = build_local_map(database, source_uid=_uid(database, "start.md"))
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["target"] == _uid(database, "shown.md")


def test_topic_search_stays_local_and_map_does_not_write(tmp_path: Path) -> None:
    database, _, _ = _vault(tmp_path, {"a.md": "写作 [[b]]", "b.md": "阅读"})
    before = database.connect().total_changes
    graph = build_local_map(database, topic="写作")
    assert len(graph["nodes"]) == 2
    assert database.connect().total_changes == before
    assert database.fetchone("SELECT COUNT(*) FROM atom_vectors")[0] == 0
    assert build_local_map(database, limit=1000)["meta"]["node_limit"] == 40
