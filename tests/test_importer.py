"""导入层测试：frontmatter、时间分离、作者归属、幂等同步、移动稳定身份。"""
from __future__ import annotations

from pathlib import Path

from memory_garden.config import Settings
from memory_garden.db import Database
from memory_garden.importer import (
    VaultSyncService,
    infer_authorship,
    parse_markdown_document,
    parse_wechat_document,
    split_frontmatter,
    vault_markdown_hash,
)


def test_split_frontmatter_flat_and_list() -> None:
    text = "---\ntitle: 测试\ntags:\n  - a\n  - b\ncreated: 2024-01-02\n---\n正文"
    meta, body, _ = split_frontmatter(text)
    assert meta["title"] == "测试"
    assert meta["tags"] == ["a", "b"]
    assert meta["created"] == "2024-01-02"
    assert body.strip() == "正文"


def test_event_time_only_from_explicit_declaration(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("---\ntitle: 转折\ncreated: 2025-12-01\nevent_date: 2021-10-09\n---\n补记", encoding="utf-8")
    doc = parse_markdown_document(path, "note.md", path.read_text(encoding="utf-8"))
    assert doc.event_time == "2021-10-09"
    assert doc.recorded_at == "2025-12-01"


def test_event_time_ignores_body_dates(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("---\ntitle: 日记\n---\n我在 2019 年 3 月 4 日想过这件事", encoding="utf-8")
    doc = parse_markdown_document(path, "note.md", path.read_text(encoding="utf-8"))
    assert doc.event_time is None


def test_authorship_rules() -> None:
    assert infer_authorship({"content_origin": "quote"}, "书摘中的独立", "") == "quoted"
    assert infer_authorship({"generated_by": "ChatGPT"}, "AI 生成的草稿", "") == "ai_generated"
    assert infer_authorship({}, "引用-独立", "") == "quoted"
    assert infer_authorship({}, "AI草稿-独立", "", "AI草稿-独立.md") == "ai_generated"
    assert infer_authorship({}, "普通笔记", "") == "user"


def test_wechat_parse_extracts_message_times() -> None:
    text = "# 墨漪\n\n### 2025-07-01 15:34\n\n直接操控想法意识\n\n### 2025-07-02 09:00\n\n[引用] 回复内容\n"
    doc = parse_wechat_document("_sources/wechat/export.md", text)
    assert doc is not None
    assert len(doc.atoms) == 2
    assert doc.atoms[0].event_time == "2025-07-01T15:34:00"
    assert doc.atoms[0].authorship == "user"
    assert doc.atoms[1].authorship == "quoted"


def test_sync_idempotent_and_vault_untouched(settings: Settings) -> None:
    before = vault_markdown_hash(settings.vault_path)
    database = Database(settings.database_path)
    database.initialize()
    service = VaultSyncService(database, settings.vault_path)
    first = service.sync()
    second = service.sync()
    after = vault_markdown_hash(settings.vault_path)
    assert first["changed"] is True
    assert second["changed"] is False
    assert before == after  # Vault 只读


def test_moved_file_keeps_uid(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "旧目录").mkdir(parents=True)
    note = vault / "旧目录" / "想法.md"
    note.write_text("---\ntitle: 想法\ncreated: 2024-01-01\n---\n内容保持不变", encoding="utf-8")
    database = Database(tmp_path / "db.sqlite")
    database.initialize()
    service = VaultSyncService(database, vault)
    service.sync()
    old_uid = database.fetchone("SELECT uid FROM sources")["uid"]

    (vault / "新目录").mkdir()
    note.rename(vault / "新目录" / "想法.md")
    service.sync()
    rows = database.fetchall("SELECT uid, rel_path, is_present FROM sources")
    assert len(rows) == 1
    assert rows[0]["uid"] == old_uid
    assert rows[0]["rel_path"] == "新目录/想法.md"
    assert rows[0]["is_present"] == 1


def test_removed_file_marked_absent(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    note = vault / "a.md"
    note.write_text("---\ntitle: A\n---\n内容", encoding="utf-8")
    database = Database(tmp_path / "db.sqlite")
    database.initialize()
    service = VaultSyncService(database, vault)
    service.sync()
    note.unlink()
    service.sync()
    assert database.fetchone("SELECT is_present FROM sources")["is_present"] == 0


def test_changed_content_creates_revision(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    note = vault / "a.md"
    note.write_text("---\ntitle: A\n---\n第一版", encoding="utf-8")
    database = Database(tmp_path / "db.sqlite")
    database.initialize()
    service = VaultSyncService(database, vault)
    service.sync()
    note.write_text("---\ntitle: A\n---\n第二版内容不同", encoding="utf-8")
    service.sync()
    revisions = database.fetchall("SELECT content_hash FROM source_revisions")
    assert len(revisions) == 2
