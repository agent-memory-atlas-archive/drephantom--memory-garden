"""Only synthetic notes are used to verify the selected Vault read boundary."""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_garden.db import Database
from memory_garden.importer import VaultSyncService, iter_markdown_paths, vault_markdown_hash


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f'This platform does not permit synthetic symlinks: {exc}')


def test_ordinary_markdown_filters_and_sorting_are_preserved(tmp_path):
    vault = tmp_path / 'vault'
    for relative in ['z.md', 'folder/a.md', '_draft.md', '未命名.md',
                     'templates/example.md', '.obsidian/settings.md', '.git/readme.md', 'plain.txt']:
        note = vault / relative
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text('合成笔记。', encoding='utf-8')
    assert [relative for _, relative in iter_markdown_paths(vault)] == ['folder/a.md', 'z.md']


def test_scan_permission_error_propagates_without_marking_existing_notes_absent(tmp_path, monkeypatch):
    vault = tmp_path / 'vault'
    vault.mkdir()
    (vault / 'inside.md').write_text('合成库内记录。', encoding='utf-8')
    database = Database(tmp_path / 'derived.db')
    try:
        database.initialize()
        service = VaultSyncService(database, vault)
        service.sync()

        def unreadable_walk(*args, onerror=None, **kwargs):
            if onerror is not None:
                onerror(PermissionError('synthetic unreadable directory'))
            return iter(())

        monkeypatch.setattr('memory_garden.importer.os.walk', unreadable_walk)
        with pytest.raises(PermissionError, match='synthetic unreadable directory'):
            service.sync()
        assert database.fetchone('SELECT COUNT(*) FROM sources WHERE is_present=1')[0] == 1
        assert database.fetchone('SELECT COUNT(*) FROM sync_runs')[0] == 1
    finally:
        database.close()


def test_external_file_link_never_enters_hash_or_index(tmp_path):
    vault = tmp_path / 'vault'
    vault.mkdir()
    (vault / 'inside.md').write_text('合成库内记录。', encoding='utf-8')
    outside = tmp_path / 'outside.md'
    outside.write_text('合成库外记录第一版。', encoding='utf-8')
    _symlink(vault / 'outside-link.md', outside)
    before = vault_markdown_hash(vault)
    outside.write_text('合成库外记录第二版。', encoding='utf-8')
    assert vault_markdown_hash(vault) == before
    database = Database(tmp_path / 'derived.db')
    try:
        database.initialize()
        VaultSyncService(database, vault).sync()
        assert [row['rel_path'] for row in database.fetchall('SELECT rel_path FROM sources')] == ['inside.md']
        assert all('库外' not in row['text'] for row in database.fetchall('SELECT text FROM source_atoms'))
    finally:
        database.close()


def test_same_vault_file_link_is_allowed_but_directory_alias_is_not_followed(tmp_path):
    vault = tmp_path / 'vault'
    notes = vault / 'notes'
    notes.mkdir(parents=True)
    original = notes / 'original.md'
    original.write_text('合成库内记录。', encoding='utf-8')
    _symlink(vault / 'alias.md', original)
    _symlink(vault / 'directory-alias', notes, directory=True)
    _symlink(notes / 'parent-loop', vault, directory=True)
    assert [relative for _, relative in iter_markdown_paths(vault)] == ['alias.md', 'notes/original.md']


def test_external_directory_and_broken_file_links_are_skipped(tmp_path):
    vault = tmp_path / 'vault'
    vault.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'outside.md').write_text('合成库外记录。', encoding='utf-8')
    _symlink(vault / 'outside-directory', outside, directory=True)
    _symlink(vault / 'broken.md', tmp_path / 'not-present.md')
    assert iter_markdown_paths(vault) == []
