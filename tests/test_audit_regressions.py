"""Regressions found during the September product audit; synthetic data only."""
from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from memory_garden.agent import AgentHarness
from memory_garden.cognitive import DiscoveryService
from memory_garden.db import Database
from memory_garden.evaluation import ScriptedProvider, standard_trace_script
from memory_garden.importer import VaultSyncService
from memory_garden.snapshots import DeterministicExtractor, ensure_snapshots
from memory_garden.web import create_app


def test_editing_discovered_note_preserves_old_evidence(settings, tmp_path):
    vault = tmp_path / 'editable-vault'
    shutil.copytree(settings.vault_path, vault)
    database = Database(tmp_path / 'revisions.db')
    database.initialize()
    sync = VaultSyncService(database, vault)
    sync.sync()
    from memory_garden.retrieval import RetrievalQuery, build_retriever

    retriever = build_retriever(database, replace(settings, vault_path=vault))
    _, candidates = DiscoveryService(database, retriever).run_scan(limit=3)
    candidate = candidates[0]
    old = database.fetchone(
        'SELECT a.*, s.rel_path FROM source_atoms a JOIN sources s ON s.id=a.source_id '
        'WHERE a.id=?', (candidate.early_atom_id,),
    )
    note = vault / old['rel_path']
    note.write_text('---\ntitle: 修订记录\ncreated: 2025-01-01\n---\n全新的修订内容', encoding='utf-8')
    sync.sync()
    retained = database.fetchone('SELECT text FROM source_atoms WHERE id=?', (old['id'],))
    assert retained is not None and retained['text'] == old['text']
    assert not database.fetchall('PRAGMA foreign_key_check')
    current = retriever.search(RetrievalQuery(text='全新的修订内容'))
    assert current and all(hit.atom_id != old['id'] for hit in current)
    assert database.fetchone('SELECT COUNT(*) AS n FROM discoveries')['n'] > 0
    database.close()


def test_local_discovery_never_uses_saved_generation_connection(settings, database, monkeypatch):
    ensure_snapshots(database, DeterministicExtractor())
    def forbidden(*args, **kwargs):
        raise AssertionError('local mode attempted a cloud classification')
    monkeypatch.setattr('memory_garden.llm.OpenAICompatibleClient.chat_json', forbidden)
    configured = replace(settings, backend='local', llm_base_url='https://example.invalid',
                         llm_api_key='synthetic-test-key', llm_chat_model='fake')
    with TestClient(create_app(configured)) as client:
        assert client.get('/api/discover').status_code == 200


def test_batch_tool_calls_cannot_exceed_budget(settings, database, retriever):
    class BatchProvider:
        name = 'batch-test'
        def complete(self, messages, tools):
            if not tools:
                return {'content': '证据不足。', 'tool_calls': []}
            return {'content': '', 'tool_calls': [
                {'id': f'call_{i}', 'type': 'function', 'function': {
                    'name': 'get_user_verdicts',
                    'arguments': json.dumps({'topic': f'主题{i}'})}}
                for i in range(5)
            ]}
    result = AgentHarness(database, retriever, replace(settings, agent_max_tool_calls=1),
                          provider=BatchProvider()).run('自主判断')
    assert result.tool_calls <= 1


def test_factual_answer_cannot_drop_all_citations(settings, database, retriever):
    script = standard_trace_script('自主判断', '这一变化与区间内的经历有关')
    script[-1] = {'final': '你已经彻底不再依赖任何人的认可，这已得到记录证实。'}
    result = AgentHarness(database, retriever, settings, provider=ScriptedProvider(script)).run('自主判断')
    assert '彻底不再依赖任何人' not in result.reply
    assert '[A' in result.reply


def test_database_cannot_silently_switch_vault(settings, tmp_path):
    with TestClient(create_app(settings)) as client:
        assert client.get('/api/health').json()['counts']['sources'] > 0
    other = tmp_path / 'other-vault'
    other.mkdir()
    with pytest.raises(ValueError, match='Vault|vault|笔记库'):
        create_app(replace(settings, vault_path=other))


def test_nonexistent_vault_does_not_mark_every_source_removed(settings, database, tmp_path):
    before = database.fetchone('SELECT COUNT(*) AS n FROM sources WHERE is_present=1')['n']
    with pytest.raises(ValueError, match='Vault|vault|笔记库'):
        VaultSyncService(database, tmp_path / 'missing').sync()
    assert database.fetchone('SELECT COUNT(*) AS n FROM sources WHERE is_present=1')['n'] == before


def test_long_note_tail_and_original_line_numbers_are_indexed(settings, tmp_path):
    from memory_garden.importer import parse_markdown_document
    from memory_garden.retrieval import RetrievalQuery, build_retriever
    vault = tmp_path / 'long-vault'
    vault.mkdir()
    note = vault / 'long.md'
    text = '---\ntitle: 很长的记录\ncreated: 2024-01-01\n---\n' + ('前面的日常记录。\n' * 700) + '尾部独有的星辰线索'
    note.write_text(text, encoding='utf-8')
    parsed = parse_markdown_document(note, 'long.md', text)
    assert len(parsed.atoms) > 1
    assert parsed.atoms[-1].text.endswith('尾部独有的星辰线索')
    for atom in parsed.atoms:
        span = '\n'.join(text.splitlines()[atom.line_start-1:atom.line_end])
        assert atom.text in span
    db = Database(tmp_path / 'long.db')
    db.initialize()
    VaultSyncService(db, vault).sync()
    retriever = build_retriever(db, replace(settings, vault_path=vault, retrieval_mode='bm25'))
    hits = retriever.search(RetrievalQuery(text='星辰线索'))
    assert hits and '尾部独有' in hits[0].fields['excerpt']
    db.close()


def test_v3_upgrade_backs_up_before_adding_history_columns(tmp_path):
    path = tmp_path / 'v3.db'
    db = Database(path)
    db.initialize()
    db.execute("UPDATE schema_meta SET value='3' WHERE key='schema_version'")
    db.close()
    legacy = sqlite3.connect(path)
    legacy.execute('ALTER TABLE source_atoms DROP COLUMN is_current')
    legacy.execute('ALTER TABLE source_atoms DROP COLUMN revision_hash')
    legacy.commit()
    legacy.close()
    db.initialize()
    assert {'is_current', 'revision_hash'} <= {r['name'] for r in db.fetchall('PRAGMA table_info(source_atoms)')}
    from memory_garden.db import SCHEMA_VERSION

    backup = sqlite3.connect(path.with_name(path.name+f'.pre-v{SCHEMA_VERSION}.bak'))
    assert backup.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == '3'
    backup.close()
    db.close()


def test_long_chat_export_preserves_tail_time_and_original_lines():
    from memory_garden.importer import parse_wechat_document
    text = '### 2025-01-01 12:30\n' + ('这是一段很长的消息。\n' * 700) + '末尾的独有记忆'
    parsed = parse_wechat_document('wechat.md', text)
    assert parsed is not None and len(parsed.atoms) > 1
    assert parsed.atoms[-1].text.endswith('末尾的独有记忆')
    for atom in parsed.atoms:
        assert atom.event_time == '2025-01-01T12:30:00'
        assert atom.text in '\n'.join(text.splitlines()[atom.line_start-1:atom.line_end])


def test_tool_deadline_stops_followup_work_and_keeps_attempt_count(settings, database, retriever, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr('memory_garden.budget.time.monotonic', lambda: clock[0])
    from memory_garden.tools import ToolObservation
    def slow_tool(self, args):
        clock[0] += 2.0
        return ToolObservation(tool='get_user_verdicts', data={'verdicts': []})
    monkeypatch.setattr('memory_garden.tools.CognitiveTools.get_user_verdicts', slow_tool)
    script = [{'tool':'get_user_verdicts', 'args':{'topic':'自主判断'}}, {'final':'不能到达此处'}]
    provider = ScriptedProvider(script)
    result = AgentHarness(database, retriever, replace(settings, agent_tool_timeout_seconds=1),
                          provider=provider).run('自主判断')
    assert result.stop_reason == 'time_budget_exceeded'
    assert result.tool_calls == 1 and provider.cursor == 1
    assert result.answer.abstained


def test_citation_list_exactly_matches_reply_and_feedback_survives_reload(settings):
    import re
    with TestClient(create_app(settings)) as client:
        result = client.post('/api/ask', json={'question':'自主判断的想法有没有变化？'}).json()
        refs = {int(v) for v in re.findall(r'\[A(\d+)\]', result['reply'])}
        citations = result['answer']['citations']
        assert refs == {c['atom_id'] for c in citations}
        assert all(c['path'] and c['line_start'] and c['excerpt'] for c in citations)
        assert client.post('/api/verdict', json={'message_id':result['message_id'],
            'verdict':'partly_accurate','user_revision':'我会参考别人，但不把决定交出去。'}).status_code == 200
        history = client.get('/api/threads/'+str(result['thread_id'])).json()
        assert history[-1]['verdict']['verdict'] == 'partly_accurate'
        assert history[-1]['verdict']['user_revision'].startswith('我会参考别人')
        assert client.post('/api/ask', json={'question':'测试','thread_id':99999}).status_code == 404
        assert client.post('/api/ask', json={'question':'   '}).status_code == 400


def test_local_origin_and_host_are_checked_and_name_is_escaped(settings):
    name = "O'Brien <script>alert(1)</script>"
    with TestClient(create_app(replace(settings, assistant_name=name))) as client:
        assert client.get('/api/settings', headers={'Host':'unexpected.example'}).status_code == 400
        assert client.post('/api/sync', json={}, headers={'Origin':'https://unexpected.example'}).status_code == 403
        page = client.get('/').text
        assert '<script>alert(1)</script>' not in page
        assert 'O&#x27;Brien' in page
        assert client.get('/static/garden.js').status_code == 200


def test_dismissal_does_not_become_a_belief_verdict(settings, database, retriever):
    service = DiscoveryService(database, retriever)
    _, first = service.run_scan(limit=3)
    target = first[0]
    before = database.fetchone('SELECT COUNT(*) AS n FROM verdicts')['n']
    service.dismiss(target.discovery_id)
    _, second = service.run_scan(limit=3)
    assert all((c.early_atom_id,c.recent_atom_id) != (target.early_atom_id,target.recent_atom_id) for c in second)
    assert database.fetchone('SELECT COUNT(*) AS n FROM verdicts')['n'] == before


def test_chat_retries_share_one_deadline(settings, monkeypatch):
    import httpx

    from memory_garden.budget import BudgetExceeded
    from memory_garden.llm import OpenAICompatibleClient
    clock = [1000.0]
    monkeypatch.setattr('memory_garden.llm.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('memory_garden.llm.time.sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    timeouts = []
    class Client:
        def __init__(self, *, timeout):
            timeouts.append(timeout)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def post(self, *args, **kwargs):
            clock[0] += 2
            raise httpx.ConnectError('synthetic timeout')
    monkeypatch.setattr('memory_garden.llm.httpx.Client', Client)
    configured = replace(settings, llm_base_url='https://example.invalid', llm_api_key='synthetic', llm_max_retries=5)
    with pytest.raises(BudgetExceeded):
        OpenAICompatibleClient(configured).chat({}, timeout_seconds=3)
    assert timeouts == [3.0, 0.5]
