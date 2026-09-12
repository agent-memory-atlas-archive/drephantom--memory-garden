"""User memory must retain scope/provenance and honor later edits and withdrawal."""
from __future__ import annotations

import json
import sqlite3

from memory_garden.cognitive import VerdictService
from memory_garden.db import SCHEMA_VERSION, Database
from memory_garden.memory import MemoryService
from memory_garden.tools import CognitiveTools, fallback_topic_key


def add_answer(database, topic='自主判断', question='那后来呢？', citations=None):
    thread_id = database.execute("INSERT INTO threads(title,created_at) VALUES('回看','2026-01-01')")
    database.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,'2026-01-01')", (thread_id, question))
    answer = {'topic': topic, 'answer_type': 'no_clear_change', 'summary': '这两段记录不足以说明改变。', 'citations': citations or []}
    message_id = database.execute(
        "INSERT INTO messages(thread_id,role,content,answer_json,created_at) VALUES(?,'assistant',?,?, '2026-01-01')",
        (thread_id, answer['summary'], json.dumps(answer, ensure_ascii=False)),
    )
    return message_id


def test_followup_feedback_uses_resolved_topic_and_verbatim_revision(database, retriever):
    message_id = add_answer(database)
    saved = VerdictService(database).save_verdict(message_id=message_id, verdict='partly_accurate', user_revision='我会听意见，但决定由自己做。')
    assert saved['topic_key'] == fallback_topic_key('自主判断')
    memory = MemoryService(database).context_for_topic('自主判断')
    assert len(memory) == 1 and memory[0]['statement'] == '我会听意见，但决定由自己做。'
    assert memory[0]['source_message_id'] == message_id and memory[0]['source_verdict_id'] == saved['id']
    assert not MemoryService(database).context_for_topic('职业判断')
    assert not VerdictService(database).for_topic(fallback_topic_key('后来'))
    assert CognitiveTools(database, retriever).get_user_verdicts({'topic': '自主判断'}).data['verdicts']


def test_models_and_deferred_feedback_do_not_create_long_term_view(database):
    message_id = add_answer(database)
    assert not MemoryService(database).list_items()
    VerdictService(database).save_verdict(message_id=message_id, verdict='defer')
    assert not MemoryService(database).context_for_topic('自主判断')
    assert database.fetchone('SELECT verdict FROM verdicts')['verdict'] == 'defer'


def test_edit_then_revoke_never_resurrects_earlier_judgment(database, retriever):
    message_id = add_answer(database)
    service, memory = VerdictService(database), MemoryService(database)
    first = service.save_verdict(message_id=message_id, verdict='not_my_view', user_revision='旧修正')
    second = service.save_verdict(message_id=message_id, verdict='accurate', user_revision='重新核对后的修正')
    assert [item['id'] for item in service.for_topic('自主判断')] == [second['id']]
    active = memory.context_for_topic('自主判断')
    assert [item['statement'] for item in active] == ['重新核对后的修正']
    memory.revoke(active[0]['id'])
    assert memory.context_for_topic('自主判断') == []
    assert service.for_topic('自主判断') == []
    assert not CognitiveTools(database, retriever).get_user_verdicts({'topic': '自主判断'}).data['verdicts']
    archived = memory.list_items(include_inactive=True)
    assert {item['status'] for item in archived} == {'revoked', 'superseded'}
    assert database.fetchone('SELECT status FROM verdicts WHERE id=?', (first['id'],))['status'] == 'superseded'


def test_different_evidence_interpretations_keep_their_own_dated_memory(database):
    first, second = add_answer(database), add_answer(database)
    for message_id, revision in [(first, '当时更在意别人的认可。'), (second, '这次更愿意自己判断。')]:
        VerdictService(database).save_verdict(message_id=message_id, verdict='partly_accurate', user_revision=revision)
    memory = MemoryService(database).context_for_topic('自主判断')
    assert len(memory) == 2
    assert {item['source_message_id'] for item in memory} == {first, second}
    assert all(item['created_at'] for item in memory)


def test_withdrawn_source_is_not_recalled_as_long_term_memory(database):
    atom = database.fetchone('SELECT id,source_id FROM source_atoms LIMIT 1')
    message_id = add_answer(database, citations=[{'atom_id': atom['id']}])
    VerdictService(database).save_verdict(message_id=message_id, verdict='accurate')
    memory = MemoryService(database)
    assert memory.context_for_topic('自主判断')
    database.execute('UPDATE sources SET searchable=0 WHERE id=?', (atom['source_id'],))
    assert not memory.context_for_topic('自主判断')
    assert not VerdictService(database).for_topic('自主判断')
    assert memory.list_items()[0]['available_for_recall'] is False


def test_v4_migration_backfills_only_latest_feedback_with_resolved_scope(database):
    message_id = add_answer(database)
    for verdict in ['accurate', 'not_my_view']:
        database.execute("INSERT INTO verdicts(message_id,topic_key,verdict,created_at) VALUES(?,'terms:后来',?,'2026-01-01')", (message_id, verdict))
    database.execute("UPDATE schema_meta SET value='4' WHERE key='schema_version'")
    path = database.path
    database.close()
    migrated = Database(path)
    migrated.initialize()
    active = MemoryService(migrated).context_for_topic('自主判断')
    assert len(active) == 1 and active[0]['kind'] == 'rejected_interpretation'
    assert migrated.fetchone('SELECT COUNT(*) AS n FROM verdicts')['n'] == 2
    with sqlite3.connect(path.with_name(path.name + f'.pre-v{SCHEMA_VERSION}.bak')) as backup:
        assert backup.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == '4'
        assert backup.execute('SELECT topic_key FROM verdicts LIMIT 1').fetchone()[0] == 'terms:后来'
    migrated.initialize()
    assert len(MemoryService(migrated).list_items(include_inactive=True)) == 2
    migrated.close()
