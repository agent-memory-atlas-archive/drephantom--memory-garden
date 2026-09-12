"""Every saved evidence role shares withdrawal and feedback recall boundaries."""
from __future__ import annotations

import json

import pytest

from memory_garden.cognitive import VerdictService
from memory_garden.context import build_thread_context
from memory_garden.db import utc_now
from memory_garden.memory import MemoryService
from memory_garden.models import (
    CognitiveAnswer,
    DialogueState,
    EvidenceItem,
    PositionEvidence,
    SourceCitation,
)


def structured_feedback(database):
    atoms = database.fetchall('SELECT MIN(id) AS id,source_id FROM source_atoms GROUP BY source_id ORDER BY source_id LIMIT 6')
    assert len(atoms) == 6
    citations = [SourceCitation(atom_id=row['id'], recorded_at='2024-01-01') for row in atoms]
    summary = '这份模型解释必须随全部依赖来源一起受撤下约束。'
    answer = CognitiveAnswer(answer_type='no_clear_change', summary=summary, topic='自主判断',
        dialogue=DialogueState(topic_query='自主判断'), citations=citations[:1],
        early_position=PositionEvidence(statement='早期', status='source_grounded', citations=citations[1:2]),
        recent_position=PositionEvidence(statement='较近', status='latest_memory_candidate', citations=citations[2:3]),
        supporting_evidence=[EvidenceItem(statement='支持', relation='support', citations=citations[3:4])],
        counter_evidence=[EvidenceItem(statement='挑战', relation='challenge', citations=citations[4:5])],
        interval_events=[EvidenceItem(statement='区间', relation='within_interval', citations=citations[5:6])])
    tid = database.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', ('合成回看', utc_now()))
    uid = database.execute('INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,?,?,?)',
                           (tid, 'user', '回看自主判断', utc_now()))
    mid = database.execute('INSERT INTO messages(thread_id,role,content,answer_json,created_at,reply_to_user_message_id) VALUES(?,?,?,?,?,?)',
                           (tid, 'assistant', summary, answer.model_dump_json(), utc_now(), uid))
    verdict = VerdictService(database).save_verdict(message_id=mid, verdict='partly_accurate', user_revision='用户明确保存的修正。')
    return atoms, tid, mid, verdict


@pytest.mark.parametrize('source_index', range(6))
def test_each_structured_evidence_role_controls_recall_after_withdrawal(database, source_index):
    atoms, tid, _, verdict = structured_feedback(database)
    memory = MemoryService(database)
    recalled = memory.context_for_topic('自主判断')
    assert set(recalled[0]['evidence']['atom_ids']) == {row['id'] for row in atoms}
    context = build_thread_context(database, tid, topic='自主判断')
    assert {anchor['atom_id'] for anchor in context.data['source_anchors']} == {row['id'] for row in atoms}
    assert context.data['user_corrections']
    database.execute('UPDATE sources SET searchable=0 WHERE id=?', (atoms[source_index]['source_id'],))
    assert memory.context_for_topic('自主判断') == []
    assert not memory.list_items()[0]['available_for_recall']
    assert not memory.verdict_available_for_recall(verdict['id'])
    assert VerdictService(database).for_topic('自主判断') == []
    for scope in ('自主判断', None):
        context = build_thread_context(database, tid, topic=scope)
        assert context.data['user_corrections'] == []
        assert context.data['source_anchors'] == []
        assert '这份模型解释' not in context.rendered + str(context.recent_messages)
        assert '回看自主判断' in str(context.recent_messages) + context.rendered


def test_old_cached_memory_evidence_is_enriched_without_reactivating_status(database):
    atoms, _, _, verdict = structured_feedback(database)
    database.execute('UPDATE memory_items SET evidence_json=? WHERE source_verdict_id=?',
                     (json.dumps({'atom_ids': [atoms[0]['id']]}), verdict['id']))
    database.execute('UPDATE sources SET is_present=0 WHERE id=?', (atoms[-1]['source_id'],))
    memory = MemoryService(database)
    assert memory.context_for_topic('自主判断') == []
    item = memory.list_items()[0]
    assert set(item['evidence']['atom_ids']) == {row['id'] for row in atoms}
    memory.revoke(item['id'])
    database.execute('UPDATE sources SET is_present=1 WHERE id=?', (atoms[-1]['source_id'],))
    assert memory.context_for_topic('自主判断') == []
    assert not memory.verdict_available_for_recall(verdict['id'])
    assert VerdictService(database).for_topic('自主判断') == []
    assert memory.list_items(include_inactive=True)[0]['status'] == 'revoked'


def test_long_user_revision_keeps_its_qualifying_tail_verbatim(database):
    _, _, mid, _ = structured_feedback(database)
    revision = '我在解释这段特定时期的经历。' * 100 + '但这只针对当时的处境，不代表永久观点，也不能推广到职业选择。'
    VerdictService(database).save_verdict(message_id=mid, verdict='partly_accurate', user_revision=revision)
    recalled = MemoryService(database).context_for_topic('自主判断')
    assert recalled[0]['statement'] == revision
    assert '不能推广到职业选择' in recalled[0]['statement']


def test_oversize_interpretation_retains_locator_instead_of_silent_truncation(database):
    _, _, mid, _ = structured_feedback(database)
    interpretation = '很长的历史模型解释。' * 700
    VerdictService(database).save_verdict(message_id=mid, verdict='not_my_view', confirmed_interpretation=interpretation)
    recalled = MemoryService(database).context_for_topic('自主判断')
    assert recalled[0]['statement'] == '' and recalled[0]['statement_omitted'] is True
    assert recalled[0]['kind'] == 'rejected_interpretation'
    assert recalled[0]['source_message_id'] == mid
    assert recalled[0]['statement_chars'] == len(interpretation)
