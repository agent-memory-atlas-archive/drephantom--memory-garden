"""Inspectable user-authorized memory, separate from raw notes and model hypotheses.

Only a saved user verdict can produce a memory. Revisions are verbatim; confirmations
refer to one dated interpretation and its evidence, not a permanent personality fact.
"""
from __future__ import annotations

import json
from typing import Any

from .db import Database, utc_now


def answer_atom_ids(answer: Any) -> list[int]:
    """All saved source anchors, including evidence omitted from the short prose."""
    found: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'atom_id' and child is not None:
                    try:
                        atom_id = int(child)
                    except (ValueError, TypeError):
                        continue
                    if atom_id > 0:
                        found.add(atom_id)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(answer)
    return sorted(found)


def sources_available(database: Database, atom_ids: list[int]) -> bool:
    ids = sorted(set(atom_ids))
    if not ids:
        return True
    visible = database.fetchone(
        'SELECT COUNT(*) AS n FROM source_atoms a JOIN sources s ON s.id=a.source_id '
        'WHERE s.is_present=1 AND s.searchable=1 AND a.id IN (' + ','.join('?' for _ in ids) + ')', ids)
    return bool(visible and visible['n'] == len(ids))


def topic_key_for(topic: str) -> str:
    from .tools import fallback_topic_key

    return topic if topic.startswith(('terms:', 'text:')) else fallback_topic_key(topic)


def answer_scope(database: Database, message_id: int) -> tuple[str, dict[str, Any]]:
    row = database.fetchone('SELECT thread_id, answer_json,reply_to_user_message_id FROM messages WHERE id=?', (message_id,))
    if not row:
        return '', {}
    try:
        answer = json.loads(row['answer_json'] or '{}')
    except (ValueError, TypeError):
        answer = {}
    topic = str(answer.get('topic') or (answer.get('dialogue') or {}).get('topic_query') or '').strip()
    if not topic:
        # Legacy replies may omit a resolved topic. Prefer the preceding answer's
        # scope over a short pronoun-only follow-up, then fall back to user wording.
        previous = database.fetchall(
            "SELECT answer_json FROM messages WHERE thread_id=? AND id<? "
            "AND role='assistant' AND answer_json IS NOT NULL ORDER BY id DESC LIMIT 6",
            (row['thread_id'], message_id),
        )
        for item in previous:
            try:
                candidate = json.loads(item['answer_json'])
                topic = str(candidate.get('topic') or (candidate.get('dialogue') or {}).get('topic_query') or '').strip()
            except (ValueError, TypeError):
                continue
            if topic:
                break
    if not topic:
        user = database.fetchone('SELECT content FROM messages WHERE id=?', (row['reply_to_user_message_id'],)) if row['reply_to_user_message_id'] else database.fetchone(
            "SELECT content FROM messages WHERE thread_id=? AND id<? AND role='user' ORDER BY id DESC LIMIT 1",
            (row['thread_id'], message_id),
        )
        topic = str(user['content']) if user else ''
    return topic, answer


class MemoryService:
    def __init__(self, database: Database):
        self.database = database

    def backfill(self) -> None:
        """One-time v5 migration: recover resolved topics and superseded decisions."""
        rows = self.database.fetchall('SELECT * FROM verdicts ORDER BY id')
        for row in rows:
            if not row['message_id'] and not row['source_discovery_id']:
                # Old discovery reviews did not retain a discovery id. Never
                # reconstruct personal facts or source consent by guessing.
                self.database.execute("UPDATE verdicts SET status='needs_review' WHERE id=? AND status='active'", (row['id'],))
            if row['message_id']:
                topic, _ = answer_scope(self.database, int(row['message_id']))
                if topic:
                    self.database.execute('UPDATE verdicts SET topic_key=?,topic_label=? WHERE id=?',
                                          (topic_key_for(topic), topic, row['id']))
                self.database.execute(
                    "UPDATE verdicts SET status='superseded' WHERE message_id=? AND id<? AND status='active'",
                    (row['message_id'], row['id']),
                )
        for row in self.database.fetchall('SELECT id FROM verdicts ORDER BY id'):
            self.record_verdict(int(row['id']))

    def record_verdict(self, verdict_id: int) -> None:
        row = self.database.fetchone('SELECT * FROM verdicts WHERE id=?', (verdict_id,))
        if row is None or row['verdict'] == 'defer':
            return
        topic, answer = answer_scope(self.database, int(row['message_id'])) if row['message_id'] else ('', {})
        topic = str(row['topic_label'] or topic or row['topic_key'])
        summary = str(row['confirmed_interpretation'] or answer.get('summary') or '').strip()
        revision = str(row['user_revision'] or '').strip()
        kind = ('user_revision' if revision else 'confirmed_interpretation' if row['verdict'] == 'accurate'
                else 'rejected_interpretation' if row['verdict'] in {'no_change', 'not_my_view'} else 'uncertainty')
        statement = revision or summary
        if not statement:
            statement = {'accurate': '用户确认这组记录的解释贴近自己的看法。',
                         'no_change': '用户判断这组材料不构成观点变化。',
                         'not_my_view': '用户否认这份解释代表自己的观点。'}.get(
                             row['verdict'], '用户认为这份解释仍需要补充或核对。')
        def ids(position: Any) -> list[int]:
            return [int(c['atom_id']) for c in (position or {}).get('citations', []) if c.get('atom_id')]
        evidence = {
            'answer_type': answer.get('answer_type'), 'verdict': row['verdict'],
            'early_atom_ids': ids(answer.get('early_position')),
            'recent_atom_ids': ids(answer.get('recent_position')),
            'atom_ids': answer_atom_ids(answer),
            'missing_event': row['missing_event'],
        }
        if row['source_discovery_id']:
            discovery = self.database.fetchone('SELECT * FROM discoveries WHERE id=?', (row['source_discovery_id'],))
            if discovery:
                evidence.update(early_atom_ids=[discovery['early_atom_id']], recent_atom_ids=[discovery['recent_atom_id']],
                                atom_ids=[discovery['early_atom_id'], discovery['recent_atom_id']])
        connection = self.database.connect()
        was_in_transaction = connection.in_transaction
        connection.execute(
            'INSERT OR IGNORE INTO memory_items(topic_key,topic,kind,statement,status,source_message_id,'
            'source_verdict_id,evidence_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (row['topic_key'], topic, kind, statement[:10000], row['status'], row['message_id'], verdict_id,
             json.dumps(evidence, ensure_ascii=False), row['created_at'], utc_now()),
        )
        if not was_in_transaction:
            connection.commit()

    def list_items(self, topic: str = '', *, include_inactive: bool = False, limit: int = 60, offset: int = 0) -> list[dict[str, Any]]:
        clauses, params = [], []
        if topic.strip():
            clauses.append('i.topic_key=?')
            params.append(topic_key_for(topic))
        if not include_inactive:
            clauses.append("i.status='active' AND v.status='active'")
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        rows = self.database.fetchall(
            'SELECT i.*, m.thread_id AS source_thread_id,m.answer_json,v.status AS verdict_status FROM memory_items i '
            'JOIN verdicts v ON v.id=i.source_verdict_id LEFT JOIN messages m ON m.id=i.source_message_id'
            + where + ' ORDER BY i.id DESC LIMIT ? OFFSET ?', [*params, min(max(limit, 1), 100), max(offset, 0)],
        )
        result = []
        for row in rows:
            item = dict(row)
            item['evidence'] = json.loads(item.pop('evidence_json'))
            try:
                answer = json.loads(item.pop('answer_json') or '{}')
            except (ValueError, TypeError):
                answer = {}
            # Read-through enrichment also closes the gap for already-saved v5/v6
            # items, without reactivating or rewriting a revoked historical memory.
            atom_ids = sorted(set(item['evidence'].get('atom_ids', [])) | set(answer_atom_ids(answer)))
            item['evidence']['atom_ids'] = atom_ids
            verdict_status = item.pop('verdict_status')
            item['available_for_recall'] = (item['status'] == 'active' and verdict_status == 'active'
                                            and sources_available(self.database, atom_ids))
            result.append(item)
        return result

    def verdict_available_for_recall(self, verdict_id: int) -> bool:
        row = self.database.fetchone(
            'SELECT v.status,v.verdict,i.status AS memory_status,i.evidence_json,m.answer_json '
            'FROM verdicts v LEFT JOIN memory_items i ON i.source_verdict_id=v.id '
            'LEFT JOIN messages m ON m.id=v.message_id WHERE v.id=?', (verdict_id,))
        if row is None or row['status'] != 'active' or row['verdict'] == 'defer':
            return False
        if row['memory_status'] is not None and row['memory_status'] != 'active':
            return False
        try:
            answer = json.loads(row['answer_json'] or '{}')
            evidence = json.loads(row['evidence_json'] or '{}')
        except (ValueError, TypeError):
            return False
        return sources_available(self.database, sorted(set(evidence.get('atom_ids', [])) | set(answer_atom_ids(answer))))

    def context_for_topic(self, topic_query: str, limit: int = 6) -> list[dict[str, Any]]:
        if not topic_query.strip():
            return []
        result, budget = [], 6000
        for item in self.list_items(topic_query, limit=30):
            if not item['available_for_recall']:
                continue
            size = len(json.dumps(item, ensure_ascii=False))
            if size > budget:
                # Do not cut the tail off a user's correction: it may contain the
                # exception that changes its meaning. Keep an explicit locator
                # when the whole quote does not fit; never present a prefix as all.
                item = {**item, 'statement': '', 'statement_omitted': True,
                        'statement_chars': len(item['statement']),
                        'recall_boundary': '完整原话未载入；仅保留反馈类型、日期、证据和原消息索引，不推测被省略的内容。'}
                size = len(json.dumps(item, ensure_ascii=False))
                if size > budget:
                    continue
            result.append(item)
            budget -= size
            if len(result) >= min(max(limit, 1), 10):
                break
        return result

    def revoke(self, memory_id: int) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = connection.execute('SELECT * FROM memory_items WHERE id=?', (memory_id,)).fetchone()
            if row is None:
                raise KeyError('这条记忆不存在。')
            connection.execute("UPDATE memory_items SET status='revoked',updated_at=? WHERE id=?", (utc_now(), memory_id))
            connection.execute("UPDATE verdicts SET status='revoked' WHERE id=?", (row['source_verdict_id'],))
        return {'id': memory_id, 'status': 'revoked'}
