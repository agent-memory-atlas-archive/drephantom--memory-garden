"""Budgeted extractive working context rebuilt from canonical messages each turn.

No model-generated summary is fed back into this compressor. User words remain
verbatim, assistant interpretations never become facts, and omissions are explicit.
This is disposable working context; it does not write durable personal memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .db import Database
from .memory import MemoryService, answer_atom_ids, sources_available, topic_key_for
from .models import CognitiveAnswer


@dataclass
class ThreadContext:
    data: dict[str, Any]
    recent_messages: list[dict[str, str]]
    selected_user_ids: list[int]
    scanned_count: int
    window_limit: int
    max_chars: int
    scanned_chars: int = 0

    @property
    def rendered(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, separators=(',', ':'))

    @property
    def char_count(self) -> int:
        return len(self.rendered) + sum(len(message['content']) for message in self.recent_messages)

    @property
    def audit(self) -> dict[str, Any]:
        return {'method': 'extractive_v1', 'scanned_messages': self.scanned_count,
                'window_limit': self.window_limit, 'max_chars': self.max_chars,
                'scanned_message_chars': self.scanned_chars,
                'used_chars': self.char_count, 'selected_user_message_ids': self.selected_user_ids,
                'omitted_user_messages': self.data['omitted_user_messages'],
                'older_history_not_scanned': self.data['older_history_not_scanned']}


def _same_topic(left: str, right: str) -> bool:
    from .tools import fallback_topic_key

    return (not left and not right) or bool(left and right and fallback_topic_key(left) == fallback_topic_key(right))


def build_thread_context(
    database: Database, thread_id: int | None, *, topic: str | None = None,
    before_message_id: int | None = None, max_messages: int = 100, max_chars: int = 10000,
    recent_limit: int = 6,
) -> ThreadContext:
    """Bound the historical message contents, not the new input/system/tool budgets.

    Full user quotes are selected or omitted; we never silently paraphrase/truncate
    a current view or correction. An oversize item remains in the canonical history.
    """
    max_messages = min(max(1, max_messages), 100)
    max_chars = min(max(1000, max_chars), 16000)
    data: dict[str, Any] = {'scope': topic, 'kind': 'extractive_working_context',
        'boundary': '用户原话不是永久事实；历史助手解释不在事实记忆中；历史来源编号需本轮工具重新读取后才可引用。',
        'user_corrections': [], 'current_view': None, 'past_user_quotes': [],
        'pending_question': None, 'latest_user_feedback': None, 'source_anchors': [], 'omitted_user_messages': 0,
        'oversize_message_ids': [], 'older_history_not_scanned': False}
    context = ThreadContext(data, [], [], 0, max_messages, max_chars)
    if thread_id is None:
        return context
    rows = [dict(row) for row in database.fetchall(
        "SELECT id,role,content,answer_json,created_at,reply_to_user_message_id FROM messages "
        "WHERE thread_id=? AND (? IS NULL OR id<?) AND role IN ('user','assistant') ORDER BY id DESC LIMIT ?",
        (thread_id, before_message_id, before_message_id, max_messages))]
    rows.reverse()
    context.scanned_count = len(rows)
    context.scanned_chars = sum(len(str(row['content'])) for row in rows)
    if rows:
        data['older_history_not_scanned'] = bool(database.fetchone(
            'SELECT 1 FROM messages WHERE thread_id=? AND id<? LIMIT 1', (thread_id, rows[0]['id'])))
    by_id = {int(row['id']): row for row in rows}
    matched: list[dict[str, Any]] = []
    answers: list[tuple[dict[str, Any], CognitiveAnswer]] = []
    previous_user = None
    selected_rows: set[int] = set()
    for row in rows:
        if row['role'] == 'user':
            previous_user = row
            continue
        try:
            answer = CognitiveAnswer.model_validate_json(row['answer_json'] or '{}')
        except ValueError:
            continue
        answer_topic = answer.dialogue.topic_query if answer.dialogue else answer.topic or ''
        if topic is not None and not _same_topic(topic, answer_topic):
            previous_user = None
            continue
        parent = by_id.get(row['reply_to_user_message_id']) if row['reply_to_user_message_id'] else previous_user
        if parent and int(parent['id']) not in selected_rows:
            parent['topic'] = answer_topic
            matched.append(parent)
            selected_rows.add(int(parent['id']))
        if not sources_available(database, answer_atom_ids(answer.model_dump())):
            # Source withdrawal applies to recalled assistant interpretations too;
            # it must not be bypassed by the recent-message window.
            previous_user = None
            continue
        matched.append(row)
        selected_rows.add(int(row['id']))
        answers.append((row, answer))
        previous_user = None
    if topic is None:
        # The planner may need the most recent incomplete user turn as context.
        matched = [row for row in rows if row['role'] == 'user' or int(row['id']) in selected_rows]

    def add(field: str, value: Any, *, append: bool = False, user_id: int | None = None) -> bool:
        old = list(data[field]) if append else data[field]
        data[field] = [*old, value] if append else value
        if context.char_count > max_chars - 100:
            data[field] = old
            return False
        if user_id and user_id not in context.selected_user_ids:
            context.selected_user_ids.append(user_id)
        return True

    # Explicit user corrections outrank historical model interpretations/current-view labels.
    revisions = database.fetchall(
        "SELECT v.id,v.message_id,v.user_revision,v.created_at FROM verdicts v "
        "JOIN messages m ON m.id=v.message_id WHERE m.thread_id=? AND v.status='active' "
        "AND v.user_revision IS NOT NULL AND (? IS NULL OR m.id<?) "
        "AND (? IS NULL OR v.topic_key=?) ORDER BY v.id DESC LIMIT 6",
        (thread_id, before_message_id, before_message_id, topic,
         topic_key_for(topic) if topic is not None else None))
    memory = MemoryService(database)
    revisions = [row for row in revisions if memory.verdict_available_for_recall(int(row['id']))]
    for revision in revisions[:1]:
        add('user_corrections', {'verdict_id': revision['id'], 'source_answer_id': revision['message_id'],
            'quote': revision['user_revision'], 'created_at': revision['created_at']}, append=True)

    statements: list[dict[str, Any]] = []
    seen_statements: set[int] = set()
    for _row, answer in reversed(answers):
        state = answer.dialogue
        if state is None or not state.current_statement:
            continue
        parent = by_id.get(state.current_statement_message_id) if state.current_statement_message_id is not None else None
        if parent is None and state.current_statement_message_id is not None:
            found = database.fetchone("SELECT id,role,content,created_at FROM messages WHERE id=? AND thread_id=? AND role='user'",
                                      (state.current_statement_message_id, thread_id))
            parent = dict(found) if found else None
        if parent is None:
            parent = next((item for item in matched if item['role'] == 'user' and
                           item['content'] == state.current_statement), None)
        if not parent or parent['content'] != state.current_statement or int(parent['id']) in seen_statements:
            continue
        seen_statements.add(int(parent['id']))
        statements.append({'message_id': parent['id'], 'quote': parent['content'],
                           'created_at': parent['created_at'], 'topic': state.topic_query})
    last_state = next((answer.dialogue for _, answer in reversed(answers) if answer.dialogue is not None), None)
    if statements and last_state and last_state.current_statement:
        latest = statements[0]
        if not add('current_view', latest, user_id=int(latest['message_id'])):
            data['oversize_message_ids'].append(latest['message_id'])
    for revision in revisions[1:]:
        add('user_corrections', {'verdict_id': revision['id'], 'source_answer_id': revision['message_id'],
            'quote': revision['user_revision'], 'created_at': revision['created_at']}, append=True)
    if answers:
        last_row, last_answer = answers[-1]
        feedback = database.fetchone(
            "SELECT id,verdict,created_at FROM verdicts WHERE message_id=? AND status='active' ORDER BY id DESC LIMIT 1",
            (last_row['id'],))
        if feedback:
            add('latest_user_feedback', {'verdict_id': feedback['id'], 'source_answer_id': last_row['id'],
                'choice': feedback['verdict'], 'created_at': feedback['created_at'],
                'boundary': '只针对这一份解释；defer 表示暂时放下，不代表确认，也不重复催问。'})
        if not feedback and last_answer.question_to_user and not (last_answer.dialogue and last_answer.dialogue.current_statement):
            add('pending_question', {'source_answer_id': last_row['id'], 'question': last_answer.question_to_user,
                                    'kind': 'previous_agent_question_not_a_fact'})

    # Preserve the recent exchange, then use remaining room for older key user quotes.
    recent: list[dict[str, str]] = []
    for row in reversed(matched[-recent_limit:]):
        if row['role'] == 'user' and int(row['id']) in context.selected_user_ids:
            continue
        content = str(row['content'])
        if row['role'] == 'assistant' and len(content) > 1200:
            content = content[:1200] + '（此条历史助手回复已截断，不能作为事实来源。）'
        message = {'role': str(row['role']), 'content': content}
        if context.char_count + len(content) > max_chars - 100:
            continue
        recent.append(message)
        context.recent_messages = list(reversed(recent))
        if row['role'] == 'user':
            context.selected_user_ids.append(int(row['id']))
    users = [row for row in matched if row['role'] == 'user']
    # Include the earliest surviving question, older explicit working-view updates, then other turns.
    prioritized = [*users[:1], *[by_id[item['message_id']] for item in statements[1:]
                                if item['message_id'] in by_id], *reversed(users)]
    for row in prioritized:
        if int(row['id']) in context.selected_user_ids:
            continue
        add('past_user_quotes', {'message_id': row['id'], 'quote': row['content'],
                                'created_at': row['created_at'], 'topic': row.get('topic', '')}, append=True, user_id=int(row['id']))
    seen_anchors: set[int] = set()
    for row, answer in reversed(answers):
        all_sources = [*answer.citations,
            *[source for position in (answer.early_position, answer.recent_position) if position for source in position.citations],
            *[source for item in [*answer.supporting_evidence, *answer.counter_evidence, *answer.interval_events]
              for source in item.citations]]
        for source in all_sources:
            if source.atom_id is None or source.atom_id in seen_anchors:
                continue
            seen_anchors.add(source.atom_id)
            add('source_anchors', {'atom_id': source.atom_id, 'recorded_at': source.recorded_at,
                'event_time': source.event_time, 'source_answer_id': row['id'],
                'status': 'historical_reference_requires_reread'}, append=True)
            if len(seen_anchors) >= 12:
                break
        if len(seen_anchors) >= 12:
            break
    data['omitted_user_messages'] = sum(int(row['id']) not in context.selected_user_ids for row in users)
    context.selected_user_ids = sorted(set(context.selected_user_ids))
    return context
