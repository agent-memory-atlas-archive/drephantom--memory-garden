"""Durable, idempotent local requests; navigating away does not cancel generation."""
from __future__ import annotations

import json
import threading
from typing import Any

from .db import Database, utc_now


def result_payload(result: Any) -> dict[str, Any]:
    return {
        'reply': result.reply, 'answer': result.answer.model_dump(),
        'answer_type': result.answer.answer_type,
        'unknowns': [u for u in result.answer.unknowns if '现有来源不能证明' not in u and '尚未经用户确认' not in u],
        'verdict_worthy': result.answer.answer_type in {'traced_change', 'no_clear_change'},
        'message_id': result.message_id, 'thread_id': result.thread_id,
        'backend': result.backend, 'steps': result.steps, 'tool_calls': result.tool_calls,
        'stop_reason': result.stop_reason,
    }


class JobConflict(ValueError):
    pass


class AskJobs:
    def __init__(self, database: Database, harness: Any, operation_lock: Any):
        self.database, self.harness, self.operation_lock = database, harness, operation_lock
        self._submit_lock = threading.Lock()

    def recover_interrupted(self) -> None:
        # No automatic replay after a process restart: it could duplicate a paid
        # model call. If the reply already committed, recover it from the audit.
        for row in self.database.fetchall("SELECT * FROM ask_jobs WHERE status IN ('queued','running')"):
            answer = self.database.fetchone(
                "SELECT m.*,r.backend,r.steps,r.tool_calls,r.stop_reason FROM messages m "
                "LEFT JOIN agent_runs r ON r.message_id=m.id WHERE m.thread_id=? AND m.role='assistant' AND m.id>? "
                "AND (m.reply_to_user_message_id=? OR (m.reply_to_user_message_id IS NULL AND NOT EXISTS("
                "SELECT 1 FROM messages u WHERE u.thread_id=m.thread_id AND u.role='user' AND u.id>? AND u.id<m.id))) "
                'ORDER BY m.id LIMIT 1', (row['thread_id'], row['user_message_id'], row['user_message_id'], row['user_message_id']),
            )
            if answer and answer['answer_json']:
                data = json.loads(answer['answer_json'])
                payload = {'reply': answer['content'], 'answer': data, 'answer_type': data['answer_type'],
                           'unknowns': data.get('unknowns', []), 'verdict_worthy': data['answer_type'] in {'traced_change', 'no_clear_change'},
                           'message_id': answer['id'], 'thread_id': row['thread_id'], 'backend': answer['backend'],
                           'steps': answer['steps'], 'tool_calls': answer['tool_calls'], 'stop_reason': answer['stop_reason']}
                self.database.execute("UPDATE ask_jobs SET status='completed',result_json=?,updated_at=? WHERE request_id=?",
                                      (json.dumps(payload, ensure_ascii=False), utc_now(), row['request_id']))
            else:
                self.database.execute(
                    "UPDATE ask_jobs SET status='failed',error=?,updated_at=? WHERE request_id=?",
                    ('服务重启中断了这次回应。你的消息已保留，可以重新发起回看。', utc_now(), row['request_id']),
                )

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self.database.fetchone('SELECT * FROM ask_jobs WHERE request_id=?', (request_id,))
        if row is None:
            return None
        data = dict(row)
        data.pop('question')
        encoded = data.pop('result_json')
        if encoded:
            data['result'] = json.loads(encoded)
        return data

    def list_for_thread(self, thread_id: int) -> list[dict[str, Any]]:
        rows = self.database.fetchall('SELECT request_id FROM ask_jobs WHERE thread_id=? ORDER BY created_at DESC LIMIT 10', (thread_id,))
        return [item for row in rows if (item := self.get(row['request_id'])) is not None]

    def submit(self, request_id: str, question: str, thread_id: int | None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError('请先写下一句话。')
        with self._submit_lock:
            existing = self.database.fetchone('SELECT * FROM ask_jobs WHERE request_id=?', (request_id,))
            if existing:
                if question != existing['question'] or (thread_id is not None and thread_id != existing['thread_id']):
                    raise JobConflict('这个请求编号已用于另一个问题，请重新发送。')
                return self.get(request_id) or {}
            if thread_id is not None and not self.database.fetchone('SELECT 1 FROM threads WHERE id=?', (thread_id,)):
                raise KeyError('这段对话不存在。')
            if not self.operation_lock.acquire(blocking=False):
                raise JobConflict('还有一项回看正在进行。你可以离开页面，完成后从历史继续。')
            try:
                now = utc_now()
                with self.database.transaction() as connection:
                    if thread_id is None:
                        cursor = connection.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', (question[:60], now))
                        thread_id = int(cursor.lastrowid or 0)
                    cursor = connection.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,?)", (thread_id, question, now))
                    user_message_id = int(cursor.lastrowid or 0)
                    connection.execute('INSERT INTO ask_jobs(request_id,thread_id,user_message_id,question,status,created_at,updated_at) '
                                       "VALUES(?,?,?,?,'queued',?,?)", (request_id, thread_id, user_message_id, question, now, now))
                worker = threading.Thread(target=self._run, args=(request_id, question, thread_id, user_message_id),
                                          name='memory-garden-response', daemon=True)
                worker.start()
            except Exception:
                self.operation_lock.release()
                self.database.execute("UPDATE ask_jobs SET status='failed',error=?,updated_at=? WHERE request_id=?",
                                      ('这次回应未能启动，请重新发起回看。', utc_now(), request_id))
                raise
        return self.get(request_id) or {}

    def _run(self, request_id: str, question: str, thread_id: int, user_message_id: int) -> None:
        try:
            self.database.execute("UPDATE ask_jobs SET status='running',updated_at=? WHERE request_id=?", (utc_now(), request_id))
            result = self.harness.run(question, thread_id=thread_id, user_message_id=user_message_id)
            self.database.execute("UPDATE ask_jobs SET status='completed',result_json=?,updated_at=? WHERE request_id=?",
                                  (json.dumps(result_payload(result), ensure_ascii=False), utc_now(), request_id))
        except Exception:
            # Exceptions can contain service payloads or secrets; expose only a
            # stable user-facing error. Original user message remains committed.
            self.database.execute("UPDATE ask_jobs SET status='failed',error=?,updated_at=? WHERE request_id=?",
                                  ('这次回看没有完成。你的消息已保留，请检查连接与检索设置后重试。', utc_now(), request_id))
        finally:
            self.database.close()
            self.operation_lock.release()
