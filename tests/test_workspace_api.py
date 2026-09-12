"""Recoverable asks and user-visible memory/history, using synthetic local data only."""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from memory_garden.cognitive import VerdictService
from memory_garden.db import Database
from memory_garden.jobs import AskJobs
from memory_garden.web import create_app


def wait_for_job(client, request_id):
    end = time.monotonic() + 8
    while time.monotonic() < end:
        job = client.get('/api/ask/jobs/' + request_id).json()
        if job['status'] in {'completed', 'failed'}:
            return job
        time.sleep(.02)
    raise AssertionError('synthetic local request did not complete')


def test_pending_question_survives_navigation_and_duplicate_request(settings, monkeypatch):
    app = create_app(settings)
    started, finish = threading.Event(), threading.Event()
    original = app.state.harness.run
    calls = []
    def blocked(question, **kwargs):
        calls.append(question)
        started.set()
        assert finish.wait(5)
        return original(question, **kwargs)
    monkeypatch.setattr(app.state.harness, 'run', blocked)
    with TestClient(app) as client:
        body = {'question': '自主判断', 'request_id': 'synthetic-request-0001'}
        pending = client.post('/api/ask/jobs', json=body)
        assert pending.status_code == 202
        assert started.wait(2)
        job = pending.json()
        try:
            history = client.get('/api/threads/' + str(job['thread_id'])).json()
            assert [item['content'] for item in history] == ['自主判断']
            assert client.get('/api/threads').json()[0]['thread_id'] == job['thread_id']
            duplicate = client.post('/api/ask/jobs', json=body).json()
            assert duplicate['user_message_id'] == job['user_message_id']
            changed = client.post('/api/ask/jobs', json={**body, 'question': '另一个问题'})
            assert changed.status_code == 409
        finally:
            finish.set()
        result = wait_for_job(client, body['request_id'])
        assert result['status'] == 'completed', result
        history = client.get('/api/threads/' + str(job['thread_id'])).json()
        assert [item['role'] for item in history] == ['user', 'assistant']
        assert calls == ['自主判断']
        again = client.post('/api/ask/jobs', json=body).json()
        assert again['result']['message_id'] == result['result']['message_id']


def test_failed_job_keeps_user_message_and_restart_never_replays(settings, monkeypatch):
    app = create_app(settings)
    def failed(*args, **kwargs):
        raise RuntimeError('private service payload should not be exposed')
    monkeypatch.setattr(app.state.harness, 'run', failed)
    with TestClient(app) as client:
        data = client.post('/api/ask/jobs', json={'question': '自主判断', 'request_id': 'synthetic-failed-0001'}).json()
        failed_job = wait_for_job(client, data['request_id'])
        assert failed_job['status'] == 'failed' and 'private' not in failed_job['error']
        db = Database(settings.database_path)
        db.execute("UPDATE ask_jobs SET status='running' WHERE request_id=?", (data['request_id'],))
        manager = AskJobs(db, app.state.harness, threading.Lock())
        manager.recover_interrupted()
        assert manager.get(data['request_id'])['status'] == 'failed'
        assert db.fetchone('SELECT COUNT(*) AS n FROM messages')['n'] == 1
        db.close()


def test_history_activity_order_search_and_pagination(settings):
    app = create_app(settings)
    db = app.state.database
    for index in range(55):
        thread_id = db.execute('INSERT INTO threads(title,created_at) VALUES(?,?)', (f'合成-{index}', '2025-01-01'))
        db.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,'2025-01-01')", (thread_id, f'合成-{index}'))
    db.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(1,'user','旧主题的新补充','2026-09-12')")
    with TestClient(app) as client:
        first = client.get('/api/threads?limit=30').json()
        second = client.get('/api/threads?limit=30&offset=30').json()
        assert first[0]['thread_id'] == 1
        assert len(first) + len(second) == 55
        assert {row['thread_id'] for row in first}.isdisjoint(row['thread_id'] for row in second)
        assert len(client.get('/api/threads?q=新补充').json()) == 1
        assert client.get('/api/threads/999999').status_code == 404


def test_feedback_revision_and_memory_revoke_are_visible(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        answer = client.post('/api/ask', json={'question': '自主判断'}).json()
        saved = client.post('/api/verdict', json={'message_id': answer['message_id'], 'verdict': 'partly_accurate', 'user_revision': '我仍然重视其他人的意见。'}).json()
        memory = client.get('/api/memory?topic=自主判断').json()['items']
        assert len(memory) == 1 and memory[0]['statement'] == '我仍然重视其他人的意见。'
        assert memory[0]['source_verdict_id'] == saved['id']
        history = client.get('/api/threads/' + str(answer['thread_id'])).json()
        assert history[-1]['verdict']['user_revision'] == memory[0]['statement']
        assert history[-1]['verdict']['created_at']
        assert client.post('/api/memory/' + str(memory[0]['id']) + '/revoke', json={}).status_code == 200
        assert client.get('/api/memory?topic=自主判断').json()['items'] == []
        assert not VerdictService(app.state.database).for_topic('自主判断')
