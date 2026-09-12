"""Local connection -> downloadable preview -> explicit memory import, with no real account."""
import json

import httpx
from fastapi.testclient import TestClient

from memory_garden.local_chat import LocalQCEConnector
from memory_garden.web import create_app


def test_local_export_download_does_not_index_until_user_confirms(settings, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == '/api/friends':
            return httpx.Response(200, json={'success': True, 'data': {
                'friends': [{'uid': 'u_peer', 'uin': '12345', 'nick': '合成朋友'}], 'hasNext': False, 'totalCount': 1}})
        assert request.url.path == '/api/messages/fetch'
        assert json.loads(request.content)['peer'] == {'chatType': 1, 'peerUid': 'u_peer'}
        return httpx.Response(200, json={'success': True, 'data': {'hasNext': True, 'messages': [
            {'msgId': '1', 'senderUid': 'u_me', 'sendNickName': '我', 'msgTime': '1700000000',
             'elements': [{'elementType': 1, 'textElement': {'content': '合成原话：选择由我自己判断。'}}]},
            {'msgId': '2', 'senderUid': 'u_peer', 'sendNickName': '合成朋友', 'msgTime': '1700000060',
             'elements': [{'elementType': 1, 'textElement': {'content': '我只是提供意见。'}}]},
        ]}})

    monkeypatch.setattr('memory_garden.local_chat.LocalQCEConnector',
                        lambda base_url, token: LocalQCEConnector(base_url, token, transport=httpx.MockTransport(handler)))
    app = create_app(settings)
    with TestClient(app) as client:
        connection = {'base_url': 'http://127.0.0.1:40653', 'token': 'synthetic-token', 'kind': 'friend'}
        sessions = client.post('/api/import/local/sessions', json=connection)
        assert sessions.status_code == 200 and sessions.json()['items'][0]['id'] == 'u_peer'
        preview = client.post('/api/import/local/preview', json={**connection, 'peer_id': 'u_peer', 'count': 100})
        assert preview.status_code == 200, preview.text
        data = preview.json()
        assert data['connector']['has_more'] and data['counts']['messages'] == 2
        downloaded = client.get('/api/import/previews/' + data['preview_id'] + '/download')
        assert downloaded.status_code == 200 and 'attachment' in downloaded.headers['content-disposition']
        assert downloaded.json()[0]['text'] == '合成原话：选择由我自己判断。'
        assert 'synthetic-token' not in downloaded.text
        assert not client.get('/api/imports').json()['items']
        assert not (settings.database_path.parent / 'imports').exists()
        saved = client.post('/api/import/commit', json={'preview_id': data['preview_id'],
            'own_names': [downloaded.json()[0]['sender']], 'confirm_agent_access': True})
        assert saved.status_code == 200, saved.text
        assert saved.json()['own_messages'] == saved.json()['quoted_messages'] == 1
        assert len(calls) == 2  # Download/commit do not reread the client or fetch attachments.


def test_local_export_errors_do_not_contact_remote_hosts(settings):
    with TestClient(create_app(settings)) as client:
        blocked = client.post('/api/import/local/sessions', json={'base_url': 'https://example.com:443'})
        assert blocked.status_code == 400 and '本机' in blocked.json()['error']
        assert client.get('/api/import/previews/expired/download').status_code == 404
