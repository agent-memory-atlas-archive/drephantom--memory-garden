"""Local-only chat preview, consent and retrieval lifecycle integration."""
from fastapi.testclient import TestClient

from memory_garden.retrieval import RetrievalQuery
from memory_garden.web import create_app


def test_import_preview_consent_retrieval_and_withdrawal(settings):
    app = create_app(settings)
    db = app.state.database
    content = '2025-01-01 12:00:00 我\n合成星河主题：我开始自己判断。\n2025-01-02 13:00:00 朋友\n合成星河主题：我的意见仅供参考。'
    with TestClient(app) as client:
        page = client.get('/import')
        assert page.status_code == 200
        assert "script-src 'self'" in page.headers['content-security-policy']
        preview = client.post('/api/import/preview', json={
            'filename': 'synthetic.txt', 'content': content}).json()
        assert set(preview['participants']) == {'我', '朋友'}
        assert db.fetchone("SELECT COUNT(*) n FROM sources WHERE source_kind='chat'")['n'] == 0
        body = {'preview_id': preview['preview_id'], 'own_names': ['我']}
        assert client.post('/api/import/commit', json=body).status_code == 400
        assert not (settings.database_path.parent / 'imports').exists()
        body['confirm_agent_access'] = True
        committed = client.post('/api/import/commit', json=body)
        assert committed.status_code == 200, committed.text
        result = committed.json()
        assert result['own_messages'] == result['quoted_messages'] == 1
        assert client.post('/api/import/commit', json=body).json()['duplicate']
        listing = client.get('/api/imports').json()['items']
        assert len(listing) == 1 and listing[0]['is_present']
        assert listing[0]['indexed_messages'] == 2
        hits = app.state.harness.retriever.search(RetrievalQuery(text='合成星河主题', limit=10))
        chat_hits = [hit for hit in hits if hit.fields['source_kind'] == 'chat']
        assert {hit.fields['sender'] for hit in chat_hits} == {'我', '朋友'}
        assert {hit.fields['authorship'] for hit in chat_hits} == {'user', 'quoted'}
        stopped = client.post('/api/imports/' + result['source_uid'] + '/deactivate')
        assert stopped.status_code == 200 and not stopped.json()['searchable']
        assert not client.get('/api/imports').json()['items'][0]['is_present']
        assert not [hit for hit in app.state.harness.retriever.search(RetrievalQuery(text='合成星河主题', limit=10))
                    if hit.fields['source_kind'] == 'chat']


def test_import_rejects_cross_origin_and_bad_preview(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post('/api/import/preview', json={'filename': 'x.txt', 'content': 'x'},
                           headers={'Origin': 'https://unrelated.example'}).status_code == 403
        assert client.post('/api/import/preview', json={'filename': 'x.db', 'content': 'x'}).status_code == 400
        assert client.post('/api/import/commit', json={
            'preview_id': 'expired', 'confirm_agent_access': True}).status_code == 400
