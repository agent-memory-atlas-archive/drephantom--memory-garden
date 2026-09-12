"""Browser request boundaries and validation errors use synthetic secrets only."""
from __future__ import annotations

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from memory_garden.web import create_app


@pytest.mark.parametrize('body', [
    {'token': 'synthetic-private-token-' * 200},
    [{'token': 'synthetic-private-token'}],
    {'token': {'raw': 'synthetic-private-token'}},
])
def test_invalid_local_connection_does_not_echo_credentials(settings, monkeypatch, body):
    def no_connection(*args, **kwargs):
        raise AssertionError('Invalid credentials must not reach the local client')

    monkeypatch.setattr('memory_garden.local_chat.LocalQCEConnector', no_connection)
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/import/local/sessions', json=body)
        assert response.status_code == 422
        assert 'synthetic-private-token' not in response.text
        assert response.headers['cache-control'] == 'no-store'
        assert response.json()['detail']
        assert all(set(error) == {'type', 'loc', 'msg'} for error in response.json()['detail'])


def test_custom_validation_message_cannot_echo_private_content(settings):
    app = create_app(settings)

    @app.get('/synthetic-validation-error')
    def synthetic_error():
        raise RequestValidationError([{
            'type': 'value_error', 'loc': ('body', 'token'),
            'msg': 'Rejected synthetic-private-token', 'input': 'synthetic-private-token',
            'ctx': {'error': 'synthetic-private-token'},
        }])

    with TestClient(app) as client:
        response = client.get('/synthetic-validation-error')
        assert response.status_code == 422
        assert 'synthetic-private-token' not in response.text


@pytest.mark.parametrize('headers', [
    {'Origin': 'http://['},
    {'Origin': 'null'},
    {'Origin': 'https://unrelated.example'},
    {'Sec-Fetch-Site': 'cross-site'},
])
def test_local_connection_rejects_cross_site_and_malformed_origins(settings, monkeypatch, headers):
    def no_connection(*args, **kwargs):
        raise AssertionError('Rejected browser requests must not contact the local client')

    monkeypatch.setattr('memory_garden.local_chat.LocalQCEConnector', no_connection)
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/import/local/sessions', json={}, headers=headers)
        assert response.status_code == 403


def test_same_origin_preview_download_has_private_attachment_headers(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/import/preview', json={
            'filename': 'synthetic.json',
            'content': '[{"sender":"我","timestamp":"2025-01-01","text":"<script>synthetic</script>"}]',
        }, headers={'Origin': 'http://testserver', 'Sec-Fetch-Site': 'same-origin'})
        assert response.status_code == 200
        preview_id = response.json()['preview_id']
        downloaded = client.get(f'/api/import/previews/{preview_id}/download')
        assert downloaded.status_code == 200
        assert downloaded.headers['cache-control'] == 'no-store'
        assert downloaded.headers['x-content-type-options'] == 'nosniff'
        assert downloaded.headers['content-type'].startswith('application/json')
        assert downloaded.headers['content-disposition'].startswith('attachment;')
