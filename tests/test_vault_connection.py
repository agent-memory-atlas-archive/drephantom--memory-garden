"""Local Vault transitions preserve history, authority and per-request identity."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memory_garden.config import Settings
from memory_garden.db import Database, utc_now
from memory_garden.importer import VaultSyncService
from memory_garden.web import create_app


def _vault(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "note.md").write_text(text, encoding="utf-8")
    return path


def _setup(tmp_path: Path) -> tuple[Settings, Path]:
    initial = _vault(tmp_path, "demo", "仅供验证的合成演示记录。")
    other = _vault(tmp_path, "personal", "另一份合成笔记，包含独立思考。")
    return Settings(vault_path=initial, database_path=tmp_path / "demo.db", public_demo_mode=True,
                    retrieval_mode="bm25", reranker_backend="none"), other


def _headers(workspace: dict) -> dict[str, str]:
    return {"X-MG-Workspace-ID": workspace["workspace_id"]}


def _connect(client: TestClient, path: Path, workspace: dict) -> dict:
    response = client.post("/api/workspace/connect", json={"vault_path": str(path)}, headers=_headers(workspace))
    assert response.status_code == 200, response.text
    return response.json()


def test_connect_is_local_preserves_old_database_and_rejects_stale_ids(tmp_path: Path) -> None:
    base, personal = _setup(tmp_path)
    configured = replace(base, backend="deepseek", demo_use_model=True,
                         llm_base_url="https://synthetic.invalid", llm_api_key="synthetic-demo-secret",
                         llm_chat_model="synthetic-model")
    app = create_app(configured)
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        old_db = app.state.database
        old_thread = client.post("/api/threads").json()["thread_id"]
        old_db.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,?)",
                       (old_thread, "合成旧库的历史原话", utc_now()))
        preview = client.post("/api/import/preview", json={"filename": "chat.json", "content": json.dumps([
            {"sender": "合成用户", "text": "旧库导入片段。", "timestamp": "2026-01-01 00:00:00"}])}).json()
        assert client.post("/api/import/commit", json={"preview_id": preview["preview_id"],
            "own_names": ["合成用户"], "confirm_agent_access": True}).status_code == 200
        assert len(client.get("/api/imports").json()["items"]) == 1
        connected = _connect(client, personal, before)
        assert connected["workspace_id"] != before["workspace_id"]
        assert connected["public_demo_mode"] is False and connected["backend"] == "local"
        assert not connected["generation_connected"] and not connected["cloud_retrieval"]
        assert connected["counts"]["notes"] == 1 and connected["counts"]["threads"] == 0
        assert app.state.database.path != old_db.path
        assert client.get("/api/threads", headers=_headers(connected)).json() == []
        assert client.get("/api/imports", headers=_headers(connected)).json()["items"] == []
        new_thread = client.post("/api/threads", headers=_headers(connected)).json()["thread_id"]
        assert new_thread == old_thread  # Real collision of local numeric IDs.
        assert client.get(f"/api/threads/{new_thread}", headers=_headers(before)).status_code == 409
        assert client.get(f"/api/threads/{new_thread}").json()["code"] == "workspace_required"
        assert client.get(f"/api/threads/{new_thread}", headers=_headers(connected)).json() == []
        stale_preview = client.post("/api/import/commit", headers=_headers(connected), json={
            "preview_id": preview["preview_id"], "own_names": ["合成用户"], "confirm_agent_access": True})
        assert stale_preview.status_code == 400
        runtime = client.get("/api/settings", headers=_headers(connected)).json()
        assert runtime["retrieval_mode"] == "bm25" and runtime["embedding_backend"] == "local_hash"
        assert not runtime["api_key_set"] and not runtime["embedding_api_key_set"] and not runtime["reranker_api_key_set"]
        assert "synthetic-demo-secret" not in app.state.workspaces.registry_path.read_text(encoding="utf-8")
        restored = _connect(client, base.vault_path, connected)
        assert restored["workspace_id"] == before["workspace_id"]
        history = client.get(f"/api/threads/{old_thread}", headers=_headers(restored)).json()
        assert history[0]["content"] == "合成旧库的历史原话"
        assert old_db.fetchone("SELECT COUNT(*) n FROM messages")["n"] == 1


def test_restart_restores_selection_and_only_its_explicit_model_settings(tmp_path: Path) -> None:
    base, personal = _setup(tmp_path)
    app = create_app(base)
    with TestClient(app) as client:
        selected = _connect(client, personal, client.get("/api/workspace").json())
        thread = client.post("/api/threads", headers=_headers(selected)).json()["thread_id"]
        saved = client.post("/api/settings", headers=_headers(selected), json={
            "backend": "deepseek", "llm_base_url": "https://explicit-synthetic.invalid",
            "llm_chat_model": "explicit-model", "llm_api_key": "synthetic-target-secret"})
        assert saved.status_code == 200
    # Passing the same demo Settings must still restore the user's selected Vault.
    restarted = create_app(base)
    with TestClient(restarted) as client:
        current = client.get("/api/workspace").json()
        assert current["workspace_id"] == selected["workspace_id"]
        assert current["vault_path"] == str(personal.resolve()) and current["generation_connected"]
        assert client.get(f"/api/threads/{thread}", headers=_headers(current)).status_code == 200
        assert client.get("/api/threads").status_code == 409
        settings = client.get("/api/settings", headers=_headers(current)).json()
        assert settings["llm_chat_model"] == "explicit-model" and settings["api_key_set"]
        assert client.get("/api/health").status_code == 200  # Discovery endpoint.


@pytest.mark.parametrize("bad_path", ["relative/path", "D:notes", "\\\\server\\share", "//server/share"])
def test_connection_rejects_nonlocal_or_relative_paths_without_switching(tmp_path: Path, bad_path: str) -> None:
    base, _ = _setup(tmp_path)
    app = create_app(base)
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        response = client.post("/api/workspace/connect", json={"vault_path": bad_path})
        assert response.status_code == 400
        assert client.get("/api/workspace").json()["workspace_id"] == before["workspace_id"]
        assert client.get("/api/map").status_code == 200


def test_missing_vault_and_failed_sync_or_registry_write_keep_old_app(tmp_path: Path, monkeypatch) -> None:
    base, personal = _setup(tmp_path)
    app = create_app(base)
    manager = app.state.workspaces
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        forbidden = client.post("/api/workspace/connect", json={"vault_path": str(personal)},
                                headers={"Origin": "https://unrelated.example"})
        assert forbidden.status_code == 403
        assert client.post("/api/workspace/connect", json={"vault_path": str(tmp_path / "missing")}).status_code == 400
        original_factory = manager.factory
        monkeypatch.setattr(manager, "factory", lambda settings: (_ for _ in ()).throw(RuntimeError("synthetic private error")))
        failed = client.post("/api/workspace/connect", json={"vault_path": str(personal)})
        assert failed.status_code == 400 and "private" not in failed.text
        assert client.get("/api/workspace").json()["workspace_id"] == before["workspace_id"]
        assert client.post("/api/threads").status_code == 201
        monkeypatch.setattr(manager, "factory", original_factory)
        monkeypatch.setattr(manager, "_save", lambda *args: (_ for _ in ()).throw(OSError("synthetic full disk")))
        failed_save = client.post("/api/workspace/connect", json={"vault_path": str(personal)})
        assert failed_save.status_code == 400
        assert client.get("/api/workspace").json()["workspace_id"] == before["workspace_id"]
        assert client.post("/api/threads").status_code == 201


def test_busy_background_job_survives_rejected_switch(tmp_path: Path, monkeypatch) -> None:
    base, personal = _setup(tmp_path)
    app = create_app(base)
    started, finish = threading.Event(), threading.Event()
    original_run = app.state.harness.run

    def blocked(*args, **kwargs):
        started.set()
        assert finish.wait(8)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(app.state.harness, "run", blocked)
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        body = {"question": "合成记录", "request_id": "workspace-busy-synthetic-001"}
        job = client.post("/api/ask/jobs", json=body)
        assert job.status_code == 202 and started.wait(3)
        try:
            response = client.post("/api/workspace/connect", json={"vault_path": str(personal)})
            assert response.status_code == 409 and response.json()["code"] == "workspace_busy"
            assert client.get("/api/workspace").json()["workspace_id"] == before["workspace_id"]
        finally:
            finish.set()
        for _ in range(100):
            final = client.get("/api/ask/jobs/" + body["request_id"]).json()
            if final["status"] in {"completed", "failed"}:
                break
            time.sleep(.02)
        assert final["status"] == "completed"
        _connect(client, personal, before)


def test_inflight_request_is_bound_and_prevents_transition(tmp_path: Path) -> None:
    base, personal = _setup(tmp_path)
    app = create_app(base)
    entered, finish = threading.Event(), threading.Event()
    selected = app.state.workspaces.active

    @selected.app.get("/api/synthetic-slow")
    def slow():
        entered.set()
        assert finish.wait(8)
        return {"workspace_id": selected.workspace_id}

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client.get, "/api/synthetic-slow")
        assert entered.wait(3)
        try:
            failed = client.post("/api/workspace/connect", json={"vault_path": str(personal)})
            assert failed.status_code == 409 and failed.json()["code"] == "workspace_busy"
        finally:
            finish.set()
        response = result.result(3)
        assert response.json()["workspace_id"] == selected.workspace_id
        assert response.headers["x-mg-workspace-id"] == selected.workspace_id


def test_register_existing_preserves_known_history_without_guessing_database(tmp_path: Path) -> None:
    base, personal = _setup(tmp_path)
    known = replace(base, vault_path=personal, database_path=tmp_path / "known-history.db",
                    runtime_settings_path=tmp_path / "known-local-settings.json", public_demo_mode=False)
    existing = Database(known.database_path)
    existing.initialize()
    VaultSyncService(existing, personal).sync()
    thread = existing.execute("INSERT INTO threads(title,created_at) VALUES('合成旧库历史',?)", (utc_now(),))
    existing.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,'user',?,?)",
                     (thread, "需要原样保留的合成历史", utc_now()))
    app = create_app(base)
    changes = existing.connect().total_changes
    registered = app.state.workspaces.register_existing(known)
    assert existing.connect().total_changes == changes
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        assert len(before["workspaces"]) == 2
        assert client.get("/api/threads").json()["code"] == "workspace_required"
        chosen = _connect(client, personal, before)
        assert chosen["workspace_id"] == registered["workspace_id"]
        assert app.state.database.path == known.database_path
        assert client.get(f"/api/threads/{thread}", headers=_headers(chosen)).json()[0]["content"] == "需要原样保留的合成历史"
        duplicate = _connect(client, Path(str(personal) + "/."), chosen)
        assert duplicate["workspace_id"] == chosen["workspace_id"] and len(duplicate["workspaces"]) == 2
    unbound = Database(tmp_path / "unbound.db")
    unbound.initialize()
    with pytest.raises(ValueError, match="绑定"):
        app.state.workspaces.register_existing(replace(known, database_path=unbound.path))
    with pytest.raises(ValueError, match="另一个笔记库"):
        app.state.workspaces.register_existing(replace(known, vault_path=base.vault_path))


def test_download_query_identity_is_get_only_and_cannot_override_wrong_header(tmp_path: Path) -> None:
    base, personal = _setup(tmp_path)
    with TestClient(create_app(base)) as client:
        before = client.get("/api/workspace").json()
        current = _connect(client, personal, before)
        data = client.post("/api/import/preview", headers=_headers(current), json={"filename": "test.json", "content": json.dumps([
            {"sender": "合成人", "timestamp": "2026-01-01 12:00:00", "text": "合成下载文本"}])}).json()
        endpoint = "/api/import/previews/" + data["preview_id"] + "/download"
        assert client.get(endpoint).status_code == 409
        assert client.get(endpoint, params={"workspace": current["workspace_id"]}).status_code == 200
        assert client.get(endpoint, params={"workspace": before["workspace_id"]}).status_code == 409
        assert client.get(endpoint, params={"workspace": current["workspace_id"]}, headers=_headers(before)).status_code == 409
        assert client.post("/api/threads", params={"workspace": current["workspace_id"]}).status_code == 409
