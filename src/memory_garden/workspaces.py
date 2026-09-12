"""Small local workspace switchboard; each Vault owns a complete app and database.

Build and sync before publishing a selection. Requests bind to one app, and
switching cannot interrupt a request or background generation. The registry
contains local paths and identities, never model credentials or note content.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders, QueryParams
from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

from .config import Settings


class WorkspaceConflict(ValueError):
    def __init__(self, message: str, code: str, workspace_id: str):
        super().__init__(message)
        self.code, self.workspace_id = code, workspace_id

    def response(self) -> JSONResponse:
        return JSONResponse({"error": str(self), "code": self.code,
                             "workspace_id": self.workspace_id}, status_code=409)


def local_vault_path(value: str | Path) -> Path:
    """Accept explicit local directories, never cwd-relative or UNC locations."""
    raw = str(value).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1]
    path = Path(raw)
    if not raw or not path.is_absolute() or raw.startswith(("\\\\", "//")):
        raise ValueError("请输入本机笔记库文件夹的完整绝对路径，不支持相对路径或网络共享。")
    try:
        resolved = path.resolve(strict=True)
        valid = resolved.is_dir() and not str(resolved).startswith(("\\\\", "//"))
    except (OSError, RuntimeError):
        valid = False
        resolved = path
    if not valid:
        raise ValueError("笔记库文件夹不存在或不可读取，请检查本地路径。")
    return resolved


def _path_key(path: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(path.resolve())).encode("utf-8")).hexdigest()[:24]


def _local_settings(base: Settings, entry: dict[str, Any]) -> Settings:
    """A new Vault starts locally; only its own subsequently saved settings apply."""
    settings = replace(
        base, vault_path=Path(entry["vault_path"]), database_path=Path(entry["database_path"]),
        runtime_settings_path=Path(entry["runtime_settings_path"]),
        public_demo_mode=bool(entry.get("public_demo_mode", False)), demo_use_model=False,
        backend="local", retrieval_mode="bm25", embedding_backend="local_hash", reranker_backend="none",
        allow_cloud_embedding=False, allow_cloud_rerank=False,
        llm_base_url="", llm_api_key="", llm_chat_model="", llm_embedding_model="",
        embedding_base_url="", embedding_api_key="", reranker_base_url="", reranker_api_key="",
        reranker_model="", soul_path=None, _env={},
    )
    runtime = settings.runtime_settings_path
    if runtime and runtime.is_file():
        data = json.loads(runtime.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("此笔记库的模型设置格式无效。")
        identity_fields = {"vault_path", "database_path", "runtime_settings_path", "workspace_registry_path",
                           "public_demo_mode", "demo_use_model", "soul_path", "_env"}
        allowed = {field.name for field in fields(Settings)} - identity_fields
        settings = replace(settings, **{key: value for key, value in data.items() if key in allowed})
    return settings


@dataclass
class Workspace:
    key: str
    settings: Settings
    app: FastAPI
    workspace_id: str

    def entry(self) -> dict[str, Any]:
        return {"vault_path": str(self.settings.vault_path.resolve()),
                "database_path": str(self.settings.database_path.resolve()),
                "runtime_settings_path": str((self.settings.runtime_settings_path or
                                               self.settings.database_path.parent / "settings.json").resolve()),
                "public_demo_mode": self.settings.public_demo_mode, "workspace_id": self.workspace_id}


class WorkspaceManager:
    def __init__(self, settings: Settings, factory: Callable[[Settings], FastAPI]):
        self.base_settings, self.factory = settings, factory
        self.registry_path = (settings.workspace_registry_path or settings.database_path.with_name(
            settings.database_path.stem + "-workspaces.json")).resolve()
        self.storage_root = self.registry_path.parent / (self.registry_path.stem + ".d")
        self._lock = threading.RLock()
        self._switching = False
        self._requests = 0
        self.on_activate: Callable[[FastAPI], None] | None = None
        self.restore_notice: str | None = None
        # Validate the caller's initial DB/Vault binding even when restoring a selection.
        self.active = self._build(settings)
        self.initial_key = self.active.key
        initial_entry = self.active.entry()
        saved: dict[str, Any] = {}
        if self.registry_path.exists():
            saved = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("version") != 1 or not isinstance(saved.get("workspaces"), dict):
                raise ValueError("本地笔记库登记文件格式无效，请检查该文件后重启。")
        self.entries = dict(saved.get("workspaces", {}))
        self.entries[self.initial_key] = initial_entry
        self.require_header = bool(saved.get("require_workspace_header", False)) or len(self.entries) > 1
        selected = saved.get("active_key", self.initial_key)
        if selected != self.initial_key and selected in self.entries:
            try:
                self.active = self._build_entry(self.entries[selected])
            except (ValueError, OSError, RuntimeError, sqlite3.Error):
                self.restore_notice = "上次选择的笔记库暂时无法打开，已保留其登记并恢复启动笔记库。"
        self._save(self.entries, self.active.key, self.require_header)

    def _build(self, settings: Settings) -> Workspace:
        settings = replace(settings, vault_path=local_vault_path(settings.vault_path))
        app = self.factory(settings)
        row = app.state.database.fetchone("SELECT value FROM schema_meta WHERE key='workspace_id'")
        if row is None:
            raise ValueError("此数据库缺少工作区身份。")
        return Workspace(_path_key(settings.vault_path), settings, app, str(row["value"]))

    def _build_entry(self, entry: dict[str, Any]) -> Workspace:
        path = local_vault_path(entry["vault_path"])
        if _path_key(path) == self.initial_key:
            return self._build(self.base_settings)
        return self._build(_local_settings(self.base_settings, entry))

    def _save(self, entries: dict[str, Any], active_key: str, required: bool) -> None:
        payload = {"version": 1, "active_key": active_key,
                   "require_workspace_header": required, "workspaces": entries}
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def _check_identity(self, expected: str | None, *, discovery: bool = False) -> None:
        if expected is not None and expected != self.active.workspace_id:
            raise WorkspaceConflict("当前笔记库已改变，请刷新页面后继续。", "workspace_changed", self.active.workspace_id)
        if expected is None and self.require_header and not discovery:
            raise WorkspaceConflict("请刷新页面，确认当前笔记库后继续。", "workspace_required", self.active.workspace_id)

    def begin_request(self, expected: str | None, *, discovery: bool = False) -> Workspace:
        with self._lock:
            self._check_identity(expected, discovery=discovery)
            if self._switching:
                raise WorkspaceConflict("正在连接笔记库，请稍候再试。", "workspace_busy", self.active.workspace_id)
            self._requests += 1
            return self.active

    def end_request(self) -> None:
        with self._lock:
            self._requests -= 1

    def describe(self, expected: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._check_identity(expected, discovery=True)
            active = self.active
            database = active.app.state.database
            counts = {}
            queries = {
                "notes": "SELECT COUNT(*) n FROM sources WHERE is_present=1 AND searchable=1 AND authorship!='derived' AND source_kind='vault'",
                "sources": "SELECT COUNT(*) n FROM sources WHERE is_present=1 AND searchable=1 AND authorship!='derived'",
                "source_atoms": "SELECT COUNT(*) n FROM source_atoms a JOIN sources s ON s.id=a.source_id WHERE a.is_current=1 AND s.is_present=1 AND s.searchable=1 AND s.authorship!='derived' AND a.authorship!='derived'",
                "threads": "SELECT COUNT(*) n FROM threads",
                "chat_imports": "SELECT COUNT(*) n FROM sources WHERE source_kind='chat' AND is_present=1 AND searchable=1",
            }
            for name, query in queries.items():
                row = database.fetchone(query)
                counts[name] = int(row["n"]) if row else 0
            return {"workspace_id": active.workspace_id, "vault_path": str(active.settings.vault_path),
                    "public_demo_mode": active.settings.public_demo_mode, "backend": active.settings.backend,
                    "cloud_retrieval": active.app.state.harness.retriever.vector.is_cloud or active.settings.reranker_backend == 'api',
                    "generation_connected": active.app.state.harness.provider is not None,
                    "counts": counts, "switching": self._switching, "restore_notice": self.restore_notice,
                    "workspaces": [{"workspace_id": entry["workspace_id"], "vault_path": entry["vault_path"],
                                    "public_demo_mode": entry["public_demo_mode"], "active": key == active.key}
                                   for key, entry in self.entries.items()]}

    def register_existing(self, settings: Settings) -> dict[str, Any]:
        """Internal explicit registration, never an HTTP database-path parameter.

        The caller must already have verified/migrated the known existing DB.
        Validate its binding read-only; do not infer a Vault for an unbound DB.
        """
        path = local_vault_path(settings.vault_path)
        db_path = settings.database_path.resolve(strict=True)
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM schema_meta WHERE key IN ('vault_root','workspace_id')"))
        if not metadata.get("workspace_id") or not metadata.get("vault_root"):
            raise ValueError("已有数据库尚未确认笔记库绑定，请先核对并迁移该已知数据库。")
        if _path_key(Path(metadata["vault_root"])) != _path_key(path):
            raise ValueError("已有数据库属于另一个笔记库，不能登记到此路径。")
        entry = {"vault_path": str(path), "database_path": str(db_path),
                 "runtime_settings_path": str((settings.runtime_settings_path or
                                                self.storage_root / _path_key(path) / "settings.json").resolve()),
                 "public_demo_mode": settings.public_demo_mode, "workspace_id": metadata["workspace_id"]}
        with self._lock:
            if self._switching:
                raise WorkspaceConflict("正在连接笔记库，请稍后登记。", "workspace_busy", self.active.workspace_id)
            key = _path_key(path)
            if key == self.active.key and db_path != self.active.settings.database_path.resolve():
                raise ValueError("当前已打开同路径的另一份数据库，请先切回其他笔记库再登记。")
            entries = {**self.entries, key: entry}
            required = self.require_header or len(entries) > 1
            self._save(entries, self.active.key, required)
            self.entries, self.require_header = entries, required
        return {"workspace_id": entry["workspace_id"], "vault_path": entry["vault_path"]}

    def connect(self, value: str, expected: str | None = None) -> dict[str, Any]:
        path = local_vault_path(value)
        with self._lock:
            self._check_identity(expected)
            current = self.active
            if self._switching or self._requests or not current.app.state.operation_lock.acquire(blocking=False):
                raise WorkspaceConflict("还有一项操作正在进行，完成后再连接笔记库。", "workspace_busy", current.workspace_id)
            self._switching = True
        try:
            key = _path_key(path)
            folder = self.storage_root / key
            entry = self.entries.get(key, {"vault_path": str(path), "database_path": str(folder / "memory.db"),
                                          "runtime_settings_path": str(folder / "settings.json"), "public_demo_mode": False})
            candidate = self._build_entry(entry)
            entries = {**self.entries, key: candidate.entry()}
            required = self.require_header or len(entries) > 1
            self._save(entries, key, required)
            with self._lock:
                self.entries, self.require_header, self.active = entries, required, candidate
                self.restore_notice = None
                if self.on_activate:
                    self.on_activate(candidate.app)
        finally:
            with self._lock:
                self._switching = False
                current.app.state.operation_lock.release()
        return self.describe()


class WorkspaceDispatcher:
    def __init__(self, manager: WorkspaceManager):
        self.manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = Headers(scope=scope)
        path = scope.get("path", "")
        discovery = path == "/api/health" or not path.startswith("/api/")
        expected = headers.get("x-mg-workspace-id")
        # Native download links cannot attach custom headers. This narrow GET
        # fallback carries exactly the same identity check, never a write bypass.
        if (expected is None and scope.get("method") == "GET"
                and re.fullmatch(r"/api/import/previews/[^/]+/download", path)):
            expected = QueryParams(scope.get("query_string", b"")).get("workspace")
        try:
            selected = self.manager.begin_request(expected, discovery=discovery)
        except WorkspaceConflict as exc:
            await exc.response()(scope, receive, send)
            return

        async def bound_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-MG-Workspace-ID"] = selected.workspace_id
            await send(message)

        try:
            await selected.app(scope, receive, bound_send)
        finally:
            self.manager.end_request()
