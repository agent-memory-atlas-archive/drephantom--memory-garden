"""极简本地 Web UI：一个对话页 + 发现评审页。FastAPI 单文件，无前端构建链。"""
from __future__ import annotations

import contextlib
import html
import json
import sqlite3
import threading
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .agent import AgentHarness
from .chat_import import ChatImportService
from .cli import build_database
from .cognitive import DiscoveryService, VerdictService
from .config import Settings
from .db import Database, utc_now
from .graph import GLOBAL_MAX_NODES
from .jobs import AskJobs, JobConflict, result_payload
from .memory import MemoryService
from .retrieval import build_retriever


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    thread_id: int | None = Field(default=None, gt=0)


class JobBody(AskBody):
    request_id: str = Field(min_length=16, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')


class VerdictBody(BaseModel):
    message_id: int
    verdict: str
    user_revision: str = Field(default='', max_length=3000)
    missing_event: str = Field(default='', max_length=3000)


class ReviewBody(BaseModel):
    discovery_id: int
    verdict: str
    user_revision: str = ""
    missing_event: str = ""


class RuntimeSettingsBody(BaseModel):
    assistant_name: str = ""
    llm_base_url: str = ""
    llm_chat_model: str = ""
    llm_embedding_model: str = ""
    llm_api_key: str = Field(default="", repr=False)
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = Field(default="", repr=False)
    vault_path: str = ""
    backend: str = ""
    embedding_backend: str = ""
    retrieval_mode: str = ""
    reranker_backend: str = ""
    reranker_provider: str = ""
    reranker_base_url: str = ""
    reranker_api_key: str = Field(default="", repr=False)
    reranker_model: str = ""
    rerank_candidate_limit: int | None = None
    reranker_fusion: str = ""
    allow_cloud_embedding: bool | None = None
    allow_cloud_rerank: bool | None = None


class EmbeddingBuildBody(BaseModel):
    confirm_cloud_send: bool = False


class DismissBody(BaseModel):
    discovery_id: int = Field(gt=0)


class ImportPreviewBody(BaseModel):
    filename: str = Field(min_length=1, max_length=260)
    content: str = Field(min_length=1, max_length=2 * 1024 * 1024)
    format: str = Field(default='auto', max_length=30)
    own_names: list[str] = Field(default_factory=list, max_length=100)


class ImportCommitBody(BaseModel):
    preview_id: str = Field(min_length=1, max_length=80)
    own_names: list[str] = Field(default_factory=list, max_length=100)
    confirm_agent_access: bool = False


class LocalChatBody(BaseModel):
    base_url: str = Field(default='http://127.0.0.1:40653', max_length=200)
    token: str = Field(default='', max_length=4096, repr=False)
    kind: Literal['friend', 'group'] = 'friend'


class LocalSessionsBody(LocalChatBody):
    page: int = Field(default=1, ge=1)


class LocalPreviewBody(LocalChatBody):
    peer_id: str = Field(min_length=1, max_length=200)
    count: int = Field(default=200, ge=1, le=500)


class ConnectVaultBody(BaseModel):
    vault_path: str = Field(min_length=1, max_length=4096)


# 页面与 Python 分离：用户可直接改 page.html 调整样式与文案（改完刷新即生效）
PAGE_FILE = Path(__file__).resolve().parent / "page.html"



def load_page(settings: Settings) -> str:
    if PAGE_FILE.exists():
        name = html.escape(settings.assistant_name, quote=True)
        return PAGE_FILE.read_text(encoding="utf-8").replace("{{NAME}}", name)
    return "<html><body><p>page.html 缺失，请重新拉取仓库。</p></body></html>"



def _create_workspace_app(settings: Settings) -> FastAPI:
    database: Database = build_database(settings)
    retriever = build_retriever(database, settings)
    # 启动时同步当前版本；先校验 Vault 绑定，云端向量不在启动时发送。
    from .importer import VaultSyncService

    VaultSyncService(database, settings.vault_path).sync()
    if "vector" in retriever.routes and not retriever.vector.is_cloud:
        retriever.vector.ensure_vectors()
    provider = None
    if settings.backend != "local" and settings.llm_ready:
        from .agent import OpenAIProvider
        from .llm import OpenAICompatibleClient

        provider = OpenAIProvider(OpenAICompatibleClient(settings))
    harness = AgentHarness(database, retriever, settings, provider=provider)
    operation_lock = threading.Lock()
    jobs = AskJobs(database, harness, operation_lock)
    jobs.recover_interrupted()
    chat_import = ChatImportService(database, settings)

    app = FastAPI(title="Memory Garden", version="2.0.0")
    app.state.database, app.state.harness, app.state.jobs = database, harness, jobs
    app.state.settings, app.state.operation_lock = settings, operation_lock
    app.mount('/static', StaticFiles(directory=PAGE_FILE.parent / 'static'), name='static')
    _configure_security(app)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return load_page(settings)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page() -> str:
        return (PAGE_FILE.parent / 'settings.html').read_text(encoding='utf-8')

    @app.get("/history", response_class=HTMLResponse)
    def history_page() -> str:
        return (PAGE_FILE.parent / 'history.html').read_text(encoding='utf-8')

    @app.get('/import', response_class=HTMLResponse)
    def import_page() -> str:
        return (PAGE_FILE.parent / 'import.html').read_text(encoding='utf-8')

    @app.post('/api/import/preview')
    def preview_import(body: ImportPreviewBody) -> JSONResponse:
        try:
            return JSONResponse(chat_import.preview(**body.model_dump()))
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)

    @app.post('/api/import/commit')
    def commit_import(body: ImportCommitBody) -> JSONResponse:
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '正在处理上一项操作，请稍后再导入。'}, status_code=409)
        try:
            return JSONResponse(chat_import.commit(**body.model_dump()))
        except (ValueError, KeyError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        except OSError:
            return JSONResponse({'error': '无法保存导入副本，请检查本地数据目录的空间和权限。'}, status_code=500)
        finally:
            operation_lock.release()

    @app.post('/api/import/local/sessions')
    def local_chat_sessions(body: LocalSessionsBody) -> JSONResponse:
        from .local_chat import LocalQCEConnector

        try:
            connector = LocalQCEConnector(body.base_url, body.token)
            return JSONResponse(connector.sessions(kind=body.kind, page=body.page))
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)

    @app.post('/api/import/local/preview')
    def local_chat_preview(body: LocalPreviewBody) -> JSONResponse:
        from .local_chat import LocalQCEConnector

        try:
            exported = LocalQCEConnector(body.base_url, body.token).export(
                kind=body.kind, peer_id=body.peer_id, count=body.count)
            result = chat_import.preview(filename=exported['filename'], content=exported['content'])
            result['warnings'] = [*exported['warnings'], *result['warnings']]
            result['connector'] = exported['scope']
            return JSONResponse(result)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)

    @app.get('/api/import/previews/{preview_id}/download')
    def download_chat_preview(preview_id: str) -> Response:
        try:
            filename, content = chat_import.export_preview(preview_id)
        except KeyError:
            return JSONResponse({'error': '预览已过期，请重新读取所选会话。'}, status_code=404)
        return Response(content, media_type='application/json' if filename.endswith('.json') else 'text/plain',
                        headers={'Content-Disposition': "attachment; filename*=UTF-8''" + quote(filename, safe='')})

    @app.get('/api/imports')
    def list_imports(limit: int = Query(default=30, ge=1, le=100),
                     offset: int = Query(default=0, ge=0)) -> JSONResponse:
        rows = database.fetchall(
            "SELECT uid,title,rel_path,is_present,searchable,frontmatter_json FROM sources "
            "WHERE source_kind='chat' ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        items = []
        for row in rows:
            metadata = json.loads(row['frontmatter_json'])
            items.append({'source_uid': row['uid'], 'title': row['title'], 'path': row['rel_path'],
                          'is_present': bool(row['is_present'] and row['searchable']),
                          'created_at': metadata.get('imported_at'),
                          'indexed_messages': metadata.get('message_count', 0)})
        return JSONResponse({'items': items})

    @app.post('/api/imports/{source_uid}/deactivate')
    def deactivate_import(source_uid: str) -> JSONResponse:
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '正在处理上一项操作，请稍后再停止检索。'}, status_code=409)
        try:
            return JSONResponse(chat_import.deactivate(source_uid))
        except KeyError:
            return JSONResponse({'error': '这份聊天导入不存在。'}, status_code=404)
        finally:
            operation_lock.release()

    RUNTIME_SETTINGS_FILE = settings.runtime_settings_path or Path(settings.database_path).parent / "settings.json"

    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        return JSONResponse(
            {
                "assistant_name": settings.assistant_name,
                "llm_base_url": settings.llm_base_url,
                "llm_chat_model": settings.llm_chat_model,
                "llm_embedding_model": settings.llm_embedding_model,
                "embedding_provider": settings.embedding_provider,
                "embedding_base_url": settings.embedding_base_url,
                "vault_path": str(settings.vault_path),
                "backend": settings.backend,
                "embedding_backend": settings.embedding_backend,
                "retrieval_mode": settings.retrieval_mode,
                "reranker_backend": settings.reranker_backend,
                "reranker_provider": settings.reranker_provider,
                "reranker_base_url": settings.reranker_base_url,
                "reranker_model": settings.reranker_model,
                "rerank_candidate_limit": settings.rerank_candidate_limit,
                "reranker_fusion": settings.reranker_fusion,
                "allow_cloud_embedding": settings.allow_cloud_embedding,
                "allow_cloud_rerank": settings.allow_cloud_rerank,
                "api_key_set": bool(settings.llm_api_key),
                "embedding_api_key_set": bool(settings.embedding_api_key),
                "reranker_api_key_set": bool(settings.reranker_api_key),
                "public_demo_mode": settings.public_demo_mode,
                "demo_use_model": settings.demo_use_model,
            }
        )

    @app.post("/api/settings")
    def save_settings(body: RuntimeSettingsBody) -> JSONResponse:
        RUNTIME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        current: dict = {}
        if RUNTIME_SETTINGS_FILE.exists():
            try:
                current = json.loads(RUNTIME_SETTINGS_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        allowed_values = {
            'backend': {'local', 'deepseek'},
            "embedding_backend": {"local_hash", "mock", "api"},
            "retrieval_mode": {"bm25", "hash_vector", "embedding", "hybrid"},
            "reranker_backend": {"none", "local_heuristic", "api"},
            "reranker_fusion": {"replace", "rank_fusion"},
        }
        for field, choices in allowed_values.items():
            value = getattr(body, field).strip().lower()
            if value and value not in choices:
                return JSONResponse({"error": f"{field} 值无效"}, status_code=400)
        if body.vault_path.strip():
            candidate_vault = Path(body.vault_path.strip())
            if not candidate_vault.is_dir():
                return JSONResponse({'error': '笔记库路径不存在，请检查后再保存。'}, status_code=400)
            if candidate_vault.resolve() != settings.vault_path.resolve():
                return JSONResponse({'error': '请使用“连接笔记库”切换路径，每个笔记库会保留独立历史。'}, status_code=400)
        effective_embedding = (
            body.embedding_backend.strip().lower()
            or str(current.get("embedding_backend") or settings.embedding_backend)
        )
        effective_retrieval = (
            body.retrieval_mode.strip().lower()
            or str(current.get("retrieval_mode") or settings.retrieval_mode)
        )
        effective_reranker = (
            body.reranker_backend.strip().lower()
            or str(current.get("reranker_backend") or settings.reranker_backend)
        )
        effective_cloud_consent = (
            body.allow_cloud_embedding
            if body.allow_cloud_embedding is not None
            else bool(current.get("allow_cloud_embedding", settings.allow_cloud_embedding))
        )
        effective_rerank_consent = (
            body.allow_cloud_rerank
            if body.allow_cloud_rerank is not None
            else bool(current.get("allow_cloud_rerank", settings.allow_cloud_rerank))
        )
        if effective_retrieval == "embedding" and effective_embedding == "local_hash":
            return JSONResponse(
                {"error": "embedding 模式需选择 mock 或 api；字符哈希请选 hash_vector"},
                status_code=400,
            )
        if (
            effective_retrieval in {"embedding", "hybrid"}
            and effective_embedding == "api"
            and not effective_cloud_consent
        ):
            return JSONResponse(
                {"error": "选择 API Embedding 时必须明确勾选允许云端发送"},
                status_code=400,
            )
        if effective_reranker == "api" and not effective_rerank_consent:
            return JSONResponse(
                {"error": "选择 API Rerank 时必须明确勾选允许发送查询和候选文本"},
                status_code=400,
            )
        for field in (
            "assistant_name", "llm_base_url", "llm_chat_model", "llm_embedding_model",
            "embedding_provider", "embedding_base_url",
            "vault_path", "backend", "embedding_backend", "retrieval_mode",
            "reranker_backend", "reranker_provider", "reranker_base_url",
            "reranker_model", "reranker_fusion",
        ):
            value = getattr(body, field).strip()
            if value:
                current[field] = value
        if body.allow_cloud_embedding is not None:
            current["allow_cloud_embedding"] = body.allow_cloud_embedding
        if body.allow_cloud_rerank is not None:
            current["allow_cloud_rerank"] = body.allow_cloud_rerank
        if body.rerank_candidate_limit is not None:
            if not 1 <= body.rerank_candidate_limit <= 200:
                return JSONResponse(
                    {"error": "rerank_candidate_limit 必须在 1 到 200 之间"},
                    status_code=400,
                )
            current["rerank_candidate_limit"] = body.rerank_candidate_limit
        if body.llm_api_key.strip():
            current["llm_api_key"] = body.llm_api_key.strip()
        if body.embedding_api_key.strip():
            current["embedding_api_key"] = body.embedding_api_key.strip()
        if body.reranker_api_key.strip():
            current["reranker_api_key"] = body.reranker_api_key.strip()
        setting_fields = {field.name for field in fields(Settings)}
        effective = {key: value for key, value in current.items() if key in setting_fields}
        if 'vault_path' in effective:
            effective['vault_path'] = Path(effective['vault_path'])
        candidate = replace(settings, **effective)
        if candidate.backend != 'local' and not candidate.llm_ready:
            return JSONResponse({'error': '连接模型需要填写服务地址、模型名和 API Key。'}, status_code=400)
        try:
            # 只检查配置，不发起请求；不能保存会导致下次启动失败的组合。
            build_retriever(database, candidate)
        except (RuntimeError, ValueError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        RUNTIME_SETTINGS_FILE.write_text(
            json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return JSONResponse({"saved": True, "restart_required": True})

    @app.post("/api/embeddings/build")
    def build_embeddings(body: EmbeddingBuildBody) -> JSONResponse:
        if "vector" not in retriever.routes:
            return JSONResponse(
                {"error": "当前 retrieval_mode=bm25，不需要向量缓存"}, status_code=400
            )
        if retriever.vector.is_cloud and not body.confirm_cloud_send:
            return JSONResponse(
                {
                    "error": "云端构建需要显式确认：将发送标题、标题层级、标签、正文到 Embedding API"
                },
                status_code=400,
            )
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '正在处理上一项操作，请稍后再试。'}, status_code=409)
        try:
            filled = retriever.vector.ensure_vectors()
        except (RuntimeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        finally:
            operation_lock.release()
        return JSONResponse(
            {
                "vectors_rebuilt": filled,
                "provider": retriever.vector.backend.provider,
                "model": retriever.vector.backend.model,
                "cloud": retriever.vector.is_cloud,
            }
        )

    @app.post("/api/ask")
    def ask(body: AskBody) -> JSONResponse:
        if not body.question.strip():
            return JSONResponse({'error': '请先写下一句话。'}, status_code=400)
        if body.thread_id is not None and not database.fetchone('SELECT 1 FROM threads WHERE id=?', (body.thread_id,)):
            return JSONResponse({'error': '这段对话不存在，请开始新的回看。'}, status_code=404)
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '还有一项回看正在进行。完成后再试就好。'}, status_code=409)
        try:
            result = harness.run(body.question, thread_id=body.thread_id)
        except (RuntimeError, ValueError):
            return JSONResponse({'error': '这次回看没有完成。请检查连接与检索设置，原始笔记未被修改。'}, status_code=503)
        finally:
            operation_lock.release()
        return JSONResponse(result_payload(result))

    @app.post('/api/ask/jobs')
    def start_job(body: JobBody) -> JSONResponse:
        try:
            return JSONResponse(jobs.submit(body.request_id, body.question, body.thread_id), status_code=202)
        except JobConflict as exc:
            return JSONResponse({'error': str(exc)}, status_code=409)
        except KeyError:
            return JSONResponse({'error': '这段对话不存在。'}, status_code=404)
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)

    @app.get('/api/ask/jobs/{request_id}')
    def get_job(request_id: str) -> JSONResponse:
        data = jobs.get(request_id)
        return JSONResponse(data if data else {'error': '这次请求不存在。'}, status_code=200 if data else 404)

    @app.get('/api/threads/{thread_id}/jobs')
    def thread_jobs(thread_id: int) -> JSONResponse:
        return JSONResponse({'items': jobs.list_for_thread(thread_id)})

    @app.post('/api/threads')
    def create_thread() -> JSONResponse:
        thread_id = database.execute("INSERT INTO threads(title,created_at) VALUES('',?)", (utc_now(),))
        return JSONResponse({'thread_id': thread_id}, status_code=201)

    @app.get('/api/memory')
    def list_memory(topic: str = Query(default='', max_length=500), include_inactive: bool = False,
                    limit: int = Query(default=30, ge=1, le=100), offset: int = Query(default=0, ge=0)) -> JSONResponse:
        return JSONResponse({'items': MemoryService(database).list_items(topic, include_inactive=include_inactive, limit=limit, offset=offset)})

    @app.post('/api/memory/{memory_id}/revoke')
    def revoke_memory(memory_id: int) -> JSONResponse:
        try:
            return JSONResponse(MemoryService(database).revoke(memory_id))
        except KeyError:
            return JSONResponse({'error': '这条记忆不存在。'}, status_code=404)

    @app.get('/api/map')
    def local_map(topic: str = Query(default='', max_length=500), source_uid: str | None = None,
                  limit: int | None = Query(default=None, ge=1, le=GLOBAL_MAX_NODES),
                  scope: Literal['local', 'global'] = 'local',
                  query: str = Query(default='', max_length=500)) -> JSONResponse:
        from .graph import build_map

        selected_limit = limit if limit is not None else GLOBAL_MAX_NODES if scope == 'global' else 30
        return JSONResponse(build_map(database, topic=topic, source_uid=source_uid,
                                      limit=selected_limit, scope=scope, query=query))

    @app.post("/api/verdict")
    def verdict(body: VerdictBody) -> JSONResponse:
        try:
            saved = VerdictService(database).save_verdict(
                message_id=body.message_id,
                verdict=body.verdict,
                user_revision=body.user_revision,
                missing_event=body.missing_event,
            )
        except (ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(saved)

    @app.get("/api/discover")
    def discover(limit: int = Query(default=3, ge=1, le=5)) -> JSONResponse:
        classifier = None
        if settings.backend != 'local' and settings.llm_ready:
            from .llm import OpenAICompatibleClient
            from .snapshots import LLMChangeClassifier

            classifier = LLMChangeClassifier(OpenAICompatibleClient(settings))
        service = DiscoveryService(database, retriever, classifier)
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '正在回看记录，请稍后再试。'}, status_code=409)
        try:
            _, candidates = service.run_scan(limit=limit)
            service.mark_shown([item.discovery_id for item in candidates])
        finally:
            operation_lock.release()
        return JSONResponse(
            {"candidates": [item.model_dump() for item in candidates]}
        )

    @app.post('/api/discover/dismiss')
    def dismiss_discovery(body: DismissBody) -> JSONResponse:
        try:
            return JSONResponse(DiscoveryService(database, retriever).dismiss(body.discovery_id))
        except KeyError:
            return JSONResponse({'error': '这条线索已不可用。'}, status_code=404)

    @app.post('/api/sync')
    def sync_vault() -> JSONResponse:
        if not operation_lock.acquire(blocking=False):
            return JSONResponse({'error': '正在回看记录，请稍后再更新索引。'}, status_code=409)
        try:
            report = VaultSyncService(database, settings.vault_path).sync()
            return JSONResponse({'synced': True, 'changed': report['changed']})
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        finally:
            operation_lock.release()

    class ReactBody(BaseModel):
        discovery_id: int
        reaction: str  # accurate | wrong | boring

    @app.post("/api/discover/react")
    def react(body: ReactBody) -> JSONResponse:
        try:
            outcome = DiscoveryService(database, retriever).react(
                body.discovery_id, body.reaction
            )
        except (ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(outcome)

    @app.post("/api/discover/review")
    def review(body: ReviewBody) -> JSONResponse:
        try:
            outcome = DiscoveryService(database, retriever).review(
                body.discovery_id, body.verdict, body.user_revision, body.missing_event
            )
        except (ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(outcome)

    @app.get("/api/runs/{message_id}")
    def runs(message_id: int) -> JSONResponse:
        rows = database.fetchall(
            "SELECT * FROM agent_runs WHERE message_id=? ORDER BY id DESC", (message_id,)
        )
        return JSONResponse(
            [
                {
                    "backend": row["backend"],
                    "steps": row["steps"],
                    "tool_calls": row["tool_calls"],
                    "latency_ms": row["latency_ms"],
                    "stop_reason": row["stop_reason"],
                    "trace": json.loads(row["trace_json"] or "[]"),
                }
                for row in rows
            ]
        )

    @app.get("/api/threads")
    def list_threads(q: str = Query(default='', max_length=200), limit: int = Query(default=30, ge=1, le=100),
                     offset: int = Query(default=0, ge=0)) -> JSONResponse:
        rows = database.fetchall(
            """
            SELECT t.id AS thread_id,
                   (SELECT content FROM messages m WHERE m.thread_id=t.id AND m.role='user'
                     ORDER BY m.id LIMIT 1) AS title,
                   (SELECT created_at FROM messages m WHERE m.thread_id=t.id
                     ORDER BY m.id DESC LIMIT 1) AS last_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.thread_id=t.id) AS turns
            FROM threads t WHERE EXISTS(SELECT 1 FROM messages m WHERE m.thread_id=t.id)
              AND (?='' OR instr(lower(t.title),lower(?))>0 OR EXISTS(
                SELECT 1 FROM messages m WHERE m.thread_id=t.id AND instr(lower(m.content),lower(?))>0))
            ORDER BY last_at DESC, t.id DESC LIMIT ? OFFSET ?
            """, (q.strip(), q.strip(), q.strip(), limit, offset)
        )
        return JSONResponse([dict(r) for r in rows])

    @app.get("/api/threads/{thread_id}")
    def get_thread(thread_id: int) -> JSONResponse:
        if not database.fetchone('SELECT 1 FROM threads WHERE id=?', (thread_id,)):
            return JSONResponse({'error': '这段对话不存在。'}, status_code=404)
        rows = database.fetchall(
            "SELECT id, role, content, answer_json, created_at FROM messages"
            " WHERE thread_id=? ORDER BY id",
            (thread_id,),
        )
        out = []
        for r in rows:
            item: dict[str, Any] = {
                "id": int(r["id"]),
                "role": str(r["role"]),
                "content": str(r["content"]),
                "created_at": str(r["created_at"]),
            }
            if r["answer_json"]:
                with contextlib.suppress(json.JSONDecodeError):
                    item["answer"] = json.loads(str(r["answer_json"]))
            saved_verdict = database.fetchone(
                'SELECT id,verdict,user_revision,created_at,topic_key,topic_label,status FROM verdicts WHERE message_id=? ORDER BY id DESC LIMIT 1',
                (r['id'],),
            )
            if saved_verdict:
                item['verdict'] = dict(saved_verdict)
            run = database.fetchone('SELECT backend FROM agent_runs WHERE message_id=? ORDER BY id DESC LIMIT 1', (r['id'],))
            if run:
                item['backend'] = run['backend']
            out.append(item)
        return JSONResponse(out)

    @app.get("/api/health")
    def health() -> JSONResponse:
        counts = {}
        for table in ("sources", "source_atoms", "messages", "agent_runs", "verdicts"):
            row = database.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(row["n"]) if row else 0
        present_sources = database.fetchone('SELECT COUNT(*) AS n FROM sources WHERE is_present=1')
        active_atoms = database.fetchone(
            'SELECT COUNT(*) AS n FROM source_atoms a JOIN sources s ON s.id=a.source_id '
            'WHERE a.is_current=1 AND s.is_present=1'
        )
        counts['sources'] = int(present_sources['n']) if present_sources else 0
        counts['source_atoms'] = int(active_atoms['n']) if active_atoms else 0
        last_sync = database.fetchone('SELECT created_at FROM sync_runs ORDER BY id DESC LIMIT 1')
        workspace = database.fetchone("SELECT value FROM schema_meta WHERE key='workspace_id'")
        imported = database.fetchone("SELECT COUNT(*) n FROM sources WHERE source_kind='chat'")
        return JSONResponse({
            'ok': True, 'backend': settings.backend, 'counts': counts,
            'workspace_id': workspace['value'] if workspace else None,
            'imported_source_count': int(imported['n']) if imported else 0,
            'assistant_name': settings.assistant_name, 'public_demo_mode': settings.public_demo_mode,
            'demo_use_model': settings.demo_use_model,
            'generation_model': settings.llm_chat_model if provider else None,
            'generation_connected': provider is not None,
            'cloud_retrieval': retriever.vector.is_cloud or settings.reranker_backend == 'api',
            'last_sync': last_sync['created_at'] if last_sync else None,
        })

    return app


def _configure_security(app: FastAPI) -> None:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Validation errors must not echo credentials, private paths, or note text.
        return JSONResponse({'error': '请求参数格式或范围不正确，请检查后重试。',
            'detail': [{'type': error['type'], 'loc': error['loc'], 'msg': '输入格式或范围不正确。'}
                       for error in exc.errors()]}, status_code=422)

    @app.middleware('http')
    async def check_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        try:
            foreign_origin = bool(origin and urlsplit(origin).netloc != request.headers.get('host'))
        except ValueError:
            foreign_origin = True
        if foreign_origin or request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'error': '请从本机 Memory Garden 页面发起操作。'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        if request.url.path in {'/', '/settings', '/history', '/import'}:
            response.headers['Content-Security-Policy'] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            )
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    from .workspaces import WorkspaceConflict, WorkspaceDispatcher, WorkspaceManager

    manager = WorkspaceManager(settings or Settings.load(), _create_workspace_app)
    app = FastAPI(title="Memory Garden", version="2.0.0")
    _configure_security(app)
    app.state.workspaces = manager

    def activate(workspace_app: FastAPI) -> None:
        # Keep the existing testing/integration interface pointed at the active app.
        for name in ('database', 'harness', 'jobs', 'settings', 'operation_lock'):
            setattr(app.state, name, getattr(workspace_app.state, name))

    manager.on_activate = activate
    activate(manager.active.app)

    @app.get('/api/workspace')
    def current_workspace(request: Request) -> JSONResponse:
        try:
            data = manager.describe(request.headers.get('x-mg-workspace-id'))
            return JSONResponse(data, headers={'X-MG-Workspace-ID': data['workspace_id']})
        except WorkspaceConflict as exc:
            return exc.response()

    @app.post('/api/workspace/connect')
    def connect_workspace(body: ConnectVaultBody, request: Request) -> JSONResponse:
        try:
            data = manager.connect(body.vault_path, request.headers.get('x-mg-workspace-id'))
            return JSONResponse(data, headers={'X-MG-Workspace-ID': data['workspace_id']})
        except WorkspaceConflict as exc:
            return exc.response()
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        except (OSError, RuntimeError, sqlite3.Error):
            return JSONResponse({'error': '连接未完成，原笔记库仍可使用。请检查本地路径和数据目录权限。'}, status_code=400)

    # A router fallback preserves routes added by local integrations/tests after
    # create_app; mounting '/' would swallow those later routes.
    app.router.default = WorkspaceDispatcher(manager)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8766)
