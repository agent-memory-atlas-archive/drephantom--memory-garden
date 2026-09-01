"""极简本地 Web UI：一个对话页 + 发现评审页。FastAPI 单文件，无前端构建链。"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import AgentHarness
from .cli import build_database
from .cognitive import DiscoveryService, VerdictService
from .config import Settings
from .db import Database
from .retrieval import build_retriever


class AskBody(BaseModel):
    question: str
    thread_id: int | None = None


class VerdictBody(BaseModel):
    message_id: int
    verdict: str
    user_revision: str = ""
    missing_event: str = ""


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


# 页面与 Python 分离：用户可直接改 page.html 调整样式与文案（改完刷新即生效）
PAGE_FILE = Path(__file__).resolve().parent / "page.html"

SETTINGS_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>设置 · Memory Garden</title>
<style>
 :root{--ink:#33302a;--sub:#8d8779;--pine:#3c5a47;--paper:#f7f5f0;--line:#e6e1d5}
 body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,"Segoe UI",sans-serif;
      display:flex;flex-direction:column;height:100vh}
 header{padding:18px 26px;background:linear-gradient(180deg,#33493b,#3c5a47);color:#f4f2ea;
        display:flex;align-items:baseline;gap:14px}
 header h1{font-size:18px;margin:0}
 header a{color:#f4f2ea;font-size:12.5px;text-decoration:none}
 main{flex:1;overflow-y:auto}
 .wrap{max-width:640px;margin:0 auto;padding:28px 20px}
 h2{font-size:15px;color:var(--pine);margin:26px 0 10px}
 label{display:block;font-size:13px;color:var(--sub);margin:14px 0 5px}
 input,select{width:100%;padding:10px 12px;border:1px solid #d8d2c2;border-radius:10px;
        font:inherit;font-size:14px;background:#fff;box-sizing:border-box;outline:none}
 input:focus,select:focus{border-color:#b5a16b}
 .hint{font-size:12px;color:var(--sub);margin-top:4px;line-height:1.6}
 button{background:var(--pine);color:#fff;border:0;border-radius:10px;padding:11px 26px;
        font:inherit;font-size:14.5px;cursor:pointer;margin-top:22px}
 .saved{color:#3c7a4e;font-size:13.5px;margin-left:14px}
 .warn{background:#faf6ec;border:1px solid #efe6cf;border-radius:10px;padding:12px 14px;
       font-size:13px;color:#8a6d3b;line-height:1.8;margin-top:18px}
</style></head><body>
<header><h1>设置</h1><a href="/">← 回到对话</a></header>
<main><div class="wrap">
 <h2>她</h2>
 <label>名字</label>
 <input id="name" placeholder="知微">
 <div class="hint">出现在头像、标题和她对自己的称呼里。性格在项目根目录 soul.md 里改，改完即时生效。</div>

 <h2>连接模型（OpenAI 兼容）</h2>
 <label>后端</label>
 <select id="backend"><option value="local">local · 离线确定性（无需密钥，回答来自固定管线）</option>
 <option value="deepseek">模型工具循环（真实对话与发现）</option></select>
 <label>Base URL</label>
 <input id="base" placeholder="https://api.deepseek.com">
 <label>模型名</label>
 <input id="model" placeholder="deepseek-v4-flash">
 <label>API Key</label>
 <input id="key" type="password" placeholder="">
 <div class="hint" id="keyhint"></div>

 <h2>检索与 Embedding</h2>
 <label>检索模式</label>
 <select id="retrieval"><option value="hybrid">hybrid · BM25 + 当前向量后端</option>
 <option value="bm25">bm25 · 仅 SQLite FTS5</option>
 <option value="hash_vector">hash_vector · 仅离线字符 n-gram</option>
 <option value="embedding">embedding · 仅 mock/API Embedding</option></select>
 <label>Embedding 后端</label>
 <select id="embedding"><option value="local_hash">local_hash · 默认离线，不发送数据</option>
 <option value="mock">mock · 仅测试，不代表真实模型效果</option>
 <option value="api">api · OpenAI 兼容 /embeddings</option></select>
 <label>Embedding 服务商标识</label>
 <input id="embeddingProvider" placeholder="siliconflow">
 <label>Embedding Base URL</label>
 <input id="embeddingBase" placeholder="https://api.siliconflow.cn/v1">
 <label>Embedding 模型名</label>
 <input id="embeddingModel" placeholder="BAAI/bge-m3">
 <label>Embedding API Key</label>
 <input id="embeddingKey" type="password" placeholder="">
 <label>第二阶段重排序</label>
 <select id="reranker"><option value="local_heuristic">local_heuristic · 离线确定性规则重排</option>
 <option value="api">api · 真实 cross-encoder /rerank</option>
 <option value="none">none · 仅保留 RRF 排序</option></select>
 <label>Rerank 服务商标识</label>
 <input id="rerankerProvider" placeholder="siliconflow">
 <label>Rerank Base URL</label>
 <input id="rerankerBase" placeholder="https://api.siliconflow.cn/v1">
 <label>Rerank 模型名</label>
 <input id="rerankerModel" placeholder="BAAI/bge-reranker-v2-m3">
 <label>Rerank API Key</label>
 <input id="rerankerKey" type="password" placeholder="留空则沿用 Embedding Key">
 <label>重排序候选数</label>
 <input id="rerankLimit" type="number" min="1" max="200" value="30">
 <label>Cross-Encoder 与 RRF 的合并方式</label>
 <select id="rerankerFusion"><option value="rank_fusion">rank_fusion · 融合两种排名（默认）</option>
 <option value="replace">replace · 只使用 Cross-Encoder 排名（实验对照）</option></select>
 <div class="hint">local_heuristic 会综合 RRF 分数、查询词覆盖、标题/标签命中和双路一致性；
  它是可复现基线，不是 cross-encoder 或真实神经重排序模型。上方合并方式仅作用于 API Rerank。</div>
 <label><input id="allowCloud" type="checkbox" style="width:auto;margin-right:8px">
 明确允许云端 Embedding</label>
 <label><input id="allowCloudRerank" type="checkbox" style="width:auto;margin-right:8px">
 明确允许云端 cross-encoder Rerank</label>
 <div class="warn">默认不会上传私人笔记。只有选择 api 且勾选上方开关后，检索才允许把
   <strong>每条原子的标题、标题层级、标签、正文</strong>分批发送到配置的 /embeddings 接口；查询文本也会发送。
   开启 API Rerank 后还会发送<strong>查询文本与 RRF 候选的标题、标题层级、标签、正文</strong>。
   原始 Vault 仍只读，返回向量只写入派生数据库。Web 启动本身不会批量发送云端向量，
   也不会在没有查询时调用 Rerank。</div>
 <button type="button" onclick="buildEmbeddings()">显式构建当前向量缓存</button>
 <span class="saved" id="buildmsg"></span>
 <div class="hint">此按钮使用进程当前已加载的配置；保存设置后请先重启，再执行构建。</div>

 <h2>数据</h2>
 <label>Obsidian Vault 路径（只读）</label>
 <input id="vault" placeholder="D:/path/to/Obsidian Vault">
 <div class="hint">密钥保存在本机 .local/settings.json（已被 .gitignore 排除，不会进仓库）。
  保存后需要重启生效：双击 stop-memory-garden.bat，再双击 start-memory-garden.bat。</div>
 <button onclick="save()">保存</button><span class="saved" id="msg"></span>
</div></main>
<script>
fetch('/api/settings').then(r=>r.json()).then(s=>{
  document.getElementById('name').value = s.assistant_name || '';
  document.getElementById('base').value = s.llm_base_url || '';
  document.getElementById('model').value = s.llm_chat_model || '';
  document.getElementById('embeddingProvider').value = s.embedding_provider || 'openai_compatible';
  document.getElementById('embeddingBase').value = s.embedding_base_url || '';
  document.getElementById('embeddingModel').value = s.llm_embedding_model || '';
  document.getElementById('rerankerProvider').value = s.reranker_provider || 'openai_compatible';
  document.getElementById('rerankerBase').value = s.reranker_base_url || '';
  document.getElementById('rerankerModel').value = s.reranker_model || '';
  document.getElementById('vault').value = s.vault_path || '';
  document.getElementById('backend').value = s.backend || 'local';
  document.getElementById('embedding').value = s.embedding_backend || 'local_hash';
  document.getElementById('retrieval').value = s.retrieval_mode || 'hybrid';
  document.getElementById('reranker').value = s.reranker_backend || 'local_heuristic';
  document.getElementById('rerankLimit').value = s.rerank_candidate_limit || 30;
  document.getElementById('rerankerFusion').value = s.reranker_fusion || 'rank_fusion';
  document.getElementById('allowCloud').checked = !!s.allow_cloud_embedding;
  document.getElementById('allowCloudRerank').checked = !!s.allow_cloud_rerank;
  document.getElementById('key').placeholder = s.api_key_set
    ? '已保存——留空表示不修改' : 'sk-...';
  document.getElementById('embeddingKey').placeholder = s.embedding_api_key_set
    ? '已保存——留空表示不修改' : 'sk-...';
  document.getElementById('rerankerKey').placeholder = s.reranker_api_key_set
    ? '已保存——留空表示不修改' : '可留空沿用 Embedding Key';
});
function save(){
  const body = {
    assistant_name: document.getElementById('name').value.trim(),
    llm_base_url: document.getElementById('base').value.trim(),
    llm_chat_model: document.getElementById('model').value.trim(),
    embedding_provider: document.getElementById('embeddingProvider').value.trim(),
    embedding_base_url: document.getElementById('embeddingBase').value.trim(),
    llm_embedding_model: document.getElementById('embeddingModel').value.trim(),
    vault_path: document.getElementById('vault').value.trim(),
    backend: document.getElementById('backend').value,
    embedding_backend: document.getElementById('embedding').value,
    retrieval_mode: document.getElementById('retrieval').value,
    reranker_backend: document.getElementById('reranker').value,
    reranker_provider: document.getElementById('rerankerProvider').value.trim(),
    reranker_base_url: document.getElementById('rerankerBase').value.trim(),
    reranker_model: document.getElementById('rerankerModel').value.trim(),
    rerank_candidate_limit: parseInt(document.getElementById('rerankLimit').value || '30'),
    reranker_fusion: document.getElementById('rerankerFusion').value,
    allow_cloud_embedding: document.getElementById('allowCloud').checked,
    allow_cloud_rerank: document.getElementById('allowCloudRerank').checked,
  };
  const key = document.getElementById('key').value.trim();
  if (key) body.llm_api_key = key;
  const embeddingKey = document.getElementById('embeddingKey').value.trim();
  if (embeddingKey) body.embedding_api_key = embeddingKey;
  const rerankerKey = document.getElementById('rerankerKey').value.trim();
  if (rerankerKey) body.reranker_api_key = rerankerKey;
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)}).then(r=>r.json()).then(()=>{
      document.getElementById('msg').textContent = '已保存，重启后生效。';
    });
}
function buildEmbeddings(){
  const cloud = document.getElementById('embedding').value === 'api';
  if (cloud && !confirm('这会把标题、标题层级、标签、正文批量发送到已配置的 Embedding API。继续吗？')) return;
  fetch('/api/embeddings/build',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({confirm_cloud_send: cloud})}).then(async r=>{
      const data = await r.json();
      document.getElementById('buildmsg').textContent = r.ok
        ? '已构建 ' + data.vectors_rebuilt + ' 条。' : (data.error || '构建失败');
    });
}
</script></body></html>"""


def load_page(settings: Settings) -> str:
    if PAGE_FILE.exists():
        name = settings.assistant_name
        return PAGE_FILE.read_text(encoding="utf-8").replace("{{NAME}}", name)
    return "<html><body><p>page.html 缺失，请重新拉取仓库。</p></body></html>"

HISTORY_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>历史 · Memory Garden</title>
<style>
 body{margin:0;background:#f7f5f0;color:#33302a;font-family:system-ui,"Segoe UI",sans-serif;
      display:flex;flex-direction:column;height:100vh}
 header{padding:18px 26px;background:linear-gradient(180deg,#33493b,#3c5a47);color:#f4f2ea;
        display:flex;align-items:baseline;gap:14px}
 header h1{font-size:18px;margin:0}
 header a{color:#f4f2ea;font-size:12.5px;text-decoration:none}
 main{flex:1;overflow-y:auto}
 .wrap{max-width:680px;margin:0 auto;padding:26px 20px}
 .item{display:block;background:#fff;border:1px solid #e6e1d5;border-radius:12px;
       padding:14px 18px;margin-bottom:12px;text-decoration:none;color:inherit}
 .item:hover{border-color:#b5a16b}
 .item .t{font-size:14.5px;margin-bottom:4px}
 .item .m{font-size:12px;color:#8d8779}
 .new{display:inline-block;background:#3c5a47;color:#fff;border-radius:10px;padding:10px 20px;
      text-decoration:none;font-size:14px;margin-bottom:20px}
 .empty{color:#8d8779;font-size:14px}
</style></head><body>
<header><h1>历史对话</h1><a href="/">← 回到对话</a> <a href="/settings">设置</a></header>
<main><div class="wrap">
<a class="new" href="/">＋ 开始新对话</a>
<div id="list"></div>
</div></main>
<script>
fetch('/api/threads').then(r=>r.json()).then(ts=>{
  const box = document.getElementById('list');
  if (!ts.length){ box.innerHTML = '<div class="empty">还没有对话。回首页说第一句话吧。</div>'; return; }
  ts.forEach(function(t){
    const a = document.createElement('a'); a.className='item'; a.href = '/?thread=' + t.thread_id;
    const title = (t.title||'（未命名）').slice(0,50);
    const when = (t.last_at||'').replace('T',' ').slice(0,16);
    const titleNode = document.createElement('div'); titleNode.className='t';
    titleNode.textContent = title;
    const metaNode = document.createElement('div'); metaNode.className='m';
    metaNode.textContent = when + ' · ' + t.turns + ' 条消息';
    a.appendChild(titleNode); a.appendChild(metaNode);
    box.appendChild(a);
  });
});
</script></body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    database: Database = build_database(settings)
    retriever = build_retriever(database, settings)
    # serve 自包含：空库时自动完成一次只读同步。云端向量绝不在 Web 启动时批量发送。
    if not database.fetchone("SELECT 1 FROM sources LIMIT 1"):
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

    app = FastAPI(title="Memory Garden", version="2.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return load_page(settings)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page() -> str:
        return SETTINGS_HTML

    @app.get("/history", response_class=HTMLResponse)
    def history_page() -> str:
        return HISTORY_HTML

    RUNTIME_SETTINGS_FILE = Path(settings.database_path).parent / "settings.json"

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
            "embedding_backend": {"local_hash", "mock", "api"},
            "retrieval_mode": {"bm25", "hash_vector", "embedding", "hybrid"},
            "reranker_backend": {"none", "local_heuristic", "api"},
            "reranker_fusion": {"replace", "rank_fusion"},
        }
        for field, choices in allowed_values.items():
            value = getattr(body, field).strip().lower()
            if value and value not in choices:
                return JSONResponse({"error": f"{field} 值无效"}, status_code=400)
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
        try:
            filled = retriever.vector.ensure_vectors()
        except (RuntimeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
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
        result = harness.run(body.question, thread_id=body.thread_id)
        # 判定按钮只在结论型回答后出现；情感陪伴/澄清追问后面挂表单会非常出戏
        verdict_worthy = result.answer.answer_type in {"traced_change", "no_clear_change"} or bool(
            result.answer.question_to_user
        )
        # 未知项过滤掉每次都一样的套话，只留真正针对本轮的
        unknowns = [
            u for u in result.answer.unknowns
            if "现有来源不能证明" not in u and "尚未经用户确认" not in u
        ]
        return JSONResponse(
            {
                "reply": result.reply,
                "answer": result.answer.model_dump(),
                "answer_type": result.answer.answer_type,
                "unknowns": unknowns,
                "verdict_worthy": verdict_worthy,
                "message_id": result.message_id,
                "thread_id": result.thread_id,
                "backend": result.backend,
                "steps": result.steps,
                "tool_calls": result.tool_calls,
                "stop_reason": result.stop_reason,
            }
        )

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
    def discover(limit: int = 5) -> JSONResponse:
        classifier = None
        if settings.llm_ready:
            from .llm import OpenAICompatibleClient
            from .snapshots import LLMChangeClassifier

            classifier = LLMChangeClassifier(OpenAICompatibleClient(settings))
        service = DiscoveryService(database, retriever, classifier)
        _, candidates = service.run_scan(limit=limit)
        service.mark_shown([item.discovery_id for item in candidates])
        return JSONResponse(
            {"candidates": [item.model_dump() for item in candidates]}
        )

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
    def list_threads() -> JSONResponse:
        rows = database.fetchall(
            """
            SELECT t.id AS thread_id,
                   (SELECT content FROM messages m WHERE m.thread_id=t.id AND m.role='user'
                     ORDER BY m.id LIMIT 1) AS title,
                   (SELECT created_at FROM messages m WHERE m.thread_id=t.id
                     ORDER BY m.id DESC LIMIT 1) AS last_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.thread_id=t.id) AS turns
            FROM threads t ORDER BY t.id DESC LIMIT 50
            """
        )
        return JSONResponse([dict(r) for r in rows])

    @app.get("/api/threads/{thread_id}")
    def get_thread(thread_id: int) -> JSONResponse:
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
            out.append(item)
        return JSONResponse(out)

    @app.get("/api/health")
    def health() -> JSONResponse:
        counts = {}
        for table in ("sources", "source_atoms", "messages", "agent_runs", "verdicts"):
            row = database.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(row["n"]) if row else 0
        return JSONResponse({"ok": True, "backend": settings.backend, "counts": counts})

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8766)
