"""极简本地 Web UI：一个对话页 + 发现评审页。FastAPI 单文件，无前端构建链。"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent import AgentHarness
from .cli import build_database, build_retriever
from .cognitive import DiscoveryService, VerdictService
from .config import Settings
from .db import Database


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
    llm_api_key: str = ""
    vault_path: str = ""
    backend: str = ""


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

 <h2>数据</h2>
 <label>Vault 路径（你的 Obsidian 笔记夹，只读）</label>
 <input id="vault" placeholder="D:/path/to/你的 Vault">
 <div class="hint">密钥保存在本机 .local/settings.json（已被 .gitignore 排除，不会进仓库）。
  保存后需要重启生效：双击 stop-memory-garden.bat，再双击 start-memory-garden.bat。</div>
 <button onclick="save()">保存</button><span class="saved" id="msg"></span>
</div></main>
<script>
const FIELDS = ['name','base','model','vault'];
fetch('/api/settings').then(r=>r.json()).then(s=>{
  document.getElementById('name').value = s.assistant_name || '';
  document.getElementById('base').value = s.llm_base_url || '';
  document.getElementById('model').value = s.llm_chat_model || '';
  document.getElementById('vault').value = s.vault_path || '';
  document.getElementById('backend').value = s.backend || 'local';
  document.getElementById('key').placeholder = s.api_key_set
    ? '已保存（尾号 ' + s.api_key_tail + '）——留空表示不修改' : 'sk-...';
});
function save(){
  const body = {
    assistant_name: document.getElementById('name').value.trim(),
    llm_base_url: document.getElementById('base').value.trim(),
    llm_chat_model: document.getElementById('model').value.trim(),
    vault_path: document.getElementById('vault').value.trim(),
    backend: document.getElementById('backend').value,
  };
  const key = document.getElementById('key').value.trim();
  if (key) body.llm_api_key = key;
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)}).then(r=>r.json()).then(()=>{
      document.getElementById('msg').textContent = '已保存，重启后生效。';
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
    a.innerHTML = '<div class="t">' + title + '</div><div class="m">' + when + ' · ' + t.turns + ' 条消息</div>';
    box.appendChild(a);
  });
});
</script></body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    database: Database = build_database(settings)
    retriever = build_retriever(database, settings)
    # serve 自包含：空库时自动完成一次只读同步 + 向量构建
    if not database.fetchone("SELECT 1 FROM sources LIMIT 1"):
        from .importer import VaultSyncService

        VaultSyncService(database, settings.vault_path).sync()
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
        key = settings.llm_api_key
        return JSONResponse(
            {
                "assistant_name": settings.assistant_name,
                "llm_base_url": settings.llm_base_url,
                "llm_chat_model": settings.llm_chat_model,
                "vault_path": str(settings.vault_path),
                "backend": settings.backend,
                "api_key_set": bool(key),
                "api_key_tail": key[-4:] if key else "",
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
        for field in ("assistant_name", "llm_base_url", "llm_chat_model", "vault_path", "backend"):
            value = getattr(body, field).strip()
            if value:
                current[field] = value
        if body.llm_api_key.strip():
            current["llm_api_key"] = body.llm_api_key.strip()
        RUNTIME_SETTINGS_FILE.write_text(
            json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return JSONResponse({"saved": True, "restart_required": True})

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
