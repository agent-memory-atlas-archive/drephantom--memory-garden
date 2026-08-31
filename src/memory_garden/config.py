"""Settings loaded from environment / .env file. No secrets are ever logged."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


@dataclass
class Settings:
    vault_path: Path
    database_path: Path
    soul_path: Path | None = None  # 人格文件；每条消息热加载（学 Hermes SOUL.md）
    assistant_name: str = "知微"
    backend: str = "local"  # "local" (deterministic offline) | "deepseek" (tool loop)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_chat_model: str = ""
    llm_embedding_model: str = ""
    llm_reasoning_effort: str = "none"  # none 关闭思考流：工具循环步骤快且时长可控
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    agent_max_steps: int = 10
    agent_max_tool_calls: int = 16
    agent_tool_timeout_seconds: float = 10.0
    agent_overall_timeout_seconds: float = 180.0
    agent_max_repeat_calls: int = 2
    agent_no_progress_steps: int = 3
    public_demo_mode: bool = False
    _env: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = project_root or Path(__file__).resolve().parents[2]
        # 优先级：进程环境 > 运行时设置（设置页写入） > .env 文件 > 默认值。
        # 运行时设置只接受非空值，供 GitHub 用户免改文件完成配置。
        env: dict[str, str] = {}
        runtime_file = root / ".local" / "settings.json"
        if runtime_file.exists():
            try:
                runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                runtime = {}
            for key, mg_key in (
                ("vault_path", "MG_VAULT_PATH"),
                ("llm_base_url", "MG_LLM_BASE_URL"),
                ("llm_api_key", "MG_LLM_API_KEY"),
                ("llm_chat_model", "MG_LLM_CHAT_MODEL"),
                ("assistant_name", "MG_ASSISTANT_NAME"),
                ("backend", "MG_ASSISTANT_BACKEND"),
            ):
                value = str(runtime.get(key) or "").strip()
                if value:
                    env[mg_key] = value
        env.update(load_env_file(root / ".env"))
        env.update(os.environ)
        key = env.get("MG_LLM_API_KEY", "")
        key_file = env.get("MG_LLM_API_KEY_FILE", "")
        if not key and key_file and Path(key_file).exists():
            key = Path(key_file).read_text(encoding="utf-8").strip()
        settings = cls(
            vault_path=Path(env.get("MG_VAULT_PATH", str(root / "evals" / "cognitive_mvp_vault"))),
            database_path=Path(env.get("MG_DATABASE_PATH", str(root / ".local" / "memory_garden.db"))),
            soul_path=Path(env.get("MG_SOUL_PATH", str(root / "soul.md"))),
            backend=env.get("MG_ASSISTANT_BACKEND", "local"),
            llm_base_url=env.get("MG_LLM_BASE_URL", ""),
            llm_api_key=key,
            llm_chat_model=env.get("MG_LLM_CHAT_MODEL", ""),
            llm_embedding_model=env.get("MG_LLM_EMBEDDING_MODEL", ""),
            llm_reasoning_effort=env.get("MG_LLM_REASONING_EFFORT", "none"),
            assistant_name=env.get("MG_ASSISTANT_NAME", "知微"),
            llm_timeout_seconds=float(env.get("MG_LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=int(env.get("MG_LLM_MAX_RETRIES", "2")),
            agent_max_steps=int(env.get("MG_AGENT_MAX_STEPS", "10")),
            agent_max_tool_calls=int(env.get("MG_AGENT_MAX_TOOL_CALLS", "16")),
            agent_tool_timeout_seconds=float(env.get("MG_AGENT_TOOL_TIMEOUT_SECONDS", "10")),
            agent_overall_timeout_seconds=float(env.get("MG_AGENT_OVERALL_TIMEOUT_SECONDS", "180")),
            agent_max_repeat_calls=int(env.get("MG_AGENT_MAX_REPEAT_CALLS", "2")),
            agent_no_progress_steps=int(env.get("MG_AGENT_NO_PROGRESS_STEPS", "3")),
            public_demo_mode=env.get("MG_PUBLIC_DEMO_MODE", "false").lower() == "true",
        )
        settings._env = env
        if settings.public_demo_mode:
            # 公共演示模式下，Vault 必须指向仓库内的合成数据，否则拒绝启动。
            allowed = (root / "evals" / "cognitive_mvp_vault").resolve()
            if settings.vault_path.resolve() != allowed:
                raise ValueError("公共演示模式仅允许指向仓库内合成 Sample Vault")
        return settings

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_chat_model)
