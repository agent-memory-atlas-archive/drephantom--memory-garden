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


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"无法解析布尔配置值：{value!r}")


def _secret_from_env(
    env: dict[str, str], value_key: str, file_key: str, fallback: str = ""
) -> str:
    """Read a secret without ever including it in Settings repr or diagnostics."""
    value = env.get(value_key, "")
    key_file = env.get(file_key, "")
    if not value and key_file and Path(key_file).is_file():
        value = Path(key_file).read_text(encoding="utf-8").strip()
    return value or fallback


@dataclass
class Settings:
    vault_path: Path
    database_path: Path
    soul_path: Path | None = None  # 人格文件；每条消息热加载（学 Hermes SOUL.md）
    assistant_name: str = "知微"
    backend: str = "local"  # "local" (deterministic offline) | "deepseek" (tool loop)
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_chat_model: str = ""
    llm_embedding_model: str = ""
    llm_embedding_dimension: int = 0  # 0=由 provider 返回值确定
    embedding_provider: str = "openai_compatible"
    embedding_base_url: str = ""
    embedding_api_key: str = field(default="", repr=False)
    embedding_backend: str = "local_hash"  # local_hash | mock | api
    retrieval_mode: str = "hybrid"  # bm25 | hash_vector | embedding | hybrid
    reranker_backend: str = "local_heuristic"  # none | local_heuristic | api
    reranker_provider: str = "openai_compatible"
    reranker_base_url: str = ""
    reranker_api_key: str = field(default="", repr=False)
    reranker_model: str = ""
    rerank_candidate_limit: int = 30
    reranker_fusion: str = "rank_fusion"  # replace | rank_fusion
    allow_cloud_embedding: bool = False
    allow_cloud_rerank: bool = False
    local_hash_dimension: int = 512
    mock_embedding_dimension: int = 64
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
        # 运行时设置只接受非空值，使用者无需修改仓库文件即可完成配置。
        env: dict[str, str] = load_env_file(root / ".env")
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
                ("llm_embedding_model", "MG_LLM_EMBEDDING_MODEL"),
                ("llm_embedding_dimension", "MG_LLM_EMBEDDING_DIMENSION"),
                ("embedding_provider", "MG_EMBEDDING_PROVIDER"),
                ("embedding_base_url", "MG_EMBEDDING_BASE_URL"),
                ("embedding_api_key", "MG_EMBEDDING_API_KEY"),
                ("embedding_backend", "MG_EMBEDDING_BACKEND"),
                ("retrieval_mode", "MG_RETRIEVAL_MODE"),
                ("reranker_backend", "MG_RERANKER_BACKEND"),
                ("reranker_provider", "MG_RERANKER_PROVIDER"),
                ("reranker_base_url", "MG_RERANKER_BASE_URL"),
                ("reranker_api_key", "MG_RERANKER_API_KEY"),
                ("reranker_model", "MG_RERANKER_MODEL"),
                ("rerank_candidate_limit", "MG_RERANK_CANDIDATE_LIMIT"),
                ("reranker_fusion", "MG_RERANKER_FUSION"),
                ("allow_cloud_embedding", "MG_ALLOW_CLOUD_EMBEDDING"),
                ("allow_cloud_rerank", "MG_ALLOW_CLOUD_RERANK"),
                ("assistant_name", "MG_ASSISTANT_NAME"),
                ("backend", "MG_ASSISTANT_BACKEND"),
            ):
                if key not in runtime:
                    continue
                value = runtime[key]
                if isinstance(value, bool):
                    env[mg_key] = "true" if value else "false"
                elif str(value).strip():
                    env[mg_key] = str(value).strip()
        env.update(os.environ)
        key = _secret_from_env(env, "MG_LLM_API_KEY", "MG_LLM_API_KEY_FILE")
        embedding_key = _secret_from_env(
            env, "MG_EMBEDDING_API_KEY", "MG_EMBEDDING_API_KEY_FILE", fallback=key
        )
        reranker_key = _secret_from_env(
            env,
            "MG_RERANKER_API_KEY",
            "MG_RERANKER_API_KEY_FILE",
            fallback=embedding_key,
        )
        embedding_base_url = env.get("MG_EMBEDDING_BASE_URL", "") or env.get(
            "MG_LLM_BASE_URL", ""
        )
        reranker_base_url = env.get("MG_RERANKER_BASE_URL", "") or embedding_base_url
        settings = cls(
            vault_path=Path(env.get("MG_VAULT_PATH", str(root / "evals" / "cognitive_mvp_vault"))),
            database_path=Path(env.get("MG_DATABASE_PATH", str(root / ".local" / "memory_garden.db"))),
            soul_path=Path(env.get("MG_SOUL_PATH", str(root / "soul.md"))),
            backend=env.get("MG_ASSISTANT_BACKEND", "local"),
            llm_base_url=env.get("MG_LLM_BASE_URL", ""),
            llm_api_key=key,
            llm_chat_model=env.get("MG_LLM_CHAT_MODEL", ""),
            llm_embedding_model=env.get("MG_LLM_EMBEDDING_MODEL", ""),
            llm_embedding_dimension=int(env.get("MG_LLM_EMBEDDING_DIMENSION", "0")),
            embedding_provider=env.get(
                "MG_EMBEDDING_PROVIDER", "openai_compatible"
            ).strip().lower(),
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_key,
            embedding_backend=env.get("MG_EMBEDDING_BACKEND", "local_hash").strip().lower(),
            retrieval_mode=env.get("MG_RETRIEVAL_MODE", "hybrid").strip().lower(),
            reranker_backend=env.get("MG_RERANKER_BACKEND", "local_heuristic").strip().lower(),
            reranker_provider=env.get(
                "MG_RERANKER_PROVIDER", "openai_compatible"
            ).strip().lower(),
            reranker_base_url=reranker_base_url,
            reranker_api_key=reranker_key,
            reranker_model=env.get("MG_RERANKER_MODEL", ""),
            rerank_candidate_limit=int(env.get("MG_RERANK_CANDIDATE_LIMIT", "30")),
            reranker_fusion=env.get(
                "MG_RERANKER_FUSION", "rank_fusion"
            ).strip().lower(),
            allow_cloud_embedding=_as_bool(env.get("MG_ALLOW_CLOUD_EMBEDDING"), False),
            allow_cloud_rerank=_as_bool(env.get("MG_ALLOW_CLOUD_RERANK"), False),
            local_hash_dimension=int(env.get("MG_LOCAL_HASH_DIMENSION", "512")),
            mock_embedding_dimension=int(env.get("MG_MOCK_EMBEDDING_DIMENSION", "64")),
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
            public_demo_mode=_as_bool(env.get("MG_PUBLIC_DEMO_MODE"), False),
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

    @property
    def embedding_api_ready(self) -> bool:
        """API Embedding 必须具备连接信息，且由用户显式开启云端发送。"""
        return bool(
            self.allow_cloud_embedding
            and self.embedding_base_url
            and self.embedding_api_key
            and self.llm_embedding_model
        )

    @property
    def reranker_api_ready(self) -> bool:
        """API rerank also has an independent, explicit private-text consent gate."""
        return bool(
            self.allow_cloud_rerank
            and self.reranker_base_url
            and self.reranker_api_key
            and self.reranker_model
        )
