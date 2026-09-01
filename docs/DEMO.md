# 5 分钟演示脚本（离线可复现，不依赖网络与密钥）

前提：`uv sync` 已完成。以下每条命令都可现场执行；预期输出标注在注释里。

## 0. 环境（30 秒）

```powershell
cd memory-garden
# 确保未配置 LLM（离线确定性模式）；.env 中 MG_ASSISTANT_BACKEND=local 或直接不配 key
# 没有可用 Vault 时设置 MG_PUBLIC_DEMO_MODE=true，init 将强制使用仓库内合成 Vault
uv run memory-garden init
```

演示要点：**Vault 只读**——同步前后全库哈希不变（`verify-vault` 会再次验证）。

## 1. 认知回溯问答（90 秒）

```powershell
uv run memory-garden ask "自主判断这个主题，我的想法以前到现在有没有变化？"
```

预期要点（确定性本地回答）：
- 端点配对：2019「只有别人认可我的选择，我才敢相信…」→ 2024「先形成自己的判断…」；
- 引用 `[A9]` `[A12]` 可在 Vault 里定位到原文件原句；
- 明确说"区间经历与变化只是时间相邻"；反例记录被并列给出；
- 以"这条较近记录是否仍代表你现在的看法，还需要你确认"收尾。

对照组（演示“证据不足时停止推断”）：

```powershell
uv run memory-garden ask "火星殖民"
# → insufficient_evidence + 一个追问，不硬凑对照
```

## 2. 发现（90 秒）

```powershell
uv run memory-garden discover --limit 3
```

预期要点：候选呈现为**两端原话+日期并排**（"让差距自己说话"）；
问题措辞是"表达更具体了，还是想法真的变了？"——不给因果解释。
（配置 API Key 后可先运行 `uv run memory-garden extract-snapshots`，候选由 LLM 立场快照驱动，
并演示"书摘/AI 草稿被 has_stance 过滤"。）

一键反应写回：

```powershell
uv run memory-garden review <id> accurate
```

## 3. 反馈闭环（60 秒）

```powershell
uv run memory-garden ask "自主判断"
# 记下 message_id（输出末尾元信息行）
uv run memory-garden review 是上一轮的候选……
# 或直接运行端到端测试，验证“否认 → 再问 → 遵守判定”：
uv run pytest tests/test_agent.py::test_prior_denial_is_respected -q
```

## 4. 评测与工程护栏（60 秒）

```powershell
uv run pytest            # Embedding 缓存/RRF 重排/隐私/入口一致性
uv run ruff check src tests
uv run memory-garden eval-agent        # 三配置离线对照；结果仅保存在本机
uv run memory-garden verify-vault      # vault_read_only: true, second_sync_idempotent: true
```

## 5. MCP（30 秒，可选）

```powershell
uv run memory-garden mcp
# 或 Claude Desktop 配置 mcpServers: {"memory-garden": {"command": "uv", "args": ["run", "memory-garden", "mcp"]}}
```

演示要点：8 个只读领域工具与内部循环使用同一注册表，另有完整回溯入口 `ask_garden`；
演示 `read_source` 拒绝未发现 id（边界即接口）。

## 常见追问的现场证据

- "怎么评测？" → `docs/RETRIEVAL_EVAL_PROTOCOL.md`、`evals/` 中的脱敏用例和评测代码；
  生成结果与私有 golden 只保存在本机
- "如何验证真实模型链路？" → 在明确授权云端发送后运行私有 smoke，并在本地 `agent_runs` 表核对
  `backend`、`steps`、`latency` 与 `private_vault_sent`；公开仓库不附带真实模型运行结果
- "测试覆盖什么？" → `tests/`：导入幂等/移动身份、短词召回、引用守卫、判定遵守、快照语义过滤、发现评分、MCP 注册、Web 全链路
