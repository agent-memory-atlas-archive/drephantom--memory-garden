# Memory Garden 架构

更新：2026-09-07。本文描述当前实现；生成的评测结果与私有 golden 仅保存在本机。

## 1. 系统边界

```text
Obsidian Vault（只读访问；MG_PUBLIC_DEMO_MODE=true 时强制指向仓库内合成 Vault）
        │
        ▼  VaultSyncService（幂等，全库哈希短路）
 sources + source_revisions（不可变修订链，移动/重命名保持稳定 uid）
        │
        ▼
 source_atoms（标题层级切分 / 微信消息按时间戳切分）+ source_atoms_fts(trigram) + atom_vectors
        │
        ├── 检索层：统一工厂 → BM25 / hash_vector / embedding → RRF → reranker（可选名次再融合）
        ├── 快照层：DeterministicExtractor / LLMBatchExtractor → stance_snapshots(schema 版本化)
        │                └→ DiscoveryEngine（变化分类学 + 显式对比句 + 四因子评分）
        └── 认知工具：8 个只读 ToolSpec（同一注册表 → 内部 Harness / MCP Server）
                 │
                 ▼
        AgentHarness（OpenAI 兼容工具循环 | 确定性本地路径 | ScriptedProvider 评测）
                 │
        CognitiveAnswer（结构化契约）+ agent_runs（trace/预算/停止原因审计）
                 │
        用户判定（6 类）与一键反应（3 类）→ verdicts / candidate_reactions
                 │
        下一轮回溯自动加载判定记忆（词交集匹配）；反应写回发现排序权重
```

## 2. 数据层不变量

- **只读**：Vault 只被 `open/read`；`verify-vault` 校验"同步前后全库哈希不变 + 二次同步零变更"。
- **时间分离**：`event_time` 仅来自显式声明（frontmatter `event/event_date/date`、微信消息时间戳）；
  文件创建时间至多是低置信 `recorded_at` 回退。正文里的日期数字不自动采信。
- **身份稳定**：内容哈希相同的移动/重命名沿用原 uid（按批次认领防重复内容争用）；
  内容变化生成不可变修订；消失文件标 `is_present=0`。
- **作者归属**：frontmatter 声明 > `content_origin/generated_by` > 文件名/标题约定（`AI草稿*`、`引用*`）；
  原子级引用块（blockquote / `[引用]`）降级为 `quoted`。引用/AI 文本永不冒充用户立场。
- **schema 演进**：v4 增量迁移（升级前 SQLite 备份，历史原子保留）；v2 `atom_vectors(atom_id, dim, vector_json)` 原值无损复制为
  `legacy/unknown-v2` 身份，当前 provider 不会读取，按需生成带完整身份的新缓存；
  快照表自带 `snapshot_schema_version`，抽取逻辑变更可整版重建。

## 3. 检索层

| 路 | 实现 | 强项 | 弱点 |
|---|---|---|---|
| BM25 | FTS5 trigram 短语 + jieba 2字词 LIKE 兜底 | 精确词面、可解释 | trigram 无法命中 <3 字词 |
| 字符向量 | `local_hash`：确定性字符 2/3-gram 哈希（默认 512 维, L2） | 离线、无 Key、可复现 | 无真语义 |
| Embedding | `api`：OpenAI 兼容 `/embeddings`；`mock`：仅 CI 的固定替身 | 接口与缓存链可独立评估 | mock 不代表模型效果；api 会发送文本 |
| 混合 | BM25 + 当前向量后端，RRF 融合（k=60） | 多路召回 | RRF 不保证 MRR 高于最佳单路 |
| 重排 | 默认 `local_heuristic`；可选 `api` 调用 `/rerank` cross-encoder | 离线基线可审计；API 路线可选替换排序或与 RRF 名次再融合 | API 会发送查询与候选全文；效果需按 golden 验证 |

嵌入与重排文本固定为“标题 + 标题层级 + 标签 + 正文”，当前
`embedding_text_version=v2-heading`。`atom_vectors` 的缓存身份包含
`embedding_provider/model/dimension/text_version/text_hash` 和创建/更新时间；文本、模型、维度或
文本版本任一变化都会重建。查询只读取与当前 provider、模型、维度、文本版本、文本哈希全部匹配的行。
不同模型与 hash/API/mock 缓存可并存，但不会混用。

默认 `MG_EMBEDDING_BACKEND=local_hash`、`MG_RETRIEVAL_MODE=hybrid`、
`MG_RERANKER_BACKEND=local_heuristic`，每路候选池固定为 30。纯 RRF 与重排都从相同候选深度
开始，避免把候选扩展误算成重排收益；重排只作用于双路 hybrid，`MG_RERANKER_BACKEND=none`
保留纯 RRF 对照。API 路线中 `MG_RERANKER_FUSION=replace` 只使用 cross-encoder 排名，
`rank_fusion` 对 RRF 名次与 cross-encoder 名次再做等权 RRF。
Embedding `api` 要求 `MG_ALLOW_CLOUD_EMBEDDING=true`、专用 Base URL、API Key 和模型名；
Rerank `api` 独立要求 `MG_ALLOW_CLOUD_RERANK=true`、Base URL、API Key 与模型名；否则工厂拒绝构建。
两组连接与 `MG_LLM_*` 生成连接分离并采用 fail-closed；任一专用 Base URL/Key 缺失时拒绝构建，
不回退到生成模型 provider。
Embedding payload 包含查询文本及各原子的标题、标题层级、标签、正文；Rerank payload 包含查询及 RRF
候选的同一规范文本。Web 每次启动只读同步 Vault；本地后端可自动建缓存，云端缓存必须由后续明确
检索或设置页二次确认操作触发，不在启动阶段批量上传，Rerank 也只在实际查询时调用。

## 4. 立场快照与发现引擎（v2 核心）

- **抽取**：`ensure_snapshots` 只补缺失（内容变化的原子 id 改变 → 自然重抽）。
  LLM 路线批量（30/批）+ 已知主题表促进别名收敛 + `"无立场"合法`；
  确定性路线：标签/标题词为主题、首句为立场、元记录正则过滤（补记/无因果/待核对）。
- **对比**：同主题时间排序 → 相邻对 + 首尾对（去重）→ 分类学五类；
  只有 `true_change` / `parallel_stance` 成为候选；相邻对承担"转折点定位"。
- **显式对比句**：`(以前|过去|曾经|原本|当初)…[，,]…(现在|如今|这两年|最近)` 正则直取，
  置信度 0.85、`signal_type=explicit_contrast`，主题取笔记标签。
- **评分**：`变化置信度 × 主题重要度(快照数归一) × 新颖度(7天内展示过→0.3) × 反应权重(属实×1.15/不是×0.6/无聊×0.8, 上限2.0)`。
- **呈现**：每次呈现 3–5 条；暂缓的相同原子对七天不再呈现；两端原话+日期并排；对收到负反馈的候选降低排序权重，避免重复呈现。

## 5. Agent Harness

- **模型决定用途**：真实 Provider 先用 `plan_turn` 返回 `conversation/source_lookup/cognitive_trace/discovery`。
  模型根据最近对话补全检索主题；不先用关键词把每句话都当作回溯问题。
  一般交流另用一次有预算的调用生成回应，只带用户原话和解析后的话题，避免反复采信旧助手的个人解释。
- **模型提交回应**：工具循环暴露 `finish_response`，仅接受单独调用中的 `reply` 作为最终正文；
  不保存或显示伴随工具请求的过程文字。原文检索只要求相应来源；前后对照要求检索、时间线和可用端点配对。
  只有核对个人变化原因时，才额外要求区间事件与 support/challenge 双侧检索。
- **预算四护栏**：最大步数(10)/工具调用(16)/整体时长/重复调用(>2 拒绝)；无进展 = 连续出错观察。
  用途决定、预算耗尽后的最终生成和引用修复均计入模型步数；步骤耗尽不能额外调用模型。
- **协议合规**：批量 tool_calls 的每个 id 必须有应答（否则 provider 400）；
  预算耗尽 → 注入"立即基于已有观察作答"强制收敛，而不是静默丢弃。
- **引用守卫**：`[A{id}]` 必须属于本轮工具真实返回；越界时在剩余预算内允许一次修复，仍失败明确报告。
  真实主流程不另做无预算的文风改写；编号校验不等于自然语言主张已被语义验证。
- **证据计划守卫**：按本轮用途检查最低证据步骤；缺失时最多允许一次补工具。
  工具选择与顺序由模型决定，权限和证据完整性由 Harness 检查。
- **双侧检索落地**：`stance=challenge` 不只是返回标签；它加入反例/例外/转折查询信号，并对显式
  反例标签作稳定候选重排。该信号只决定核对顺序，不直接把候选判成反证。
- **不可信上下文**：工具观察与 Vault 文本都只作为数据；其中的提示注入文字不能改变系统规则或触发指令。
- **隐私遥测**：生成模型收到工具观察、Embedding 上传原子文本或 Rerank 上传候选文本时，
  本轮运行记录 `private_vault_sent=1`；纯本地与 scripted provider 保持 0。
- **失败行为**：真实主流程的 provider 错误、预算耗尽或引用失败 → `model_unavailable`，保留消息并告知未完成。
  确定性模板与旧的本地降级分支只用于显式离线演示/旧脚本评测，不冒充模型成功。
- **运行时工具收缩**：真实主流程仅在模型判为 discovery 且调用方允许时注册 `discover_cognitive_shifts`；
  本地演示仍用确定性主题判断。
- **结构化答案**：`CognitiveAnswer`（端点/区间/正反例/未知项/置信度/最多一问）；
  `traced_change` 必须双端点+可定位引用；较近端点固定 `latest_memory_candidate`。

DeepSeek 的强制结构化工具调用显式使用非思考模式；其他 OpenAI 兼容连接不发送 DeepSeek 专用参数。
接口依据：[DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) 与
[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。

`MG_PUBLIC_DEMO_MODE=true` 固定合成 Vault；`MG_DEMO_USE_MODEL=true` 另选 `.local/agent-demo.db`
及独立设置并保留生成连接，默认则使用离线 `.local/demo.db`。两者均禁用云端 Embedding/Rerank。

## 6. 判定与反应（用户是唯一 ground truth）

- 六类判定（基本准确/部分准确/不构成变化/不是我的观点/证据不足/暂不判断）沉淀为主题记忆；
  加载用主题词交集匹配（"自主"↔"自主判断"，个人规模全表过滤可行）；
  被否认的解释写入 counter_evidence 且不再作为结论提出（e2e 测试覆盖）。
- 一键反应（属实/不是/无聊）写回 `candidate_reactions`，是发现排序的个性化因子；
  评审与对话判定共用主题记忆，语义一致。

## 7. 评测体系

| 评测 | 隔离性 | 度量 |
|---|---|---|
| `eval-agent` | 一次性临时库 + 强制合成 Vault | 完成、引用、反例、拒答、判定遵守、无因果断言 |
| `eval-retrieval` | 一次性临时库；公开合成或本地私有 golden | 路径级 HitRate/Recall/Precision@5、MRR、nDCG@5 |
| `eval-discovery` | 每次清空快照重建 | 预期变化对召回、措辞深化泄漏、判定口径查准 |
| 私有 API smoke | 真实 Vault 只读 + 一次性派生库 + 显式云端授权 | 检索、重排、工具循环、引用边界与隐私遥测 |
| `verify-vault` | 原 Vault 前后哈希对照 | Vault 只读 + 同步幂等 |

命令生成的 `artifacts/evals/`、私有 golden、真实 API 结果和逐条诊断均被 Git 忽略；
公开仓库保留评测实现、指标口径与脱敏合成用例，私有数据评测产物不纳入版本控制。

诚实声明：合成/确定性评测用于验证**协议约束行为**（不伪造引用、证据不足时拒绝下结论、遵守用户判定），
不证明真实用户的长期受益；后者需要在持续真实使用中结合“属实且有意义”的反馈比例另行验证。

## 8. 技术取舍

- 显式状态机 + 单 Agent，不上多 Agent/编排框架：个人工具，可审计优先；
- SQLite 单文件（WAL），无向量库服务：当前个人知识库规模下使用暴力余弦检索，接口保留升级位；
- CLI、Web、MCP 直接导入同一个 `retrieval.build_retriever`，provider 选择、隐私门和缓存规则不分叉；
- `Reranker` 是独立协议：`LocalHeuristicReranker` 是默认离线基线，
  `APICrossEncoderReranker` 通过兼容 `/rerank` 的接口调用真实模型，可比较替换排序与
  名次再融合；链路跑通不等于排序效果提升，指标必须分开验证；
- FTS5 trigram 而非 jieba 分词索引：免维护词典、中文召回稳，短词缺口由混合检索第二路兜底；
- Web UI 使用 FastAPI + 独立 HTML/CSS/JavaScript：产品面刻意小，核心是 CLI/API/MCP 三通道；
- MCP 2.x `MCPServer`：工具经同一 `ToolRegistry` 暴露，使各入口共享相同的边界规则。
