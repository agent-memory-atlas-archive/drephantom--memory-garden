# Memory Garden 架构

更新：2026-08-30。本文描述当前实现，所有声明都有对应测试或评测产物。

## 1. 系统边界

```text
Obsidian Vault（永远只读；MG_PUBLIC_DEMO_MODE=true 时强制指向仓库内合成 Vault）
        │
        ▼  VaultSyncService（幂等，全库哈希短路）
 sources + source_revisions（不可变修订链，移动/重命名保持稳定 uid）
        │
        ▼
 source_atoms（标题层级切分 / 微信消息按时间戳切分）+ source_atoms_fts(trigram) + atom_vectors
        │
        ├── 检索层：BM25Retriever ⊕ VectorRetriever → HybridRetriever(RRF)
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
- **schema 演进**：v2 增量迁移（discoveries 新列 + stance_snapshots + candidate_reactions）；
  快照表自带 `snapshot_schema_version`，抽取逻辑变更可整版重建。

## 3. 检索层

| 路 | 实现 | 强项 | 弱点 |
|---|---|---|---|
| BM25 | FTS5 trigram 短语 + jieba 2字词 LIKE 兜底 | 精确词面、可解释 | trigram 无法命中 <3 字词 |
| 向量 | 确定性字符 2/3-gram 哈希（512维, L2）；配置 API 时升级稠密 embedding | 短词/轻度改写召回、离线可复现 | 无真语义 |
| 混合 | RRF 融合（k=60），保留每路名次归因 | recall@5 双语料 1.0 | MRR 被 RRF 稀释（私有真实库 0.929 < 单向量 0.964） |

嵌入文本 = 标题 + 标签 + 正文（`ensure_vectors` 幂等补齐）。检索文本 = 标题 + 标签 + 正文（FTS 同步写入）。

## 4. 立场快照与发现引擎（v2 核心）

- **抽取**：`ensure_snapshots` 只补缺失（内容变化的原子 id 改变 → 自然重抽）。
  LLM 路线批量（30/批）+ 已知主题表促进别名收敛 + `"无立场"合法`；
  确定性路线：标签/标题词为主题、首句为立场、元记录正则过滤（补记/无因果/待核对）。
- **对比**：同主题时间排序 → 相邻对 + 首尾对（去重）→ 分类学五类；
  只有 `true_change` / `parallel_stance` 成为候选；相邻对承担"转折点定位"。
- **显式对比句**：`(以前|过去|曾经|原本|当初)…[，,]…(现在|如今|这两年|最近)` 正则直取，
  置信度 0.85、`signal_type=explicit_contrast`，主题取笔记标签。
- **评分**：`变化置信度 × 主题重要度(快照数归一) × 新颖度(7天内展示过→0.3) × 反应权重(属实×1.15/不是×0.6/无聊×0.8, 上限2.0)`。
- **呈现**：限额 3–5 条/周；两端原话+日期并排；负反应候选直接沉底（同一伤口不揭第二次）。

## 5. Agent Harness

- **预算四护栏**：最大步数(10)/工具调用(16)/整体时长/重复调用(>2 拒绝)；无进展 = 连续出错观察。
- **协议合规**：批量 tool_calls 的每个 id 必须有应答（否则 provider 400）；
  预算耗尽 → 注入"立即基于已有观察作答"强制收敛，而不是静默丢弃。
- **引用守卫**：`[A{id}]` 必须属于本轮工具真实返回；越界 → 一次无工具改写；仍失败 → 本地确定性降级（伪引用永不到达用户）。
- **降级链**：provider 错误/预算耗尽/引用失败 → `local_fallback`（同一投影协议，backend 可审计）。
- **运行时工具收缩**：主题明确（主题词存在且不全为发现类元词）→ 不注册 `discover_cognitive_shifts`；
  这是注册表级收缩，不靠提示词自觉。
- **结构化答案**：`CognitiveAnswer`（端点/区间/正反例/未知项/置信度/最多一问）；
  `traced_change` 必须双端点+可定位引用；较近端点固定 `latest_memory_candidate`。

## 6. 判定与反应（用户是唯一 ground truth）

- 六类判定（基本准确/部分准确/不构成变化/不是我的观点/证据不足/暂不判断）沉淀为主题记忆；
  加载用主题词交集匹配（"自主"↔"自主判断"，个人规模全表过滤可行）；
  被否认的解释写入 counter_evidence 且不再作为结论提出（e2e 测试覆盖）。
- 一键反应（属实/不是/无聊）写回 `candidate_reactions`，是发现排序的个性化因子；
  评审与对话判定共用主题记忆，语义一致。

## 7. 评测体系

| 评测 | 隔离性 | 度量 | 产物 |
|---|---|---|---|
| `eval-agent` 协议消融 | 一次性临时库 + 强制合成 Vault | 12 用例 × 3 配置：完成/引用/反例/拒答/判定遵守/无因果 | artifacts/evals/agent_comparative |
| `eval-retrieval` | 只读查询 | recall@5 / MRR，按 bm25/vector/hybrid 归因；合成 15 例 + 私有真实库 14 例（仅发布聚合指标） | artifacts/evals/retrieval |
| `eval-discovery` | 每次清空快照重建（确定性基线） | 预期对召回 / 措辞深化泄漏 / 判定口径查准 | artifacts/evals/discovery |
| 真实烟测 | 手动（需 key） | 完整工具循环 + 引用边界 + private_vault_sent | .local/agent_runs |
| `verify-vault` | — | Vault 只读 + 同步幂等 | stdout JSON |

诚实声明：合成/确定性评测证明**协议约束力**（不幻觉引用、该拒答就拒答、遵守用户修正），
不证明真实用户的长期受益；后者由"自用四周 + 属实且有意思比例"判据接管。

## 8. 技术取舍

- 显式状态机 + 单 Agent，不上多 Agent/编排框架：个人工具，可审计优先；
- SQLite 单文件（WAL），无向量库服务：当前个人知识库规模下使用暴力余弦检索，接口保留升级位；
- FTS5 trigram 而非 jieba 分词索引：免维护词典、中文召回稳，短词缺口由混合检索第二路兜底；
- Web UI 单文件 FastAPI + 内联 HTML：产品面刻意小，核心是 CLI/API/MCP 三通道；
- MCP 2.x `MCPServer`：工具经同一 `ToolRegistry` 暴露，边界规则零分叉。
