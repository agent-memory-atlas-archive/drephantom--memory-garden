# Memory Garden — 个人认知回溯 Agent

> **English TL;DR:** A personal cognitive-retrospect Agent over long-term Obsidian notes. It discovers when your stated views actually changed, verifies claims against read-only sources with citable evidence, and refuses to invent causal stories — backed by 48 tests, three reproducible eval suites, and 8 read-only MCP tools. The Quick Start below runs offline on a bundled synthetic vault, no API key needed.

面向个人长期 Obsidian 记录的**认知回溯 Agent**：主动发现观点、判断与选择发生变化的候选，
用受限的只读工具核对原始来源、追踪时间区间内的经历并检验支持与反例；
证据不足时向用户提问，而不是为用户虚构一个完整的因果故事。

```
 Obsidian Vault（永远只读，哈希校验）
        │  只读同步：来源/修订链/事件时间与记录时间分离/作者归属
        ▼
 SQLite + FTS5(trigram) ──► 混合检索（BM25 + 字符n-gram向量 + RRF 融合）
        │                        │
        ▼                        ▼
 立场快照管道（离线 LLM 抽取）   认知工具（8个只读，MCP 协议同源暴露）
        │                        │
        ▼                        ▼
 发现引擎（变化分类学+显式对比句）◄── 单 Agent Harness（预算/引用守卫/结构化答案）
        │                        │
        ▼                        ▼
 候选呈现（限额+新颖度）──► 用户反应/判定 ──► 写回排序权重与判定记忆
```

## 认知回溯的价值：三件人自己做不到的事

"帮你回忆"是伪需求——笔记检索已经够用。认知回溯立得住的价值，是三件人自己做不到的事：

1. **对抗记忆的自我美化**。记忆会不断重写叙事，你以为自己"一直都是这么想的"。
   能戳穿它的只有带时间戳的原始记录，而人很少主动去翻一年前自己对同一件事的原话。
2. **发现未察觉的变化**。人只能带着主题去问；Agent 能在没有主题时扫出
   "你 3 月写过 X，8 月写过 Y——这是变化吗"。这类盲区只有外部视角能提供。
3. **拒绝虚构因果**。大模型最擅长编故事，恰好是这件事最不需要的能力：
   证据不足就提问、区间内事件 ≠ 原因、反例必须主动去找——防御姿态本身就是产品。

## 为什么不是又一个"笔记问答机器人"

大模型擅长比较文本，但它不能仅靠自身：发现尚未提供给它的私人纵向记录；证明某句话来自
哪个文件和哪一天；区分"重复记录、并行表述与真实转变"；知道最近的记录是否仍代表现在的你；
主动检索被当前叙事遗漏的反例。Memory Garden 把这些做成**数据边界与证据协议**：

- **候选 ≠ 结论**：措辞差异只是"值得核对"，确认只能来自用户；
- **较近记录 ≠ 当前观点**：最近一条永远标注 `latest_memory_candidate`，必须问用户；
- **时间相邻 ≠ 因果**：区间内事件只标 `within_interval`，回答里禁止因果断言；
- **用户判定 > 模型结论**：被否认的解释会被记录且不再复用（有端到端测试）。

## 发现：从运行时扫描到数据结构问题

Memory Garden 把发现拆成离线/在线两层，让语义比较成为一等公民：

1. **立场快照（离线）**：每条带日期的个人记录经 LLM 抽取为
   `{主题, 立场, 原文引用, 语气强度, 是否本人观点}`；"无立场"是合法输出（引用/草稿/事件记录不硬造快照）；
   快照带 schema 版本，提示词迭代可整版重抽；LLM 不可用时降级为确定性抽取。
2. **变化分类学（离线/低频）**：同主题快照按时间**相邻对比 + 首尾对照**（防慢漂移漏检），
   分类为措辞漂移/深化/真变化/并列新立场/语境性立场——只有真变化与并列新立场成为候选。
3. **显式对比句（高精度信号）**："以前我以为…现在…"正则直取，作者亲口承认的变化直接成为候选。
4. **评分限额（在线）**：候选分 = 变化置信度 × 主题重要度 × 新颖度（近 7 天展示过则降权）× 反应权重，
   每周只打扰 3–5 条；呈现两端**原话+日期并排**，让差距自己说话，不解释因果。
5. **反应写回**：一键反应（属实/不是/无聊）直接写回该主题的排序权重——四周的个人校准优于任何通用规则。

## 评测（全部产物可复现，`artifacts/evals/`）

**Agent 协议消融**（12 个合成对抗用例 × 3 配置，离线确定性 provider，`eval-agent`）：

| 配置 | 任务完成 | 引用有效 | 反例覆盖 | 拒答正确 | 判定遵守 | 无因果断言 | 通过率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_shot_baseline | 0.75 | 1.0 | — | 0.58 | 0.00 | 0.00 | 0.33 |
| agent_without_feedback | 0.92 | 1.0 | 1.0 | 1.0 | **0.00** | 1.0 | 0.92 |
| **full_agent** | **1.00** | 1.0 | 1.0 | 1.0 | **1.00** | 1.0 | **1.00** |

消融设计：`agent_without_feedback` 与 `full_agent` 唯一差异是是否加载用户判定记忆——
恰好只有 `feedback_adherence` 一项翻转（0→1），证明反馈闭环是独立起作用的机制而非整体加成。

**检索双路指标**（人工标注 golden 集，`eval-retrieval`）：

| 语料 | BM25 recall@5 / MRR | 向量 | 混合 RRF |
|---|---|---|---|
| 合成对抗库（15 例） | 0.933 / 0.822 | 1.000 / 0.967 | **1.000 / 1.000** |
| 私有真实库（14 例，仅发布聚合指标） | 0.929 / 0.893 | 1.000 / 0.964 | **1.000 / 0.929** |

BM25 的 trigram 分词器无法命中 <3 字中文词（如"独处""边界"），
字符 2/3-gram 哈希向量与 LIKE 兜底补齐了这部分召回——混合 > 单路，差距可归因。

**发现精度**（合成库，`eval-discovery`，确定性抽取基线）：
预期变化对查全 **1.0**；"措辞深化"对泄漏 **0**；判定口径查准 **0.8**
（唯一的假阳性来自事件型记录，LLM 抽取的 has_stance 语义过滤可消除，确定性模式保留为诚实基线）。

**真实模型烟测**：DeepSeek `deepseek-v4-flash` 完整工具循环（4 步 / 13 次只读调用 / 28.2s / 正常收敛），
`private_vault_sent=false`；回答带可定位引用、区分端点、明确"近期候选需确认"。

诚实边界：以上是工程与协议行为的度量，不证明真实用户长期受益率；
发现质量的最终判据是"自用四周，标记'属实且有意思'的候选比例"。

## 她：知微

助手默认叫**知微**——取自《易经》"知微知彰"：在变化显形之前，先看见它的微光。
名字可在网页设置页（右上角"设置"）里改；性格在根目录 `soul.md` 里改，**每条消息热加载，改完即生效**。

判定按钮与"还没确定的"提示只在**结论型回答**（追溯到的变化/明确不构成变化）后出现；
情感陪伴类的回复不会被一排表单打断。

开源给别人用时，对方无需改任何文件：网页右上角"设置"里填 Vault 路径与 API Key
（存本机 `.local/settings.json`，已被 gitignore），重启一次即生效。

## Quick start

无自己的 Vault 也能跑：设 `MG_PUBLIC_DEMO_MODE=true`，`init` 会强制使用仓库内合成 Vault（`evals/cognitive_mvp_vault`），开箱即用、离线可复现。

```powershell
git clone https://github.com/drephantom/memory-garden.git
cd memory-garden
Copy-Item .env.example .env        # 指向你的 Obsidian Vault；配 LLM 则填 base_url/model/key 文件
uv sync
uv run memory-garden init               # 只读同步 + 向量
uv run memory-garden extract-snapshots  # 离线立场快照（自动选择 LLM/确定性）
uv run memory-garden ask "自主判断这个主题，我的想法以前到现在有没有变化？"
uv run memory-garden discover           # 全库发现候选（限额呈现）
uv run memory-garden serve              # 本地 Web UI (127.0.0.1:8766)
```

评测与验证：

```powershell
uv run pytest
uv run ruff check src tests
uv run mypy src/memory_garden
uv run memory-garden eval-agent          # 协议消融评测（隔离临时库，离线可复现）
uv run memory-garden eval-retrieval      # 检索双路指标
uv run memory-garden eval-discovery      # 发现精度
uv run memory-garden verify-vault        # Vault 只读 + 同步幂等校验
```

## MCP：同一套工具，两个世界

`uv run memory-garden mcp` 以 stdio 启动 MCP Server，把 8 个认知工具暴露给任意 MCP 客户端
（Claude Desktop 等）。工具边界（只读、"已发现来源"才能 read、发现工具的运行时收缩）
与内部 Agent 循环完全同源——不是两套实现。

## 目录

```
src/memory_garden/
  config.py      设置加载（进程环境 > .env）
  db.py          SQLite schema v2 + 增量迁移（来源/原子/快照/对话/判定/发现）
  importer.py    只读同步：哈希修订链、移动身份稳定、event_time/recorded_at 分离、作者归属
  retrieval.py   BM25(trigram) + 字符n-gram向量 + RRF 混合，离线确定性降级
  llm.py         OpenAI 兼容客户端（工具循环/embeddings/重试/密钥脱敏）
  tools.py       8 个只读认知工具（含全库发现工具的运行时收缩）
  agent.py       Agent Harness：预算护栏/引用守卫/一次有界修复/本地确定性降级/结构化答案
  snapshots.py   立场快照管道 + 变化分类学 + 显式对比句 + 发现引擎（评分四因子）
  cognitive.py   发现扫描持久化、呈现追踪、一键反应、六选一判定记忆
  evaluation.py  协议消融评测 / 检索双路指标 / 发现精度
  mcp_server.py  MCP stdio 服务（工具与内部循环同源）
  web.py         极简本地 UI（FastAPI 单文件，无前端构建链）
  cli.py         init/sync/ask/discover/extract-snapshots/eval-*/serve/mcp/verify-vault
tests/           48 项测试（导入/检索/工具/Agent/快照/评测/MCP/Web 全链路）
evals/           合成对抗 Vault + 检索 golden 集（合成 15 例；真实集含私人标题，仅本地）+ 用例集
docs/            ARCHITECTURE.md / DEMO.md
```

## 已知边界（诚实清单）

- 发现质量依赖"同一主题被反复带日期地记录"——写作越稀疏，沉默信号与对比信号越弱；
- 主题实体归并目前是"词面 + LLM 沿用已知主题表"的轻量版，完整聚类与别名评测在 roadmap；
- 混合检索的 MRR 在私有真实库上略低于单向量路（RRF 融合会稀释第一名），换召回不换排序是当前取舍；
- 判定记忆的主题匹配用词交集（个人规模可行），不是向量语义匹配；
- 云端模式会把问题与工具筛出的少量片段发给配置的 provider；Vault 本身永远只读且不出本机。

## 长期愿景（展望，不在当前路线图内）

现在，这里只有本地的 Obsidian 记录。长期的想象是：每个人都有一座自己的记忆花园——
人与人的交流经由各自的 Agent 发生，互动留下"记忆的种子"，持续交互长成树与花园；
相似的主题让人找到同路人，私密的记忆由主人亲手交出。

这是"花园"这个名字的由来，也是第一阶段把单人系统与隐私边界做到极致的原因：
任何记忆离开本地，都必须由主人显式同意、可验证、可撤回——花园的入口，永远是自己。
