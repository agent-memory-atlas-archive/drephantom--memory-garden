# Memory Garden — 个人认知回溯 Agent

> **English TL;DR:** A personal cognitive-retrospect Agent over long-term Obsidian notes. It discovers when your stated views actually changed, verifies claims against read-only sources with citable evidence, and refuses to invent causal stories. Automated tests and reproducible evaluation code are included; the Quick Start runs offline on a bundled synthetic vault with no API key.

面向个人长期 Obsidian 记录的**认知回溯 Agent**：主动发现观点、判断与选择发生变化的候选，
用受限的只读工具核对原始来源、追踪时间区间内的经历并检验支持与反例；
证据不足时向用户提问，而不是为用户虚构一个完整的因果故事。

```
 Obsidian Vault（永远只读，哈希校验）
        │  只读同步：来源/修订链/事件时间与记录时间分离/作者归属
        ▼
 SQLite + FTS5(trigram) ──► 可切换检索（BM25 / 字符哈希 / Embedding / RRF 混合）
        │                        │
        ▼                        ▼
 立场快照管道（离线 LLM 抽取）   认知工具（8个只读领域工具，MCP 同源暴露）
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

## 评测与验证（方法公开，运行结果不入库）

仓库公开评测代码、指标定义、脱敏合成用例和复现命令，但不提交生成的评测结果：

- `pytest` 覆盖同步、检索、Embedding 缓存、隐私开关、工具调用、引用守卫、MCP 与 Web 接口；
- `eval-agent` 使用隔离临时库和合成 Vault，比较单次回答、无判定记忆 Agent 与完整 Agent；
- `eval-retrieval` 按 source path 计算 HitRate/Recall/Precision@5、MRR 与 nDCG@5，并保持各路线候选深度一致；
- 公开 `mock` 仅用于验证Embedding接口、缓存和评测管线，不代表真实语义模型效果；
- 私有 golden、真实 API 运行结果和逐条诊断仅保存在本机，`artifacts/evals/` 已加入 `.gitignore`。

完整指标口径见 `docs/RETRIEVAL_EVAL_PROTOCOL.md`。这些验证只能证明工程链路和约束行为，
不能证明真实用户的长期受益；后者需要持续自用和用户反馈验证。

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
uv run memory-garden eval-agent          # 隔离临时库，输出仅保存在本机
uv run memory-garden eval-retrieval      # 脱敏合成用例，输出仅保存在本机
uv run memory-garden eval-discovery      # 发现精度
uv run memory-garden verify-vault        # Vault 只读 + 同步幂等校验
# 明确配置并开启两个云端检索开关后：只打印聚合运行元数据
uv run python scripts/run_private_api_rag_smoke.py --group real_dev --case-index 0 --source-limit 60
```

## 检索与 Embedding 配置

默认配置是 `MG_RETRIEVAL_MODE=hybrid`、`MG_EMBEDDING_BACKEND=local_hash`、
`MG_RERANKER_BACKEND=local_heuristic`：SQLite FTS5 BM25 与 512 维字符 2/3-gram
哈希向量先经 RRF 融合，再对最多 30 个候选做离线确定性重排，全程无需 API Key。
CLI 可在子命令前临时选择：

```powershell
uv run memory-garden --retrieval-mode bm25 ask "自主判断"
uv run memory-garden --retrieval-mode hash_vector ask "自主判断"
uv run memory-garden --retrieval-mode embedding ask "自主判断"  # backend 必须是 mock/api
uv run memory-garden --retrieval-mode hybrid ask "自主判断"
uv run memory-garden --retrieval-mode hybrid --reranker none ask "自主判断"  # RRF 基线
uv run memory-garden --embedding-backend api --retrieval-mode hybrid --reranker api ask "自主判断"
```

API Embedding 必须同时配置 `MG_EMBEDDING_BACKEND=api`、`MG_LLM_EMBEDDING_MODEL`、
`MG_EMBEDDING_PROVIDER/BASE_URL/API_KEY(_FILE)`，并显式设置
`MG_ALLOW_CLOUD_EMBEDDING=true`。它与 `MG_LLM_*` 生成连接相互独立，因此可同时使用
DeepSeek 生成与 SiliconFlow 检索；未填写专用地址/密钥时仍兼容回退到旧的 `MG_LLM_*` 配置。
启用后，系统会向 `/embeddings` 发送查询文本，以及每条检索原子的**标题、标题层级、标签、正文**；
不会发送文件路径，原始 Vault 仍只读，向量仅写入派生 SQLite。Web 启动不自动批量发送云端向量，
设置页提供带二次确认的“显式构建当前向量缓存”操作。`mock` 仅供 CI/测试。

SiliconFlow 示例（密钥文件放仓库外；不要把 Key 本身写进 `.env`）：

```dotenv
MG_EMBEDDING_BACKEND=api
MG_EMBEDDING_PROVIDER=siliconflow
MG_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
MG_EMBEDDING_API_KEY_FILE=D:/path/outside-repo/siliconflow.key
MG_LLM_EMBEDDING_MODEL=BAAI/bge-m3
MG_LLM_EMBEDDING_DIMENSION=1024
MG_ALLOW_CLOUD_EMBEDDING=false

MG_RERANKER_BACKEND=api
MG_RERANKER_PROVIDER=siliconflow
MG_RERANKER_BASE_URL=https://api.siliconflow.cn/v1
MG_RERANKER_API_KEY_FILE=D:/path/outside-repo/siliconflow.key
MG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
MG_RERANKER_FUSION=rank_fusion
MG_ALLOW_CLOUD_RERANK=false
```

两个允许开关故意保持 `false`；确认发送边界后再分别改为 `true`。

`MG_RERANKER_BACKEND=api` 另需配置 `MG_RERANKER_PROVIDER/BASE_URL/API_KEY(_FILE)/MODEL`
并显式设置 `MG_ALLOW_CLOUD_RERANK=true`。它会把查询与 RRF 候选的标题、标题层级、标签、
正文发送到 `/rerank`；不开此开关时工厂直接拒绝构建。`MG_RERANKER_FUSION=replace` 使用
cross-encoder 排名替换 RRF 排名，`rank_fusion` 则合并两种名次。默认 `local_heuristic` 仍是
离线、可审计的第二阶段规则基线，不冒充神经 cross-encoder。

完整链路为：查询 → BM25/Embedding 候选 → RRF → cross-encoder 重排（可选与 RRF 名次再融合）→ 证据注入 →
模型生成与引用守卫。代码支持不等于真实模型效果；只有真实请求成功且评估命令产生指标后，
才可描述为“使用真实模型完成评估”。

## MCP：同一套工具，两个世界

`uv run memory-garden mcp` 以 stdio 启动 MCP Server，把8个只读领域工具与1个完整回溯入口
`ask_garden` 暴露给任意 MCP 客户端
（Claude Desktop 等）。工具边界（只读、"已发现来源"才能 read、发现工具的运行时收缩）
与内部 Agent 循环完全同源——不是两套实现。

## 目录

```
src/memory_garden/
  config.py      设置加载（进程环境 > .env）
  db.py          SQLite schema v3 + 无损增量迁移（Embedding 缓存身份/来源/认知数据）
  importer.py    只读同步：哈希修订链、移动身份稳定、event_time/recorded_at 分离、作者归属
  retrieval.py   统一工厂：BM25 / 字符n-gram / mock或API Embedding / RRF 混合
  llm.py         OpenAI 兼容客户端（工具循环/embeddings/rerank/重试/密钥脱敏）
  tools.py       8 个只读认知工具（含全库发现工具的运行时收缩）
  agent.py       Agent Harness：预算护栏/引用守卫/一次有界修复/本地确定性降级/结构化答案
  snapshots.py   立场快照管道 + 变化分类学 + 显式对比句 + 发现引擎（评分四因子）
  cognitive.py   发现扫描持久化、呈现追踪、一键反应、六选一判定记忆
  evaluation.py  协议消融评测 / 检索与重排对照 / 发现精度
  mcp_server.py  MCP stdio 服务（工具与内部循环同源）
  web.py         极简本地 UI（FastAPI 单文件，无前端构建链）
  cli.py         init/sync/ask/discover/extract-snapshots/eval-*/serve/mcp/verify-vault
tests/           自动化测试（含 Embedding 缓存、API cross-encoder、评估口径、数据集隔离与三入口统一）
evals/           脱敏合成 Vault 与公开回归用例；私有 golden 不入库
docs/            ARCHITECTURE.md / DEMO.md
scripts/         显式授权的私有 API RAG 烟测（临时库，只打印聚合运行元数据）
```

## 已知边界（诚实清单）

- 发现质量依赖"同一主题被反复带日期地记录"——写作越稀疏，沉默信号与对比信号越弱；
- 主题实体归并目前是"词面 + LLM 沿用已知主题表"的轻量版，完整聚类与别名评测在 roadmap；
- 多路融合和第二阶段重排不保证优于最佳单路，必须在独立、人工复核的 golden 上分别比较；
- 默认重排是可审计的规则基线；API cross-encoder 只是可选实现，不能仅凭链路跑通宣称效果提升；
- 判定记忆的主题匹配用词交集（个人规模可行），不是向量语义匹配；
- 对话云端模式会发送问题与工具筛出的少量片段；API Embedding 会发送查询和标题/标签/正文；
  API Rerank 会发送查询和 RRF 候选的标题/标题层级/标签/正文。三者分别配置，两个检索云端开关默认关闭；Vault 文件
  本身永远只读。

## 长期愿景（展望，不在当前路线图内）

现在，这里只有本地的 Obsidian 记录。长期的想象是：每个人都有一座自己的记忆花园——
人与人的交流经由各自的 Agent 发生，互动留下"记忆的种子"，持续交互长成树与花园；
相似的主题让人找到同路人，私密的记忆由主人亲手交出。

这是"花园"这个名字的由来，也是第一阶段把单人系统与隐私边界做到极致的原因：
任何记忆离开本地，都必须由主人显式同意、可验证、可撤回——花园的入口，永远是自己。
