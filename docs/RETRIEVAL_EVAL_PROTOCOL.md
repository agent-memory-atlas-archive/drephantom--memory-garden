# 检索评估协议

更新：2026-09-01。本协议用于避免 atom 切分、候选深度和指标命名造成的虚假提升。

## 评估单位与候选池

- 检索器按 atom 排序，但 golden 标注单位是 source path（整篇笔记）。评估先取 Top30 atom，
  再按 path 的首次出现位置去重，最后计算 Top5 source path 指标。
- 纯 Hybrid、Cross-Encoder-only 和 RRF + Cross-Encoder 排名融合固定使用相同的每路 Top30
  候选池。不能让基线取 Top10、重排取 Top30。
- `hybrid_rerank` 用 Cross-Encoder 排名替换 RRF 排名；`hybrid_rerank_fused` 对 RRF 名次与
  Cross-Encoder 名次再做等权 RRF。两个重排对照复用同一次 API relevance score，不重复计费，
  也不引入不同模型响应。
- Embedding 与 Rerank 的规范文本是“标题 + 标题层级 + 标签 + atom 正文”。

## 指标

- `HitRate@5`：Top5 是否至少包含一篇相关笔记。
- `Recall@5`：Top5 命中的相关笔记数 / 该查询全部相关笔记数。多个 relevant path 时不再把
  “至少命中一个”误称为 Recall。
- `Precision@5`：Top5 命中的相关笔记数 / 5。
- `MRR`：去重后的 source path 排名中，第一篇相关笔记的倒数名次；本实现的评估深度为 30。
- `nDCG@5`：二元相关性下的 Top5 折损累计增益，能反映多个相关笔记的相对位置。

`retrieval_metrics.json` 保存聚合指标；`retrieval_diagnostics.json` 只保存匿名 case 编号、相关项
数量、首个相关名次和逐路指标，不保存查询、路径、标题或正文。逐例 wins/ties/losses 用于判断
聚合变化是普遍趋势，还是少数查询移动造成。`retrieval_dataset_audit.json` 只记录 split 数量、
标注状态、类别分布和草稿 case id，并检查查询重复、路径存在性与 dev/test 路径泄漏。

## 数据集边界

- 公开合成 golden 与固定 mock embedding 只验证管线、缓存和指标实现，不代表真实模型效果。
- 私有 golden、查询、路径、标题、运行结果与逐条诊断保存在本机，不进入公开仓库。
- 助手生成的标注只能记为草稿；缺少 `annotation_status` 的旧条目同样视为未验证。
  只有 Vault 所有者逐条核对后才能显式标记为人工确认。
- 若本地划分 `real_dev` / `real_test`，相关路径不得跨 split；模型、候选深度、融合策略与文本模板
  只能依据 dev 选择。查看 test 后不得回头调参并把复跑结果当成首次 test。
- `--include-draft-goldens` 仅用于标注审计，草稿结果不能作为正式模型效果。
- 路径级标注不能证明具体 atom 都相关。若要评估重排对片段选择的真实贡献，需要另外标注
  query--atom 相关等级（例如 0/1/2），再报告 atom-level nDCG。

### 所有者复核清单

对 `evals/retrieval_goldens.real.json` 中每个草稿 case，依次核对：查询是否符合真实检索意图；
`relevant_paths` 是否遗漏同样能直接回答的笔记；是否误收只共享关键词但不能回答的笔记；多相关项
是否完整。只有确认后才把该 case 的 `annotation_status` 改为 `human_verified`。不要批量把全部状态
替换为已确认，也不要根据某个模型当前返回结果倒推 ground truth。

## 隐私

真实组始终使用一次性派生 SQLite，原始 Vault 只读。只有同时传入 `--real-models` 并显式开启
Embedding 与 Rerank 两个云端开关时，才发送授权范围内的查询与规范文本；公开 CI 不调用真实 API。
