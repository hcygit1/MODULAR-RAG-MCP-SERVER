# 面试报告模板 — 面试官使用手册

> 本文件在 Phase 3 报告生成时读取。包含：①完整报告 Markdown 模板 ②12 道题预置参考答案 ③评分细则。

---

## 一、报告 Markdown 模板

生成规则：
- 表格"参考答案"列使用 `[→ 查看](#a-锚点关键词)` 锚链接，指向本文件第二节对应答案
- 只为**本次实际问到的题目**复制对应答案块，未问到的不放入报告
- 严格按评分细则打分，不得因情绪照顾调分

```markdown
# 模拟面试报告

**项目**：Modular RAG MCP Server
**面试时间**：{datetime}
**评分**：{score}/10

---

## 一、面试记录

> ✅ 答对核心要点 | ⚠️ 方向正确但细节缺失 | ❌ 未能答出或方向错误

### 方向 1：项目综述

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {2-3 句摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 2：简历深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 露馅 | 参考答案 |
|-----|------|-------------|------|-----|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | 是/否 | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 3：技术深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

---

## 二、参考答案

> 仅复制本次实际问到的题目对应答案块，保留 <a id> 锚点。

{从下方"预置参考答案库"按需复制}

---

## 三、简历包装点评

### 包装合理 ✅
- **"{简历描述}"**：{说明候选人能自圆其说之处，具体指出哪句回答支撑了该判断}

### 露馅点 ❌
- **"{简历描述}"** → {面试中的具体表现}。**严重性：高/中/低**（{说明原因}）

### 改进建议
- {针对每个露馅点的具体、可操作建议，如"建议背下 RRF 公式并能解释 k 参数含义"}

---

## 四、综合评价

**优势**：
- {具体到哪道题答得好、好在哪个关键点}

**薄弱点**：
- {具体技术点 + 答错/答浅的表现描述}

**面试官建议**：
{针对每个薄弱点的具体改进方向，避免笼统表述}

---

## 五、评分

| 维度 | 分数（满分 10）| 评分依据（必须说明具体扣分原因） |
|-----|--------------|--------------------------------|
| 项目架构掌握 | x | {哪些点答到了，哪些点缺失} |
| 简历真实性 | x | {几处包装合理，几处露馅，差距} |
| 算法理论深度 | x | {RRF/Cross-Encoder/评估指标等作答情况} |
| 实现细节掌握 | x | {source_doc_id 替换/MCP 四工具/可插拔三步骤/sqlite+FTS5/CLI vs MCP 入库} |
| 表达清晰度 | x | {回答完整性、逻辑清晰度、因果说明} |
| **综合** | **x** | {加权说明} |
```

---

## 二、预置参考答案库

> 按需复制到报告"二、参考答案"节，保留 `<a id>` 锚点不变。

---

### <a id="a-项目架构"></a>Q: 介绍项目整体架构和你具体负责的部分

**参考答案**：
整体可分四层（常说 core / ingestion / libs，再加 MCP 边界）：
1. **libs**：可插拔基础设施（Embedding / VectorStore / LLM / Splitter / Reranker 等 + Factory）
2. **ingestion**：入库流水线（CLI：`process` 含 integrity+load；MCP：`process_document` 前在工具层 skip/load；均经 split → transform → encode → store；支持 **replace_document_chunks**）
3. **core**：检索编排（Hybrid、RRF、可选 Rerank、可选 Parent 聚合）与响应构建
4. **mcp_server**：FastMCP + stdio，对外 **4 个 Tool**（query / list / ingest_normal / ingest_mineru）

核心亮点：配置驱动组装；默认推荐 **SQLite 统一存储**（向量 + chunk + **FTS5** + images）；菜谱等结构化场景可用 **heading + 父聚合**。

---

### <a id="a-ingestion链路"></a>Q: Ingestion 链路有哪些阶段？

**参考答案**：
**CLI `process(file_path)`**：integrity → Load（如 PdfLoader）→ Split → Transform → Encode → Store。  
**MCP `process_document`**：工具层先做 integrity/skip 与加载得到 Document → Split → Transform → Encode → Store。

1. **Split**：`recursive` 或 **`heading`**（产出 `parent_id` 等）；见 `SplitterFactory` / `HeadingSplitter`
2. **Transform**（串行，受配置开关控制）：ChunkRefiner → MetadataEnricher → ImageCaptioner
3. **Embed**：Dense + Sparse，与所选 VectorStore 一致
4. **Store**：有 **`source_doc_id`** 时优先 **`replace_document_chunks`**（先 `delete_by_source_doc_id` 再 upsert），避免旧块残留

---

### <a id="a-mcp协议"></a>Q: MCP 是什么规范？暴露了哪些 Tool？

**参考答案**：
MCP（Model Context Protocol）基于 JSON-RPC 2.0；本项目 **Stdio Transport**（stdout 仅 MCP 消息，日志 stderr）。

当前 **`src/mcp_server/server.py` 注册 4 个 Tool**：

| Tool | 功能 | 关键参数 |
|------|------|---------|
| `query_knowledge_hub` | Hybrid 检索 + 可选 Rerank + 可选父聚合 | `query`, `top_k?`, `collection_name?` |
| `list_collections` | 列举集合 | （无必填） |
| `ingest_document_normal` | PDF/MD/TXT 本地解析入库 | `file_path`, `collection_name?`, `force?` |
| `ingest_document_mineru` | MinerU 云端 PDF 入库 | `file_path`, `collection_name?`, `force?` |

检索返回：`content`、`structuredContent`（如 `citations`）、`isError`；满足条件时从 SQLite **images** 表组装多模态内容。

---

### <a id="a-rrf公式"></a>Q: RRF 融合公式是什么？k 值含义？为什么不用线性加权？

**参考答案**：

$$Score_{RRF}(d) = \frac{1}{k + Rank_{Dense}(d)} + \frac{1}{k + Rank_{Sparse}(d)}$$

- **k 的含义**：平滑因子，防止排名头部文档分数被过度高估。k = 60 是 Cormack et al. 2009 论文的经验推荐值；调大 k → 分布更均匀，调小 k → 差异更大。
- **为什么不用线性加权**：BM25 分数无上界，余弦相似度在 [-1,1]，两路量纲不同，线性加权必须先归一化且引入额外超参。RRF 只依赖排名（序数信息），天然无需归一化，鲁棒性更强。

---

### <a id="a-cross-encoder"></a>Q: Cross-Encoder 和 Bi-Encoder 的区别？为什么不能做粗排召回？

**参考答案**：

| | Bi-Encoder | Cross-Encoder |
|--|-----------|--------------|
| 编码方式 | Query 和 Document **分别**编码为向量，算相似度 | Query 和 Document **拼接**一起输入模型，联合建模 |
| Document 向量 | 可**离线预计算**，查询时 O(1) | 每对 (Query, Chunk) 必须**实时推理**，O(n) |
| 精度 | 较低（无交互） | 更高（充分建模交互特征） |
| 适合场景 | 粗排召回（大规模） | 精排（10-30 条小候选集） |

**Cross-Encoder 不能做粗排**：5000+ 文档场景每次查询需推理 5000 次，延迟不可接受、成本极高。必须先用 Bi-Encoder 粗召回 Top-N，再用 Cross-Encoder 精排。

---

### <a id="a-chunkrefiner"></a>Q: ChunkRefiner 做了什么？为什么需要额外的 LLM 步骤？

**参考答案**：
`RecursiveCharacterTextSplitter` 按字符边界物理切分，会将语义连续的段落切断（如"问题背景"和"解决方案"分入不同 Chunk），导致检索命中的 Chunk 缺乏上下文。

ChunkRefiner 的工作：
1. **合并语义断裂的段落**：LLM 判断相邻 Chunk 是否逻辑连续，若是则合并
2. **去噪清理**：移除 PDF 转换产生的页眉页脚乱码、重复标题

使每个 Chunk 成为 **Self-contained 的语义单元**，提升检索精度和 LLM 生成质量。

---

### <a id="a-hit-rate"></a>Q: Hit Rate@K 是怎么计算的？

**参考答案**：

$$HitRate@K = \frac{\text{Top-K 结果中至少命中一条 Golden Answer 的查询数}}{\text{总查询数}}$$

对 Golden Test Set 中每条 `(query, expected_chunks)`，取 Top-K 检索结果，至少一条匹配则 hit=1，否则 hit=0。Hit Rate@K = 命中次数 / 总 case 数。

**@K 含义**：只要正确文档出现在 Top-K 内即算命中，不要求排第一。@10 = 正确文档在 Top-10 内即可。

---

### <a id="a-可插拔架构"></a>Q: 新增一个 Embedding Provider 需要改哪些文件？

**参考答案**：
只需改 **3 处**，已有代码零修改（开闭原则）：

1. **新建** `src/libs/embedding/your_provider.py`：继承 `BaseEmbedding`，实现 `embed_texts()` 等接口方法
2. **修改** `src/libs/embedding/factory.py`：在 `provider_map` 中注册 `"your_provider": YourProviderClass`
3. **修改** `config/settings.yaml`：将 `embedding.provider` 改为 `"your_provider"`

其他组件（LLM / Reranker / VectorStore / Loader / Splitter）遵循同一套三步流程。

---

### <a id="a-幂等性"></a>Q: 幂等与文档更新在项目里怎么体现？

**参考答案**：
- **文件级**：`FileIntegrityChecker`（默认 **`cache/processing/file_integrity.db`**），`compute_sha256` → `should_skip` → 成功后 `mark_success`；MCP/CLI 支持 **`force`** 绕过跳过。
- **文档级**：chunk metadata 带 **`source_doc_id`**（来自 `document.id` 等）时，**`VectorUpserter.replace_document_chunks`**：先 **`delete_by_source_doc_id`** 再写入新块，避免文档改短后旧 chunk 仍被检索。
- **chunk id**：Pipeline 内按文档 id + 序号等生成稳定 chunk id（见 `_generate_chunk_id`），配合 upsert；与「纯 UUID 随机」不是同一叙事。

---

### <a id="a-多模态检索"></a>Q: 图片的 Caption 如何参与检索？检索命中后图片怎么返回？

**参考答案**：
1. **摄取**：ImageCaptioner（Vision LLM，配置可为 Qwen-VL 等）生成描述，进入 chunk 文本或 metadata；图片字节可进统一存储
2. **检索**：与正文同一套 Dense/Sparse（SQLite 下稀疏为 **FTS5**）
3. **返回**：`query_knowledge_hub` → `build_mcp_content`；当 backend 为 **sqlite** 且 sparse 为 **fts5** 时，从 **`sqlite_path` 的 `images` 表** 加载，`MultimodalAssembler` 组装 **`content` + citations**，实现搜文出图

（旧版「image_index.db + 文件系统」若与当前代码不一致，以 SQLite 方案为准。）

---

### <a id="a-测试体系"></a>Q: 测试分几层？单元测试怎么 mock LLM？

**参考答案**：
三层金字塔：**Unit**（`tests/unit/`）→ **Integration**（`tests/integration/`）→ **E2E**（`tests/e2e/`）。

- 单测：对 LLM/外部 API 使用 **mock/patch**，覆盖 splitter、parent 聚合、store 行为、evaluator 等关键模块（以仓库现有用例为准）。
- 集成/E2E：按项目实际脚本与用例验证 Pipeline、MCP 子进程等。

**勿编造**固定用例数量；以 `uv run pytest` / CI 为准。

---

### <a id="a-评估体系"></a>Q: Hit Rate@K 和 MRR 怎么计算？Ragas Faithfulness 衡量什么？

**参考答案**：
- **Hit Rate@K**：见 [→ 查看](#a-hit-rate)

- **MRR（Mean Reciprocal Rank）**：

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

$rank_i$ 是第 $i$ 条查询中第一条正确结果的排名。衡量**头部排序质量**。

- **Ragas**：配置可出现于 `settings.yaml`，但 **`EvaluatorFactory` 中 ragas 可能尚未实现**；实际离线评估可关注 `scripts/evaluate.py`、`content_evaluator`、`eval_runner` 等路径。概念上 Faithfulness 仍可作为「回答是否贴合检索上下文」的讨论指标。

---

### <a id="a-文档级更新"></a>Q: 文档更新后如何保证索引一致？（替代旧「四路删除」题）

**参考答案**：
当前主线是 **按 `source_doc_id` 文档级替换**：`replace_document_chunks` → `delete_by_source_doc_id` → `upsert_chunks`，保证同一文档重入库后旧 chunk 不会残留（尤其文档变短时）。

文件级「是否跳过解析」由 **`FileIntegrityChecker`** 与 **`force`** 控制；与向量库内 chunk 是否最新是两层问题，面试需分开说。

（若候选人只答 DocumentManager / 四路 Chroma+BM25，与当前默认 SQLite 实现不符，应扣分或引导更新表述。）

---

### <a id="a-可观测性"></a>Q: Trace 是怎么实现的？Ingestion 的 5 个阶段各是什么？

**参考答案**：
**Trace 实现**：显式调用模式（非 AOP 拦截），各阶段手动向 TraceContext 写入耗时、数量、分数分布，存为 JSON Lines 结构化日志，零外部依赖（无需 LangSmith/LangFuse）。

**Ingestion 5 阶段**：Load → Split → Transform → Embed → Upsert

**Query Trace 5 阶段**：QueryProcess → DenseRecall → SparseRecall → Fusion → Rerank

Dashboard 展示：Query 追踪页面（Dense/Sparse 召回对比、Rerank 前后排名变化）、Ingestion 追踪（阶段耗时瀑布图）。

---

## 三、评分细则

**分档标准（严格执行，不得调整）**：

| 分档 | 标准 |
|-----|------|
| 9-10 | 所有核心问题答出关键细节，无露馅，表达清晰且有深度延伸 |
| 7-8 | 大部分问题答出主干，偶有细节遗漏（1-2 处），无严重露馅 |
| 5-6 | 架构层面基本掌握，但算法/实现细节有 3 处以上明显缺失，或有 1 处严重露馅 |
| 3-4 | 仅能描述表面概念，追问即露馅，简历存在明显虚报 |
| 1-2 | 核心技术点均无法解释，简历与实际能力严重不符 |

**5 个评分维度**：

| 维度 | 重点考察内容 |
|-----|------------|
| 项目架构掌握 | 三层架构、模块分工、可插拔设计能否清楚表达 |
| 简历真实性 | 量化指标有无测量方法支撑，强动词能否说清决策过程 |
| 算法理论深度 | RRF 公式、Cross-Encoder 原理、Hit Rate/MRR 计算 |
| 实现细节掌握 | source_doc_id 替换、MCP 四工具、可插拔三步骤、sqlite+FTS5、CLI vs MCP 入库入口 |
| 表达清晰度 | 回答完整性、逻辑链完整、能说清"为什么"而非只说"是什么" |
