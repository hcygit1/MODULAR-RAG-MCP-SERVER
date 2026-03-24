# 项目技术知识库 — 面试官参考手册

> 本文件供面试官（AI Agent）使用，包含本项目的关键实现细节、高频面试题及参考答案。  
> **对齐版本**：与当前仓库 `src/` + `config/settings.yaml` 一致（2026-03）。若 DEV_SPEC 与代码不一致，**以代码为准**。

---

## 模块一：Hybrid Search 混合检索

### 核心实现
- **双路并行召回**：Dense（向量语义）+ Sparse（关键词/全文）同时执行
- **融合算法**：RRF（Reciprocal Rank Fusion）  
  公式：`Score = 1/(k + Rank_Dense) + 1/(k + Rank_Sparse)`，k 通常取 60（与 `config/settings.yaml` 中 `fusion_algorithm: rrf` 一致）
- **为什么用 RRF 而不是线性加权**：RRF 无需对不同路的分数值做归一化，对排名稳健，不依赖各路分数的绝对尺度
- **Dense Route**：Query Embedding → 向量相似度检索 → Top-N
- **Sparse Route（默认 SQLite 后端）**：**FTS5** 与 chunk 同库（`retrieval.sparse_backend: fts5` 时）；非 SQLite 后端时可能走其他稀疏实现。口语上仍可称「稀疏/关键词一路」，但不要硬说成「独立 pickle BM25 目录」除非候选人明确用的是旧架构

### 高频面试题

**Q: 为什么要做 Hybrid Search？**  
A: 稀疏一路擅长字面匹配（菜名、术语），稠密一路擅长语义与同义表达；RRF 融合平衡查全与排序稳定性。

**Q: RRF 公式里 k=60 是怎么来的？**  
A: 平滑因子，常见经验值来自文献与实践；调大 k 头部优势减弱，调小则区分度更大。

**Q: 稀疏索引存在哪里？**  
A: **`vector_store.backend: sqlite` 时**：FTS5 虚拟表与向量、chunk 在同一 SQLite 文件（`vector_store.sqlite_path`），不是单独的 `data/db/bm25/` pickle 方案。**其他 backend** 以工厂实际实现为准。

---

## 模块二：Reranker 精排

### 核心实现
- **两段式架构**：粗排（Hybrid）→ 精排（可选）
- **三种后端**（`rerank.backend`）：`none` | `cross_encoder` | `llm`
- **Graceful Fallback**：精排失败/超时回退到 RRF 顺序（见 `RerankerOrchestrator`）

### 高频面试题

**Q: Cross-Encoder 和 Bi-Encoder 的区别？**  
A: 同前：Bi-Encoder 适合大规模召回；Cross-Encoder 适合小候选集精排。

**Q: 默认配置下精排开吗？**  
A: 常见默认 `rerank.backend: none`，此时以 RRF 结果为主；开启 cross_encoder 时注意候选规模与超时。

---

## 模块三：Ingestion Pipeline 数据摄取流水线

### 两条入口（面试必区分）
1. **`IngestionPipeline.process(file_path, ...)`**（如 `scripts/ingest.py`）：**integrity → load（如 PdfLoader）→ split → transform → encode → store**
2. **`IngestionPipeline.process_document(document, ...)`**（如 MCP `ingest_document_*`）：**split → transform → encode → store**；Document 已在工具层加载，且 MCP 在工具层先做 **SHA256 + should_skip + force**

### 阶段要点
- **Load**：PDF 常用 **MarkItDown（PdfLoader）**；另有 **MinerU 云端** 路径适配为 `Document` 后走 `process_document`
- **Split**：**`splitter_strategy: recursive | heading`**；heading 场景产出 **`parent_id`** 等结构 metadata，供父聚合检索
- **Transform**（串行）：ChunkRefiner → MetadataEnricher → ImageCaptioner（均由 `ingestion.*` 开关控制是否启用 LLM/Vision）
- **Embed**：Dense + Sparse 双路；具体稀疏编码与存储后端一致
- **Store**：**`VectorUpserter.replace_document_chunks`**：存在 **`metadata.source_doc_id`** 时先 **`delete_by_source_doc_id`** 再 upsert，避免文档改短后旧 chunk 残留

### 幂等性
- **文件级**：`FileIntegrityChecker`（默认库路径 **`cache/processing/file_integrity.db`**，表 `ingestion_history`），与 MCP **`force`** 语义配合
- **文档级替换**：依赖稳定的 **`source_doc_id`**（chunk metadata 来自 `document.id` 等）；**不是**简单用「UUID 随机 chunk_id」解释幂等

### 高频面试题

**Q: CLI 和 MCP 入库为何一个像「从 0 开始」一个像「从 split 开始」？**  
A: CLI 只给路径，用 `process` 一条龙；MCP 要在协议层返回 skip/结构化字段，并支持 PDF/MD/TXT 不同加载，故工具层先 integrity + 得到 `Document`，再 `process_document`。

**Q: 图片怎么进检索与响应？**  
A: Caption 等进入 chunk 文本或 metadata；检索命中后，**SQLite 统一存储**下常从 **`images` 表** 按 `sqlite_path` 加载，`build_mcp_content` / `MultimodalAssembler` 组装 **`content` + `structuredContent.citations`**（条件：`vector_store.backend == sqlite` 且 `sparse_backend == fts5` 等，见 `query_knowledge_hub`）。

---

## 模块四：可插拔架构

### 核心设计
- **工厂 + Base**：Embedding / VectorStore / Reranker / LLM 等按 `config/settings.yaml` 选择实现
- **配置驱动**：换 backend 主要改 YAML + 必要时新增 `libs` 下实现并在 Factory 注册

### 当前支持（以 settings 注释与代码为准）
- **LLM / Vision**：含 **Qwen / DashScope** 等（见 `src/libs/llm/`）
- **Embedding**：含 **Qwen、OpenAI、Ollama、local** 等
- **Vector Store**：**sqlite（推荐统一存储）**、chroma、qdrant、pinecone 等
- **Reranker**：none / cross_encoder / llm

---

## 模块五：MCP 协议集成

### 暴露的 Tools（当前代码 **4 个**）
| 工具名 | 功能 | 关键参数 |
|--------|------|---------|
| `query_knowledge_hub` | 检索，Hybrid + 可选 Rerank + 可选 Parent 聚合 | `query`, `top_k?`, `collection_name?` |
| `list_collections` | 列举集合 | （无必填） |
| `ingest_document_normal` | 本地 PDF / MD / TXT 入库 | `file_path`, `collection_name?`, `force?` |
| `ingest_document_mineru` | MinerU 云端解析 PDF 入库 | `file_path`, `collection_name?`, `force?` |

> **注意**：`get_document_summary` 若出现在 DEV_SPEC 中为规划项，**当前 `src/mcp_server/server.py` 未注册**，面试中勿当作已实现能力。

### 检索返回形态
- 顶层：**`content`**、**`structuredContent`**（如 **`citations`**）、**`isError`**
- 注册位置：`src/mcp_server/server.py`（FastMCP + stdio）

---

## 模块六：文档更新与一致性（替代旧「DocumentManager」叙述）

### 当前实现重点
- **无独立 DocumentManager 模块**与题库旧版「四路删除」描述对齐；删除/更新以 **按 `source_doc_id` 替换 chunk** + **文件 integrity** 为主
- **风险点**：CLI 与 MCP 的 **skip/force 分层不同**，但最终 **store 层 replace** 一致

### 高频面试题

**Q: 文档修改后如何避免旧块污染检索？**  
A: 有 `source_doc_id` 时 **`replace_document_chunks`**：先删该文档下旧 chunk，再写入新块。

---

## 模块七：可观测性与追踪系统

- **Trace**：`TraceContext` + `logs/traces.jsonl`（由 `observability.logging.log_file` 配置）
- **Dashboard**：Streamlit（`src/observability/dashboard/`）
- **评估面板**：若展示 Ragas，需说明 **当前 evaluator 中 ragas backend 可能未实现**（以 `EvaluatorFactory` 为准）

---

## 模块八：评估体系

- **脚本/模块**：`scripts/evaluate.py`、`src/observability/evaluation/`（如 `content_evaluator`、`eval_runner`、`e2e_runner`）
- **Golden Test Set**：`tests/fixtures/golden_test_set.json`（配置项 `evaluation.golden_test_set`）
- **Ragas**：配置里可出现 `evaluation.backends: [ragas, custom]`，但 **工厂内 ragas 可能仍为 NotImplementedError** —— 面试要问「是否已落地」而非默认已跑 Ragas

### 指标（概念仍可用）
- Hit Rate@K、MRR 等仍可作为通用讨论；与 **当前自定义/content 评估** 的关系要说清

---

## 模块九：工程化实践

### 测试体系
- **分层**：`tests/unit/`、`tests/integration/`、`tests/e2e/`
- **策略**：mock LLM/外部 API，关键路径单测（如 heading splitter、parent aggregator、sqlite 行为等）以仓库为准
- **勿编造**：不要使用「1198+ 单测」等未经验证的数字

### 持久化（默认 SQLite 统一存储）
| 对象 | 典型位置 | 用途 |
|------|----------|------|
| 向量 + chunk + FTS5 + images | `vector_store.sqlite_path`（如 `./data/db/rag.sqlite`） | 统一检索与多模态 |
| 文件完整性记录 | `cache/processing/file_integrity.db` | SHA256 处理历史 |

**其他 backend**（Chroma/Qdrant）时路径与行为以 `settings.yaml` 为准。

---

## 常见「露馅」警示点（更新）

| 简历描述 | 深挖问题 | 露馅信号 |
|---------|---------|---------|
| 「三个 MCP Tool」 | 第四个是什么？ingest 有几条路径？ | 说不出 normal/mineru |
| 「BM25 pickle 在 data/db/bm25」 | sqlite 后端稀疏一路是什么？ | 与 FTS5 同库事实不符 |
| 「DocumentManager 四路删除」 | 当前文档更新怎么保证？ | 说不出 replace_document / source_doc_id |
| 「Ragas 已集成」 | EvaluatorFactory 里 ragas 状态？ | 与 NotImplementedError 矛盾 |
| 「只用 Recursive 切分」 | heading 与 parent_id？ | 不知 `HeadingSplitter` / 父聚合 |
| 「ingestion_history 在 data/db」 | FileIntegrityChecker 默认路径？ | 说不清默认 sqlite 路径 |
