---
name: project-learner
description: "Interactive project learning coach via interview-style Q&A. Reads codebase and docs, dynamically generates interview questions per knowledge domain and sub-topic, conducts up to 4 follow-up rounds, scores answers, provides learning guidance with code/doc references, and persists progress. 10 domains × 3-5 sub-topics = 45 knowledge points for comprehensive interview coverage. Use when user says '学习项目', '了解项目', '检验项目', '项目学习', '面试准备', 'learn project', 'study project', 'review project', 'interview prep', 'knowledge check', or wants to understand/master the project through guided Q&A."
---

# Project Learner

Interactive interview-coach that helps users master this project through guided Q&A.

All user-facing interaction in **中文**. Internal instructions in English.

## Pipeline Overview

```
Discovery → Check History → User Intent → Select Domain → Select Sub-topic
→ Generate Question → Interactive Q&A (≤4 follow-ups) → Evaluate
→ Learning Guide → Persist Progress → Continue or End
```

---

## Phase 1: Project Discovery

Autonomously build project understanding. Do NOT ask user anything yet.

1. Read `DEV_SPEC.md` — project goals, architecture, tech stack, module design
2. Read `config/settings.yaml` — configuration system
3. List `src/` directory tree — module structure (core/, ingestion/, libs/, mcp_server/, observability/)
4. Read key entry points: `main.py`, `scripts/ingest.py`, `scripts/ingest_mineru.py`, `scripts/retrieve.py`
5. List `tests/` — testing strategy overview

Build an internal mental model covering these **10 Knowledge Domains**, each containing **3-5 Sub-topics** (知识点), totaling **45 interview knowledge points**:

### Domain & Sub-topic Map

| ID | 知识域 / 知识点 | Key Code Areas |
|----|----------------|---------------|
| **D1** | **RAG Pipeline 整体架构** | |
| D1.1 | 端到端数据流：从文档上传到生成回答的完整链路 | `DEV_SPEC.md`, `main.py`, `scripts/` |
| D1.2 | 三层架构设计：core/ingestion/libs 各层职责与依赖方向 | `src/core/`, `src/ingestion/`, `src/libs/` |
| D1.3 | Pipeline 组装：配置驱动的组件组合机制 | `main.py`, `config/settings.yaml`, `src/core/settings.py` |
| D1.4 | 核心数据类型：Document、Chunk、QueryResult 等类型系统 | `src/ingestion/models.py`, `src/libs/vector_store/base_vector_store.py` |
| D1.5 | 入口脚本设计：离线入库与在线检索脚本的职责边界与参数设计 | `scripts/ingest.py`, `scripts/ingest_mineru.py`, `scripts/retrieve.py`, `scripts/evaluate.py` |
| **D2** | **Ingestion Pipeline** | |
| D2.1 | Pipeline 整体流程：从文档加载到向量存储的阶段设计 | `src/ingestion/pipeline.py` |
| D2.2 | Chunking 策略：Heading/Recursive 切分与 parent_id 结构化 metadata 设计 | `src/libs/splitter/`, `src/ingestion/pipeline.py` |
| D2.3 | Transform 链：ChunkRefiner、MetadataEnricher 的职责与执行顺序 | `src/ingestion/transform/` |
| D2.4 | Embedding 编码：Dense/Sparse 双编码与 BatchProcessor 批处理 | `src/ingestion/embedding/` |
| D2.5 | 存储层：文档级替换写入（delete_by_source_doc_id + upsert）与统一存储协同 | `src/ingestion/storage/vector_upserter.py`, `src/libs/vector_store/sqlite_store.py`, `src/ingestion/pipeline.py` |
| **D3** | **Hybrid Search & Retrieval** | |
| D3.1 | Dense Retrieval：向量检索原理与 DenseRetriever 实现 | `src/core/query_engine/dense_retriever.py` |
| D3.2 | Sparse Retrieval：BM25 稀疏检索与 SparseRetriever 实现 | `src/core/query_engine/sparse_retriever.py` |
| D3.3 | Hybrid Search 融合：RRF 算法与 Fusion 模块设计 | `src/core/query_engine/hybrid_search.py`, `fusion.py` |
| D3.4 | Parent 聚合检索：parent_aggregator、aggregate_by_parent、top_m 候选裁剪机制 | `src/core/query_engine/parent_aggregator.py`, `src/core/query_engine/retrieval_pipeline.py`, `config/settings.yaml` |
| D3.5 | Query 到响应链路：QueryProcessor 与 ResponseBuilder/CitationGenerator/MultimodalAssembler 协同 | `src/core/query_engine/query_processor.py`, `src/core/response/` |
| **D4** | **Rerank 机制** | |
| D4.1 | Reranker 抽象与工厂模式：BaseReranker 与 RerankerFactory 设计 | `src/libs/reranker/base_reranker.py`, `reranker_factory.py` |
| D4.2 | CrossEncoder Reranker：模型原理与实现细节 | `src/libs/reranker/cross_encoder_reranker.py` |
| D4.3 | LLM Reranker：基于大语言模型的重排序方案与 Prompt 设计 | `src/libs/reranker/llm_reranker.py` |
| D4.4 | Rerank 在检索 Pipeline 中的集成位置与效果分析 | `src/core/query_engine/reranker.py` |
| **D5** | **MCP Server 协议** | |
| D5.1 | MCP 协议概述：JSON-RPC 交互模型与标准规范 | `src/mcp_server/server.py` |
| D5.2 | Tool 注册机制：四个工具的定义、参数与执行逻辑（query/list/normal/mineru） | `src/mcp_server/tools/`, `src/mcp_server/server.py` |
| D5.3 | Tool 执行链路：参数解析、错误模型、structuredContent 约定与可测试性设计 | `src/mcp_server/tools/mcp_utils.py`, `src/mcp_server/tools/error_utils.py`, `src/mcp_server/tools/config_utils.py` |
| D5.4 | Server 生命周期管理与工具注册：FastMCP 启动、日志与 Trace 初始化 | `src/mcp_server/server.py`, `main.py` |
| **D6** | **可插拔架构 & 配置系统** | |
| D6.1 | 工厂模式全景：LLM/Embedding/Reranker/VectorStore/Evaluator 五大工厂 | `src/libs/*/factory*.py` |
| D6.2 | settings.yaml 配置结构与 Settings 类加载机制 | `config/settings.yaml`, `src/core/settings.py` |
| D6.3 | LLM Provider 多厂商支持：Azure/OpenAI/DeepSeek/Ollama 切换逻辑 | `src/libs/llm/` |
| D6.4 | Embedding Provider 抽象：多后端实现对比与选型策略 | `src/libs/embedding/` |
| D6.5 | Base 类设计哲学：接口抽象、继承层次与扩展点 | `src/libs/*/base_*.py` |
| **D7** | **多模态处理** | |
| D7.1 | PDF 解析与版面恢复：页指纹匹配、相对位置图片占位与 FileIntegrity 协同 | `src/libs/loader/pdf_loader.py`, `src/libs/loader/file_integrity.py` |
| D7.2 | Vision LLM：Qwen/DashScope Vision 图片理解能力集成与调用链路 | `src/libs/llm/dashscope_vision_llm.py`, `src/libs/llm/llm_factory.py` |
| D7.3 | ImageCaptioner：图片描述生成流程与 Prompt 模板设计 | `src/ingestion/transform/image_captioner.py`, `config/prompts/` |
| D7.4 | 多模态统一存储与返回：SQLite images 表、ResponseBuilder 与 MultimodalAssembler 协同 | `src/libs/vector_store/sqlite_store.py`, `src/core/response/response_builder.py`, `src/core/response/multimodal_assembler.py` |
| **D8** | **可观测性 & 评估体系** | |
| D8.1 | Trace 系统：TraceCollector 与 TraceContext 的采集与关联设计 | `src/core/trace/` |
| D8.2 | Dashboard 架构：Streamlit App 分页、Services 层数据流 | `src/observability/dashboard/` |
| D8.3 | 评估指标体系：Recall、Precision、MRR 等核心指标定义与计算 | `src/observability/evaluation/`, `scripts/evaluate.py` |
| D8.4 | 评估执行链路：content_evaluator、eval_runner、e2e_runner 与评分落地（当前以 MCP 实战与 custom 评估为主） | `src/observability/evaluation/content_evaluator.py`, `src/observability/evaluation/eval_runner.py`, `src/observability/evaluation/e2e_runner.py` |
| D8.5 | 日志系统：Logger 设计、日志分级与调试支持 | `src/observability/logger.py` |
| **D9** | **测试策略 & 工程化** | |
| D9.1 | 测试分层策略：Unit/Integration/E2E 各层覆盖范围与边界 | `tests/unit/`, `tests/integration/`, `tests/e2e/` |
| D9.2 | Test Fixtures 与 conftest.py：Mock 策略与测试数据管理 | `tests/conftest.py`, `tests/fixtures/` |
| D9.3 | pyproject.toml 工程配置：依赖管理、构建配置、工具链集成 | `pyproject.toml` |
| D9.4 | 关键能力单测覆盖：heading 切分、parent 聚合、按父取块、内容评估等新增能力的测试设计 | `tests/unit/test_heading_splitter.py`, `tests/unit/test_parent_aggregator.py`, `tests/unit/test_get_chunks_by_parent_id.py`, `tests/unit/test_content_evaluator.py` |
| **D10** | **幂等性 & 文档级增量更新** | |
| D10.1 | 文件完整性检查：SHA256、should_skip、mark_success 机制与 force 语义 | `src/libs/loader/file_integrity.py`, `src/mcp_server/tools/ingest_document_normal.py`, `src/mcp_server/tools/ingest_document_mineru.py` |
| D10.2 | 文档级替换写入：按 source_doc_id 先删旧块再写新块，避免旧块残留 | `src/ingestion/storage/vector_upserter.py`, `src/libs/vector_store/sqlite_store.py`, `src/ingestion/pipeline.py` |
| D10.3 | source_doc_id 稳定性设计：路径哈希的优缺点与重命名场景影响 | `src/libs/loader/pdf_loader.py`, `src/libs/loader/mineru_result_adapter.py`, `src/ingestion/pipeline.py` |
| D10.4 | 不同入库路径一致性：CLI 与 MCP 在 skip/force/replace 流程上的对齐 | `scripts/ingest.py`, `src/mcp_server/tools/ingest_document_normal.py`, `src/mcp_server/tools/ingest_document_mineru.py` |

### D10 High-score Answer Anchors (for evaluator)

When evaluating D10 answers, check whether user covered the key anchors below:

- **D10.1 完整性检查**: 明确说出 `compute_sha256 -> should_skip -> mark_success` 链路，并解释 `force=true` 如何绕过 skip。
- **D10.2 旧块清理**: 能说明为什么文档改短时需要先按 `source_doc_id` 删除旧块，再写新块，避免旧 chunk 污染检索。
- **D10.3 稳定性与边界**: 知道 `source_doc_id` 基于路径哈希通常稳定，但重命名/移动路径会被视为新文档。
- **D10.4 路径一致性**: 能对比 CLI 与 MCP 的入库语义，并指出两者在 skip/force/replace 上应保持一致。

### D2 High-score Answer Anchors (for evaluator)

When evaluating D2 answers, check whether user covered the key anchors below:

- **D2.1 流程编排**: 能准确描述 ingestion 的关键阶段（split → transform → encode → store），并知道不同入口（CLI/MCP）在前置步骤上的差异。
- **D2.2 结构化切分**: 能说明 Heading/Recursive 切分差异，以及 `parent_id`/`chunk_index` 对后续父聚合的作用。
- **D2.3 Transform 语义**: 知道 ChunkRefiner、MetadataEnricher、ImageCaptioner 的执行顺序与职责边界。
- **D2.4 编码层**: 明确 Dense/Sparse 双编码并行存在的原因，以及 BatchProcessor 的批处理价值。
- **D2.5 存储一致性**: 能解释“先按 `source_doc_id` 删除旧块再写新块”的必要性，以及统一存储对检索一致性的影响。

> **Total: 10 domains × 3-5 sub-topics = 45 knowledge points**
> Each sub-topic can be studied multiple times with different questions, providing 100+ possible interview questions.

---

## Phase 2: Check Learning History

1. Try reading `.cursor/skills/project-learner/references/LEARNING_PROGRESS.md`
2. **File missing** → first-time learner, proceed to Phase 3
3. **File exists** → parse BOTH tables:
   - **Domain Summary**: which domains are ⬜/🔴/🔶/✅
   - **Sub-topic Progress**: which sub-topics are ⬜ (unlearned), 🔴 (weak ≤3), 🔶 (learning 4-6), ✅ (mastered ≥7)
   - Count: total sub-topics mastered / 45
   - Identify lowest-scoring sub-topics for review recommendation

---

## Phase 3: User Intent

Use `AskQuestion` (中文) to determine what the user wants:

**Question 1 — 学习模式** (single-select):

| Option | Description |
|--------|------------|
| 🆕 学习新知识点 | Pick from unlearned/weak sub-topics |
| 📖 复习已学内容 | Review previously learned low-score sub-topics |
| 📋 查看学习进度 | Display progress table, then end |
| 🎯 Agent 推荐 | Auto-pick the best next sub-topic to study |

If user picks 📋 → display the full progress table from `LEARNING_PROGRESS.md` and stop.

If user picks 🎯 → Agent auto-selects the optimal sub-topic (prioritize: ⬜ unlearned in weakest domain → 🔴 weak → 🔶 lowest score). Skip Question 2 & 3, go directly to Phase 4.

**Question 2 — 知识域选择** (single-select, only for 🆕 or 📖):

List all 10 domains with current status + completion rate. Example format:
- `D1 RAG Pipeline 整体架构 [2/5 ✅] 🔶`
- `D2 Ingestion Pipeline [0/5 ✅] ⬜`

For 📖 mode: only show domains with previous scores. For 🆕 mode: prioritize domains with most ⬜ sub-topics.

**Question 3 — 知识点选择** (single-select, only after Question 2):

List all sub-topics under the selected domain with their status:
- `D2.1 Pipeline 整体流程 ⬜ 未学习`
- `D2.2 Chunking 策略 🔶 6/10`
- `D2.3 Transform 链 ✅ 8/10`

Include option:
- 🎯 Agent 推荐 — auto-pick the weakest/unlearned sub-topic in this domain

---

## Phase 3.5: Scenario Detection (Real Project First)

Before generating question, detect whether user is learning through a real business/demo scenario.

If any of the following is true, mark `scenario_mode = recipes`:
- collection is `recipes`
- dataset path includes `data/datasets/howtocook/`
- user asks about dish/cooking retrieval behavior

When `scenario_mode = recipes`:
- Prioritize D2 / D3 / D7 / D10 sub-topics
- Prefer asking implementation + product-behavior combined questions
- Avoid purely abstract architecture questions for the first round

---

## Phase 4: Generate Interview Question

Based on the selected **sub-topic** (not just domain):

1. **Deep-read** the sub-topic's specific source code — read actual class definitions, key functions, config sections listed in the Sub-topic Map
2. **Dynamically generate** ONE main interview question (中文) grounded in this sub-topic's real code
3. **Internally prepare** up to 4 progressive follow-up questions (do NOT show these yet)
4. **Avoid repeating** questions from previous sessions — check Detailed History for this sub-topic and generate a different angle

If `scenario_mode = recipes`, main question should include at least one concrete behavior expectation, for example:
- "为什么查询一道菜时返回整道菜，而不是一个子 chunk？"
- "文档修改后如何确保旧块不会污染检索结果？"
- "force=false 与 force=true 在入库路径中的行为差异是什么？"

### Question Design Principles

- Questions MUST reference real code/architecture from THIS project, never generic
- Questions should be specific to the sub-topic, not the whole domain
- Difficulty progression for follow-ups:
  - Follow-up 1: "为什么这样设计？" (design rationale)
  - Follow-up 2: "和替代方案对比有什么优劣？" (trade-offs)
  - Follow-up 3: "边界条件/异常情况怎么处理？" (edge cases)
  - Follow-up 4: "如果让你重新设计，会怎么做？" (redesign thinking)
- Adjust follow-ups dynamically based on what the user actually answers

### Question Angle Variety

Each sub-topic can be asked from multiple angles. When a sub-topic is revisited, pick a DIFFERENT angle:
- **What**: 描述这个模块/机制做了什么
- **How**: 具体实现细节，代码层面怎么做的
- **Why**: 为什么选择这种设计方案
- **Compare**: 和替代方案的对比
- **Debug**: 如果出了问题怎么排查
- **Extend**: 如果要扩展功能怎么做

### Scenario-aware Follow-up Templates (recipes)

Use these templates when `scenario_mode = recipes`:
- **Parent aggregation**: "如果关闭 `aggregate_by_parent`，用户检索体验会怎么变化？"
- **Heading split**: "`heading_parent_level=2` 与 `heading_split_level=3` 为什么适合菜谱结构？"
- **Incremental ingestion**: "文档改短后，如果不先删旧块，会出现什么可观测问题？"
- **Normal vs MinerU**: "同一个 PDF 分别用 normal/mineru 入库，你预期差异体现在哪些字段或检索结果上？"
- **Multimodal path**: "图片占位符、image refs、最终响应中的图片展示链路分别在哪里完成？"

### Question Format

Present to user:

```
## 🎯 面试问题

**知识域**: [Domain Name] > **知识点**: [Sub-topic Name]

**面试官问**: [Question text — specific to this sub-topic, referencing project components]

请回答：
```

---

## Phase 5: Interactive Q&A (≤4 Follow-up Rounds)

```
Round 0: Main question → User answers
Round 1-4: Brief feedback on previous answer + follow-up question → User answers
Early exit: User says "结束"/"pass"/"跳过" OR answer is sufficiently comprehensive
```

### Per-Round Behavior

1. **Acknowledge** what the user got right (1-2 sentences, 中文)
2. **Hint** at what was missed without giving away the answer (1 sentence)
3. **Ask follow-up** that digs deeper based on their answer direction

### Follow-up Output Format

```
### 第 N 轮追问

✅ **答得好**: [What they got right]
💡 **提示**: [What they could explore further]

**追问**: [Follow-up question]
```

If user's answer already covers the planned follow-up, skip to a harder one or end early.

---

## Phase 6: Evaluation

After Q&A ends, output a structured evaluation report (中文):

```markdown
## 📊 评价报告

**知识域**: [Domain] > **知识点**: [Sub-topic ID & Name] — [Question summary]
**追问轮数**: N/4

### ✅ 回答亮点
- [Strength 1 — specific to what they said]
- [Strength 2]

### ⚠️ 需要加强
- [Gap 1 — what was missed or inaccurate]
- [Gap 2]

### 📈 评分明细

| 维度 | 分数 | 说明 |
|------|------|------|
| 准确性 | X/10 | [Factual correctness of answers] |
| 深度 | X/10 | [How deep they went beyond surface] |
| 代码关联 | X/10 | [Did they reference actual code/config] |
| 设计思维 | X/10 | [Trade-off analysis, architecture reasoning] |

### 🏆 综合评分: X/10

### 📊 学习进度: [mastered count]/45 知识点已掌握
```

Scoring rules:

- Average of 4 dimensions, rounded to nearest 0.5
- 9-10: Expert level, can explain design decisions and trade-offs
- 7-8: Solid understanding, knows how and why
- 4-6: Basic understanding, knows what but not deep why
- 1-3: Surface level, needs significant study

Additional evaluator rule:
- If sub-topic is D10.x and answer misses **2 or more** D10 anchors above, overall score should not exceed 7.0.
- If sub-topic is D2.x and answer misses **2 or more** D2 anchors above, overall score should not exceed 7.0.

---

## Phase 7: Learning Guide

Immediately after evaluation, provide targeted study resources (中文):

```markdown
## 📚 学习指南

### 📂 相关代码
- [file_path](file_path#LX-LY) — 说明这段代码的作用和关键逻辑

### 📄 相关文档
- [DEV_SPEC.md 对应章节](DEV_SPEC.md) — 设计原理
- [config/settings.yaml](config/settings.yaml) — 相关配置项

### 🔗 参考资料
- [External concept name] — 1-sentence explanation of relevance

### 💡 建议学习路径
1. 先阅读 [file] 理解 [what]
2. 再看 [file] 掌握 [implementation detail]
3. 运行 `[command]` 实际体验效果
4. 尝试修改 [config/code] 观察变化
```

Guidelines:
- Code references MUST use actual file paths with line numbers where relevant
- Only recommend reading 3-5 key files, not entire codebase
- Include at least one hands-on command the user can run
- External references only for concepts not explained in the codebase (e.g., RRF algorithm, BM25)

---

## Phase 8: Persist Progress

Update `.cursor/skills/project-learner/references/LEARNING_PROGRESS.md`.

If file doesn't exist, create it from the template in [references/LEARNING_PROGRESS.md](references/LEARNING_PROGRESS.md). If it exists, update it.

### Update Rules

1. **Append** one row to the `Detailed History` table (include Sub-topic ID)
2. **Update** the `Sub-topic Progress` table for the affected sub-topic:
   - 已学 = count of sessions for that sub-topic
   - 最高分 = max score across all sessions for this sub-topic
   - 最近分 = score from this session
   - Status: ≥7 → ✅ 掌握, 4-6 → 🔶 学习中, ≤3 → 🔴 薄弱, 0 sessions → ⬜ 未学习
3. **Recalculate** the `Domain Summary` table:
   - 已掌握 = count of ✅ sub-topics in that domain / total sub-topics in domain
   - 已学习 = count of non-⬜ sub-topics / total sub-topics
   - 平均分 = average score of all studied sub-topics in domain
   - Domain status: all sub-topics ✅ → ✅ 掌握, any studied → 🔶 学习中 or 🔴 薄弱 (based on avg), none → ⬜ 未学习
4. **Update** the `Last updated` timestamp
5. **Update** the session counter `#` (auto-increment)
6. **Update** the overall progress line: `总进度: X/45 知识点已掌握`

---

## Phase 9: Continue or End

After persisting, ask the user (中文):

| Option | Action |
|--------|--------|
| 🔄 继续学习下一个知识点 | Loop back to Phase 3 |
| 🎯 Agent 推荐下一个 | Auto-pick optimal next sub-topic, go to Phase 4 |
| 📋 查看当前学习进度 | Display full progress table |
| 🏁 结束本次学习 | Show session summary, stop |

### Session Summary (on 🏁 end)

```markdown
## 📝 本次学习总结

- 完成知识点: N 个
- 平均得分: X/10
- 最强知识点: [sub-topic] (X/10)
- 需加强知识点: [sub-topic] (X/10)
- 总进度: X/45 知识点已掌握 (XX%)

继续加油！下次建议学习: [recommended sub-topic name]
```

---

## Key Paths

| File | Purpose |
|------|---------|
| `.cursor/skills/project-learner/references/LEARNING_PROGRESS.md` | Persistent learning state (45 sub-topics) |
| `DEV_SPEC.md` | Project specification & architecture |
| `config/settings.yaml` | Configuration reference |
| `src/` | All source code modules |
| `tests/` | Test suite for understanding test strategy |
| `scripts/` | CLI entry points (ingest/ingest_mineru/retrieve/evaluate) |
