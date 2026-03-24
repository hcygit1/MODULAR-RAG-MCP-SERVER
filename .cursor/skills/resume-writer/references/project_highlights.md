# 项目技术亮点清单（Modular RAG MCP Server）

> 从 DEV_SPEC 与源码提炼，供简历编写时按需选取。每个亮点附带"简历话术方向"和"可量化角度"。

---

## 亮点 1：多阶段混合检索架构（Hybrid Search + Rerank）

**技术要点**：
- 设计并实现"粗排召回 → 精排重排"两段式检索架构
- 粗排阶段并行执行 Dense Retrieval（语义向量，Cosine Similarity）+ Sparse Retrieval；**默认 SQLite 后端**下稀疏一路为 **SQLite FTS5 全文/关键词检索**（与独立 BM25 服务区分）
- 通过 RRF（Reciprocal Rank Fusion）融合双路排名，再经可选 Rerank 精排，平衡查全与查准
- 精排阶段支持 Cross-Encoder 本地模型 / LLM Rerank / None 三种模式可插拔切换
- 精排失败时自动回退至融合排名（Graceful Fallback），保障系统可用性

**简历话术方向（推荐，与实现一致）**：
- "搭建 **Dense Retrieval + SQLite FTS5 关键词检索 + RRF 融合** 的混合召回，并配合 **Rerank 两阶段排序**；在自建 **Golden Test Set**（query + 标注 `golden_chunk_id`）上，于固定 **top_k** 检索结果上统计 **Hit Rate（黄金 chunk 命中比例）与 MRR**，混合检索较 **单路稠密检索** 指标提升约 **15%–25%**（请替换为自测真实值与样本量）。"
- "引入 Cross-Encoder / LLM Rerank 可插拔精排，失败时回退 RRF 排序，兼顾效果与可用性。"

**可量化角度（与 `scripts/evaluate.py` / CustomEvaluator 一致）**：`custom_hit_rate`、`custom_mrr`、每条 case 的 `top_k`、端到端 `latency_ms`；E2E 模式可加 L2：`content_non_empty` / `citations_ok` 等。NDCG、RAGAS 等为扩展/规划能力，勿与默认离线脚本混写。

---

## 亮点 2：全链路可插拔架构（Factory + 配置驱动）

**技术要点**：
- 为 LLM / Embedding / Splitter / VectorStore / Reranker / Evaluator 六大组件定义统一抽象接口（Base 类）
- 采用工厂模式（Factory Pattern）+ YAML 配置驱动，实现"改配置不改代码"的组件切换
- LLM Provider 支持 Azure OpenAI / OpenAI / Ollama / DeepSeek 四种后端
- Embedding 支持 OpenAI / Azure / Ollama 三种后端
- 向量数据库接口预留扩展（当前默认 Chroma，可切换 Qdrant/Pinecone）
- Vision LLM 独立抽象（BaseVisionLLM），支持多模态图像处理

**简历话术方向**：
- "设计了全链路可插拔架构，基于抽象接口 + 工厂模式 + 配置驱动，实现 LLM/Embedding/VectorStore 等 6 大核心组件的零代码热切换"
- "架构支持 Azure OpenAI、本地 Ollama 等多种 Provider 无缝切换，满足企业合规与成本优化需求"

**可量化角度**：支持 N 种 LLM Provider、N 种 Embedding 后端、配置切换零代码修改

---

## 亮点 3：智能数据摄取流水线（Ingestion Pipeline）

**技术要点**：
- 自研五阶段流水线：Load → Split → Transform → Embed → Upsert
- PDF 解析采用 MarkItDown 转 canonical Markdown，保留文档结构
- 使用 LangChain RecursiveCharacterTextSplitter 进行语义感知切分
- Transform 阶段包含三个 LLM 增强步骤：
  - ChunkRefiner：LLM 驱动的 Chunk 智能重组与去噪
  - MetadataEnricher：自动生成 Title/Summary/Tags 语义元数据
  - ImageCaptioner：Vision LLM 生成图片描述，实现"搜文出图"
- SHA256 文件哈希 + 内容哈希实现增量摄取与幂等 Upsert
- 双路索引/检索：Dense（Embedding + 向量表）+ 稀疏关键词（默认 SQLite 场景为 **FTS5**，非独立 BM25 服务）

**简历话术方向**：
- "设计并实现了五阶段智能数据摄取流水线，整合文档解析、语义切分、LLM 增强（智能重组/元数据注入/图片描述）、双路向量化与幂等存储"
- "实现基于 SHA256 哈希的增量摄取机制，避免重复处理，降低 API 调用成本 XX%"

**可量化角度**：处理文档数、生成 Chunk 数、增量摄取跳过率、LLM 增强覆盖率、端到端摄取耗时

---

## 亮点 4：MCP 协议集成（Model Context Protocol）

**技术要点**：
- 遵循 MCP 标准（JSON-RPC 2.0 + Stdio Transport）实现知识检索 Server
- 暴露 3 个标准 Tool：query_knowledge_hub / list_collections / get_document_summary
- 支持 GitHub Copilot、Claude Desktop 等主流 MCP Client 即插即用
- 返回格式支持 TextContent + ImageContent 多模态内容，含结构化 Citation 引用
- Stdio Transport 零配置、零网络依赖，天然适合私有知识库场景

**简历话术方向**：
- "基于 MCP（Model Context Protocol）标准实现知识检索 Server，支持 GitHub Copilot / Claude Desktop 等 AI Agent 直接调用私有知识库"
- "实现引用透明的结构化响应（Citation），支持文本 + 图像多模态返回，增强 AI 输出的可信度"

**可量化角度**：支持 N 种 MCP Client、工具调用成功率、端到端响应延迟

---

## 亮点 5：多模态图像处理（Image-to-Text）

**技术要点**：
- 采用 Image-to-Text 策略，复用纯文本 RAG 链路实现多模态检索
- Loader 阶段自动提取 PDF 图片并插入占位符标记
- Transform 阶段调用 Vision LLM（GPT-4o）生成结构化图片描述（Caption）
- 描述文本注入 Chunk 正文，被 Embedding 覆盖后可通过自然语言检索图片
- 检索命中后动态读取原始图片、编码 Base64 返回 MCP Client

**简历话术方向**：
- "设计 Image-to-Text 多模态处理方案，利用 Vision LLM 将文档图片转化为语义描述并嵌入检索链路，实现'搜文出图'能力"
- "无需引入 CLIP 等多模态向量库，复用纯文本 RAG 架构即可支持图像检索，降低架构复杂度"

**可量化角度**：处理图片数、图片描述平均长度、图片相关查询命中率

---

## 亮点 6：全链路可观测性与可视化管理平台

**技术要点**：
- 设计双链路追踪体系：Ingestion Trace（5 阶段）+ Query Trace（5 阶段）
- TraceContext 显式调用模式，低侵入记录各阶段耗时、候选数量、分数分布
- JSON Lines 结构化日志持久化，零外部依赖（无 LangSmith/LangFuse）
- 基于 Streamlit 构建六页面管理平台：
  - 系统总览（组件配置 + 数据资产统计）
  - 数据浏览器（文档/Chunk/图片详情查看)
  - Ingestion 管理（文件上传、实时进度条、文档删除）
  - Ingestion 追踪（阶段耗时瀑布图）
  - Query 追踪（Dense/Sparse 对比、Rerank 前后变化）
  - 评估面板（Ragas 指标、历史趋势）
- Dashboard 基于 Trace 中 method/provider 字段动态渲染，更换组件后自动适配

**简历话术方向**：
- "构建全链路白盒化追踪体系（Ingestion + Query 双链路），每次检索过程透明可回溯，支持精准定位坏 Case"
- "基于 Streamlit 实现六页面可视化管理平台，涵盖数据浏览、摄取管理、追踪分析、评估面板，实现 RAG 系统的全生命周期管理"

**可量化角度**：追踪覆盖阶段数、Dashboard 页面数、追踪日志条数、问题定位效率提升

---

## 亮点 7：自动化评估体系

**技术要点**：
- **默认主线**：`EvalRunner` 读取 **Golden Test Set**（`tests/fixtures/golden_test_set.json`，路径见 `config/settings.yaml` 的 `evaluation.golden_test_set`），对检索返回的 `retrieved_ids` 与 `golden_chunk_ids` 计算 **Hit Rate + MRR**（`CustomEvaluator`）
- **`python scripts/evaluate.py --e2e`**：L1 检索指标 + L2 内容规则检查（`content_non_empty`、`citations_ok`、`images_ok`、`keywords_ok`）
- 可插拔评估框架可扩展 Ragas 等后端；简历中若未实际接入，勿写成默认已跑 Ragas

**简历话术方向**：
- "基于 **自建黄金测试集** 做检索回归：`scripts/evaluate.py` 产出 **Hit Rate / MRR**（及可选 E2E 内容检查），策略变更可量化对比。"
- 量化句务必写清：**baseline**（如相对纯 Dense）、**top_k**、**样本条数**；避免空洞的「提升 15%–25%」而无指标名。

**可量化角度**：黄金集 query 条数、平均 `top_k`、`custom_hit_rate` / `custom_mrr`、E2E L2 通过率、端到端延迟

---

## 亮点 8：文档生命周期管理（DocumentManager）

**技术要点**：
- DocumentManager 独立于 Pipeline，负责跨 4 个存储的协调操作
- 支持文档列表、详情查看、协调删除（Chroma + BM25 + ImageStorage + FileIntegrity 四路同步）
- Pipeline 支持 on_progress 回调，Dashboard 实时展示各阶段进度条
- 幂等 Upsert 设计：chunk_id = hash(source_path + section_path + content_hash)

**简历话术方向**：
- "实现跨存储协调的文档生命周期管理，支持 Chroma/BM25/图片/处理记录四路同步删除，保障数据一致性"

**可量化角度**：管理文档数、跨存储操作成功率、删除操作耗时

---

## 亮点 9：工程化实践

**技术要点**：
- TDD 开发：1198+ 单元测试 + 30 E2E 测试全绿
- 9 个开发阶段、68 个子任务全部完成
- 分层测试金字塔：Unit → Integration → E2E
- SQLite 轻量持久化（ingestion_history + image_index + BM25 索引），零外部数据库依赖
- 配置驱动的零代码组件切换
- Prompt 模板外置（config/prompts/），支持独立迭代

**简历话术方向**：
- "遵循 TDD 开发范式，累计编写 1200+ 自动化测试用例，覆盖单元/集成/E2E 三层"
- "采用 SQLite Local-First 持久化方案，零外部数据库依赖，pip install 即可运行"

**可量化角度**：测试用例数、代码覆盖率、开发阶段数、子任务完成率

---

## 亮点 10：Agent 扩展性（面向 Agent 方向的延伸叙事）

**技术要点**：
- MCP Server 天然支持 Agent 调用（Tool Calling 范式）
- 系统可作为知识检索 Agent 嵌入 Multi-Agent 体系
- 支持构建自定义 Agent Client（ReAct / Chain of Thought 模式）
- 可快速适配不同业务场景（替换数据源、调整检索策略、定制 Prompt）

**简历话术方向**（适用于偏 Agent 方向的岗位）：
- "基于 MCP 协议构建知识检索 Agent，支持 Tool Calling / ReAct 模式，可嵌入 Multi-Agent 协作系统"
- "设计通用化知识检索框架，支持快速适配不同业务场景（替换数据源 + 调整检索策略 + 定制 Prompt），作为 Agent 生态的知识中枢"

**可量化角度**：支持的 Agent Client 数量、业务场景适配数
