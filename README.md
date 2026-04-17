# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG (检索增强生成) 服务框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

---

## 🔧 项目功能

本项目是一个**企业级智能问答与知识检索系统**，可应用于以下场景：

- **📖 文档问答**：支持 PDF、Markdown、代码文件等多格式文档的智能问答，快速从海量文档中提取精准答案
- **🔍 语义搜索**：基于混合检索技术（稠密向量 + BM25/FTS5），提供比传统关键词搜索更智能的语义理解能力
- **💡 知识库构建**：将企业内部文档、技术资料转化为可检索的知识库；PDF 支持 **本地解析（MarkItDown）** 与 **MinerU 云端精细解析** 两种入库路径
- **🤖 AI 助手集成**：通过 MCP (Model Context Protocol) 协议，可无缝对接 Claude、GitHub Copilot、Cursor 等支持 MCP 的客户端
- **🎯 个性化应用**：可扩展为客服机器人、技术文档助手、代码搜索引擎等垂直领域应用

> 💼 **面试与学习**：模块化实现与 `DEV_SPEC.md` 设计文档对齐，适合作为简历项目；`.cursor/skills/` 下提供面试模拟、项目学习、复习与简历等辅助技能。

---

## 🎯 项目特点

### 1️⃣ **可插拔架构 (Pluggable Architecture)**
- **LLM 后端灵活切换**：支持 Azure OpenAI、OpenAI、Ollama、DeepSeek、DashScope、Qwen 等（见 `config/settings.yaml` 中 `llm.provider`），通过配置切换
- **模型组件可替换**：Embedding、Reranker、Loader、VectorStore、Splitter 等通过工厂与配置装配
- **检索策略可配置**：稠密检索、稀疏检索（FTS5/BM25）、混合检索与融合算法（如 RRF）可在配置中调整

### 2️⃣ **全链路可观测 (Observable)**
- **结构化追踪**：可选将 MCP 调用等写入 `logs/traces.jsonl`（由 `main.py` 与配置驱动）
- **评估与金标**：支持自定义评估链路（`evaluation.backends`）；**Ragas / DeepEval 在 Evaluator 工厂中仍为占位，当前请使用 `custom` 后端**
- **Trace Dashboard**：安装 `pip install -e ".[dashboard]"` 后，可运行 `python scripts/start_dashboard.py` 查看 Streamlit 面板

### 3️⃣ **MCP 协议集成 (Model Context Protocol)**
- **标准化接口**：基于官方 MCP SDK（FastMCP），默认 **Stdio** 传输
- **当前暴露的工具**（见 `src/mcp_server/server.py`）：
  - `query_knowledge_hub`：知识库语义检索与问答
  - `list_collections`：列出已入库集合
  - `ingest_document_normal`：本地 PDF 解析入库（MarkItDown）
  - `ingest_document_mineru`：MinerU 云端解析入库（复杂版式 PDF）
- **上下文增强**：为对话客户端提供可检索的领域知识

### 4️⃣ **工程化 RAG 实践**
- **多种切分策略**：如递归切分、标题感知切分等（见 `src/libs/splitter/`）
- **混合检索**：稠密 + 稀疏，适配不同数据分布
- **重排**：支持 `none`、Cross-Encoder、`llm` 等 rerank 后端（Cross-Encoder 需 `pip install -e ".[reranker]"`）

---

## 📚 AI 驱动开发：让 AI 成为你的协作伙伴

### 💡 核心理念

> **"文档即规范，实现交给 AI"**

本项目采用 **AI 协作开发模式**：架构与接口以 `DEV_SPEC.md` 为主文档，配合 Cursor Skills 完成学习与自动化辅助。

#### ✨ 项目特色
- **Cursor Skills**：`.cursor/skills/` 下提供与仓库绑定的能力包，例如：
  - `interview-prep`：模拟技术面试并生成报告
  - `project-learner`：分域问答式学习项目
  - `project-review`：章节化复习与进度记录
  - `resume-writer`：基于本项目撰写简历项目描述
  - `skill-creator`：创建或校验新 Skill 的流程与脚本
- **规范驱动**：`DEV_SPEC.md` 描述架构、模块边界与技术选型，开发与阅读时以代码与 spec 对照为准

#### 🚀 建议工作流

```
1. 📝 阅读 DEV_SPEC.md     → 理解分层与数据流
2. ✏️ 调整 config/settings.yaml（或 settings.local.yaml）→ 对齐你的 API 与存储路径
3. 🤖 按需启用 .cursor/skills 中的技能 → 学习、面试或文档辅助
4. ✅ 运行测试与端到端验证 → pytest / MCP 客户端联调
```

#### 📖 配套资源

| 资源类型 | 内容说明 |
|---------|---------|
| 📄 **设计文档** | `DEV_SPEC.md`：架构、模块、配置约定 |
| 💻 **Skills** | `.cursor/skills/`：面试、学习、复习、简历等交互式工作流 |

> 💡 **提示**：环境变量、向量维度、SQLite 统一存储等细节以 `DEV_SPEC.md` 与 `config/settings.yaml` 为准。

### 🎁 你将收获什么

- **RAG 全链路**：入库、切分、向量化、混合检索、重排、响应与引用
- **MCP 集成**：将检索能力以工具形式暴露给 Agent
- **可维护的工程结构**：`src/core`、`src/ingestion`、`src/libs`、`src/mcp_server` 分层清晰

---

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/hcygit1/MODULAR-RAG-MCP-SERVER.git
cd MODULAR-RAG-MCP-SERVER

# 安装依赖（推荐使用 pyproject.toml）
pip install -e .

# 可选：Cross-Encoder 重排
pip install -e ".[reranker]"

# 可选：可观测性 Dashboard（Streamlit）
pip install -e ".[dashboard]"

# 配置文件：config/settings.yaml
# 可复制为 config/settings.local.yaml 覆盖本地配置
# API Keys 可通过环境变量设置，或在配置中使用 ${VAR_NAME} 语法

# 运行 MCP Server（Stdio，供 Copilot / Claude / Cursor 对接）
python main.py
# 或：python -m src.mcp_server.server
```

### 命令行脚本（节选）

- **入库**：`python scripts/ingest.py`、`python scripts/ingest_mineru.py`
- **检索调试**：`python scripts/retrieve.py`
- **评估**：`python scripts/evaluate.py`（需正确配置 `evaluation` 与金标数据）
- **Dashboard**：`python scripts/start_dashboard.py`

### 配置说明

- **主配置**：`config/settings.yaml`，支持环境变量 `MODULAR_RAG_CONFIG_PATH` 指定路径
- **本地覆盖**：创建 `config/settings.local.yaml` 覆盖密钥、模型提供商、路径等
- **统一存储（推荐）**：`vector_store.backend: sqlite` 且 `retrieval.sparse_backend: fts5`，chunk、向量、FTS5、图片等同库管理；详见 `DEV_SPEC.md` 中 SQLite / 统一存储相关章节
- **评估后端**：`evaluation.backends` 当前 **`custom` 可用**；`ragas` / `deepeval` 在 `EvaluatorFactory` 中尚未实现，选用会触发 `NotImplementedError`
- **Reranker**：`rerank.backend` 支持 `none`、`cross_encoder`、`llm`；`cross_encoder` 需安装 `pip install -e ".[reranker]"`

更多说明见 [DEV_SPEC.md](DEV_SPEC.md)。

---

## 📂 项目结构

```
.
├── DEV_SPEC.md              # 核心设计文档
├── config/
│   └── settings.yaml        # 主配置文件（可配合 settings.local.yaml）
├── .cursor/
│   └── skills/              # Cursor Skills（面试、学习、复习、简历等）
├── main.py                  # MCP Server 启动入口
├── scripts/                 # 入库、检索、评估、Dashboard 等脚本
├── src/                     # 源代码
│   ├── core/                # 配置、Query Engine、Trace、Response
│   ├── ingestion/           # 解析、切分、Embedding、存储管道
│   ├── libs/                # Loader、Embedding、Reranker、VectorStore、Evaluator 等
│   ├── mcp_server/          # FastMCP Server 与 MCP 工具
│   └── observability/       # 日志、Dashboard、评测辅助
├── tests/                   # 单元 / 集成 / 端到端测试
└── docs/                    # 预留或补充文档目录（以 DEV_SPEC 为主）
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。在贡献代码前，请：

1. 阅读 [DEV_SPEC.md](DEV_SPEC.md) 了解架构与约定
2. 保持与现有代码风格一致（可参考 `pyproject.toml` 中的 black/ruff 配置）
3. 运行测试：`pytest tests/`

---

## 📄 License

[MIT License](LICENSE)

---

## 🌟 Star History

如果这个项目对您有帮助，欢迎 Star ⭐️ 支持！

---
