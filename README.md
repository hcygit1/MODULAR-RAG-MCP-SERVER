# Modular RAG MCP Server

一个模块化、可插拔的 RAG（Retrieval-Augmented Generation）服务框架，支持通过 MCP（Model Context Protocol）将知识库检索、文档入库和问答能力暴露给外部客户端。

## 项目概述

Modular RAG MCP Server 面向文档问答、知识库检索和智能助手集成场景，提供从文档解析、文本切分、向量化、检索、重排到回答生成的完整 RAG 流程。

项目采用分层架构，核心能力通过配置和工厂模式装配，便于替换 LLM、Embedding、Reranker、Loader、VectorStore 等组件。

## 核心功能

- 文档入库：支持 PDF、Markdown、TXT 以及 Docling 可解析的 Office/HTML 等文档类型。
- 混合检索：支持稠密向量检索与稀疏检索（FTS5/BM25）组合。
- 问答生成：基于检索上下文调用 LLM 生成回答，并保留引用来源。
- MCP 服务：通过 FastMCP 暴露标准工具接口，供支持 MCP 的客户端调用。
- 可观测性：支持结构化 trace、评估脚本和 Streamlit Dashboard。
- 可扩展组件：LLM、Embedding、Reranker、Splitter、VectorStore 等模块可按配置切换。

## 主要模块

```text
.
├── main.py                  # MCP Server 启动入口
├── config/
│   └── settings.yaml        # 主配置文件
├── scripts/                 # 入库、检索、评估、Dashboard 等脚本
├── src/
│   ├── core/                # 配置、查询引擎、Trace、响应模型
│   ├── ingestion/           # 文档解析、切分、Embedding、存储流程
│   ├── libs/                # Loader、Embedding、Reranker、VectorStore、LLM 等组件
│   ├── mcp_server/          # MCP Server 与工具定义
│   └── observability/       # 日志、Dashboard、评估辅助
├── tests/                   # 单元、集成和端到端测试
└── docs/                    # 补充文档
```

## MCP 工具

当前服务注册以下 MCP 工具：

- `query_knowledge_hub`：在知识库中执行语义检索，返回 Markdown 结果和结构化引用。参数：`query`、`collection_name`、`top_k`。
- `list_collections`：列出已入库的集合名称。
- `ingest_document_normal`：使用本地解析流程入库 `.pdf`、`.md`、`.txt` 文件。参数：`file_path`、`collection_name`、`force`。
- `ingest_document_docling`：使用 Docling 结构化解析并入库 `.pdf`、`.doc`、`.docx`、`.ppt`、`.pptx`、`.html`、`.md` 文件。参数：`file_path`、`collection_name`、`force`。
- `ingest_document_mineru`：使用 MinerU 云端解析并入库复杂版式 PDF。参数：`file_path`、`collection_name`、`force`。

具体实现见 `src/mcp_server/server.py` 和 `src/mcp_server/tools/`。

## 环境要求

- Python 3.10+
- 可用的 LLM 与 Embedding Provider 配置
- 可选：SQLite、Chroma、Qdrant 等向量存储后端配置

## 安装

```bash
pip install -e .
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

安装 Cross-Encoder 重排依赖：

```bash
pip install -e ".[reranker]"
```

安装 Dashboard 依赖：

```bash
pip install -e ".[dashboard]"
```

安装 Docling 解析依赖：

```bash
pip install -e ".[docling]"
```

## 配置

默认配置文件为：

```text
config/settings.yaml
```

也可以通过环境变量指定配置文件路径：

```bash
export MODULAR_RAG_CONFIG_PATH=/path/to/settings.yaml
```

推荐创建本地覆盖配置文件保存密钥、模型提供商、存储路径等环境相关内容：

```text
config/settings.local.yaml
```

常见配置项包括：

- `llm.provider`：LLM 服务提供商。
- `embedding.provider`：Embedding 服务提供商。
- `vector_store.backend`：向量存储后端。
- `retrieval.sparse_backend`：稀疏检索后端。
- `rerank.backend`：重排后端，可选 `none`、`cross_encoder`、`llm`。
- `evaluation.backends`：评估后端；当前 `custom` 可用，`ragas` 和 `deepeval` 尚未实现。
- `mineru.api_token`：MinerU 云端解析所需 Token，也可通过 `MINERU_API_TOKEN` 设置。

## 运行 MCP Server

```bash
python main.py
```

或：

```bash
python -m src.mcp_server.server
```

默认使用 Stdio 传输，适合接入支持 MCP 的客户端。

## 常用脚本

文档入库：

```bash
python scripts/ingest.py --collection my_collection --path /path/to/document.pdf
```

MinerU 文档入库：

```bash
python scripts/ingest_mineru.py --collection my_collection /path/to/document.pdf
```

检索调试：

```bash
python scripts/retrieve.py --query "检索问题" --collection my_collection --top-k 5
```

运行评估：

```bash
python scripts/evaluate.py --golden-set tests/fixtures/golden_test_set.json --collection my_collection
```

启动 Dashboard：

```bash
python scripts/start_dashboard.py --port 8501
```

## 测试

```bash
pytest tests/
```

或安装开发依赖后运行：

```bash
pytest
```

## 开发说明

- 新增组件时优先复用现有接口和工厂模式。
- 配置项应集中在 `config/settings.yaml` 或本地覆盖配置中管理。
- 修改检索、入库、重排、评估等核心链路后，应补充对应单元测试或集成测试。
- 评估后端中 `custom` 当前可用，`ragas` 和 `deepeval` 相关实现以代码状态为准。
