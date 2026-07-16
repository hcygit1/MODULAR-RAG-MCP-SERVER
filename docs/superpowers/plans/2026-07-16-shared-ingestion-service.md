# Shared Ingestion Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提取三个 MCP 入库工具的重复编排，保持所有公开参数、响应结构、完整性语义和异常文案不变。

**Architecture:** 新增 `IngestionService` 作为同步应用服务，负责 request 校验、完整性、同文档互斥、Pipeline/trace 资源和错误映射。工具通过小型 strategy spec 提供 loader name、parser 和成功/跳过响应 builder，不在服务内硬编码 Normal/Docling/MinerU 分支。

**Tech Stack:** Python 3.12、threading、MCP SDK、pytest

---

### Task 1: 锁定共享服务契约

**Files:**
- Create: `tests/unit/test_ingestion_service.py`

- [x] 测试空路径、不存在文件和不支持后缀的统一错误。
- [x] 测试默认 collection、force 和 loader-specific processing profile。
- [x] 测试未变更文件不调用 parser/Pipeline，不 mark success。
- [x] 测试 document/chunks 两种 payload 分别委托 `process_document` / `process_chunks`。
- [x] 测试成功后 mark success，并且无论成败都 collect trace 和 close pipeline。
- [x] 测试同 hash + collection + profile 串行，不同 key 可并行，lock registry 会回收。
- [x] 运行并确认因 `ingestion_service` 模块不存在而 RED。

### Task 2: 实现 IngestionService

**Files:**
- Create: `src/mcp_server/services/__init__.py`
- Create: `src/mcp_server/services/ingestion_service.py`

- [x] 实现 `ParsedIngestion`，明确 `document` / `chunks` payload 模式和结果 metadata。
- [x] 实现不可变 `IngestionToolSpec`，持有 supported suffixes、loader resolver、parser、response builders 和错误前缀。
- [x] 实现 `IngestionService.execute(arguments, spec)`，统一完整性、Pipeline/trace、mark success 和错误映射。
- [x] 实现有引用计数的 keyed lock registry，key 为 file hash + collection + processing profile。
- [x] 通过构造函数注入 settings/checker/profile/pipeline/trace/collector 依赖，保持现有测试可隔离。
- [x] 运行 Task 1 测试并确认 GREEN。

### Task 3: 迁移三个 MCP 工具

**Files:**
- Modify: `src/mcp_server/tools/ingest_document_normal.py`
- Modify: `src/mcp_server/tools/ingest_document_docling.py`
- Modify: `src/mcp_server/tools/ingest_document_mineru.py`
- Modify: `tests/unit/test_ingest_document_normal.py`
- Modify: `tests/unit/test_ingest_processing_profiles.py`

- [x] Normal strategy 保留 PDF/Markdown/TXT loader、doc_type 和图片解析。
- [x] Docling strategy 保留支持后缀、结构化 chunks 和 empty-chunks 错误。
- [x] MinerU strategy 保留 token 优先级、content-list / markdown fallback 和 parser_mode。
- [x] 三个 async FastMCP wrapper 继续使用 `run_sync_callable`，公开 schema 不变。
- [x] 通过现有工具、processing profile、async wrapper 和 MCP schema 测试。

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_ingestion_service.py \
  tests/unit/test_ingest_document_normal.py \
  tests/unit/test_ingest_processing_profiles.py \
  tests/unit/test_mcp_async_wrappers.py \
  tests/integration/test_mcp_server.py
```

### Task 4: 验收和提交

- [x] 运行 ruff、定向 mypy、compileall、`git diff --check` 和全量 pytest。
- [x] 审查公开响应字段/文案、patch 路径、lock 回收和资源释放。
- [x] 独立规格审查和代码质量审查通过。
- [ ] 提交：

```bash
git commit -m "refactor: 统一 MCP 文档入库编排"
```
