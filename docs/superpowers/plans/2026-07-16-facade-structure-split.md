# Facade 结构拆分实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变公开 API、异常语义、trace stage 和事务行为的前提下，将 `IngestionPipeline` 与 `SQLiteVectorStore` 拆成职责单一的内部组件。

**Architecture:** 两个原有类继续作为 Facade。Pipeline 只保留参数校验、阶段编排和错误映射，分块构建、转换、编码、存储分别委托内部组件；SQLite Store 继续持有连接与事务边界，schema、记录、检索和图片 SQL 下沉到 repository/service。

**Tech Stack:** Python 3.12、SQLite、sqlite-vec、pytest

---

## 文件结构

- `src/ingestion/pipeline.py`：保留 `IngestionPipeline` 公开构造参数和公开方法，负责完整流程编排。
- `src/ingestion/document_chunk_builder.py`：调用 splitter，将 `Document` 转为带图片引用和位置信息的 `Chunk`。
- `src/ingestion/transform/transform_runner.py`：按既有顺序执行三个 transform，并以软失败方式执行 boundary repair。
- `src/ingestion/embedding/encoding_runner.py`：封装 `BatchProcessor`，提供稳定的 encode 边界。
- `src/ingestion/storage/storage_coordinator.py`：封装文档替换、后端不支持时的普通 upsert 回退。
- `src/libs/vector_store/sqlite_schema.py`：创建和迁移 chunks、vec、FTS、images schema。
- `src/libs/vector_store/sqlite_image_repository.py`：图片写入、引用清理和读取。
- `src/libs/vector_store/sqlite_record_repository.py`：记录校验、写入、删除、文档查找、collection 和 parent 查询。
- `src/libs/vector_store/sqlite_search.py`：metadata filter、dense query 和 sparse query。
- `src/libs/vector_store/sqlite_store.py`：保留连接生命周期、公开参数校验、事务提交/回滚和异常包装。

### Task 1: 锁定 Pipeline 内部组件契约

**Files:**
- Create: `tests/unit/test_pipeline_components.py`

- [x] 新增 `DocumentChunkBuilder` 测试，覆盖 chunk id、offset、split metadata 和按引用裁剪图片数据。
- [x] 新增 `TransformRunner` 测试，覆盖 transform 顺序和 boundary repair 失败时保留转换结果。
- [x] 新增 `EncodingRunner` 测试，确认参数原样委托给 batch processor。
- [x] 新增 `StorageCoordinator` 测试，覆盖文档替换和 `NotImplementedError` 回退。
- [x] 运行新测试并确认因组件模块不存在而失败。

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_pipeline_components.py
```

Expected: collection 阶段失败，提示待新增组件模块无法导入。

### Task 2: 拆分 IngestionPipeline

**Files:**
- Create: `src/ingestion/document_chunk_builder.py`
- Create: `src/ingestion/transform/transform_runner.py`
- Create: `src/ingestion/embedding/encoding_runner.py`
- Create: `src/ingestion/storage/storage_coordinator.py`
- Modify: `src/ingestion/pipeline.py`
- Modify: `tests/integration/test_ingestion_pipeline.py`

- [x] 实现四个内部组件，使 Task 1 测试通过。
- [x] 将 Facade 的 `split_document`、`_apply_transforms`、编码和 `_store_results` 改为委托。
- [x] 保留 `_vector_upserter` 和 `_store_results` 兼容属性/方法，避免现有测试和潜在调用方立即失效。
- [x] 保持 boundary repair warning、异常 cause、trace stage 名称和公开返回值不变。
- [x] 运行 Pipeline 聚焦测试。

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_pipeline_components.py \
  tests/unit/test_ingestion_splitter_integration.py \
  tests/unit/test_image_pipeline.py \
  tests/unit/test_ingest_processing_profiles.py \
  tests/integration/test_ingestion_pipeline.py
```

Expected: 全部通过。

- [x] 运行全量测试、编译和 diff 检查。
- [x] 提交：

```bash
git commit -m "refactor: 拆分文档入库流水线职责"
```

### Task 3: 锁定 SQLite 内部组件契约

**Files:**
- Create: `tests/unit/test_sqlite_store_components.py`

- [x] 新增 schema manager 测试，覆盖新库建表和旧联合主键迁移。
- [x] 新增 record repository 测试，覆盖 collection 隔离写删和 parent 查询。
- [x] 新增 image repository 测试，覆盖图片写入、读取和孤立图片清理。
- [x] 新增 search service 测试，覆盖 filter SQL、dense 结果映射和 sparse 查询。
- [x] 运行新测试并确认因组件模块不存在而失败。

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_sqlite_store_components.py
```

Expected: collection 阶段失败，提示待新增组件模块无法导入。

### Task 4: 拆分 SQLiteVectorStore

**Files:**
- Create: `src/libs/vector_store/sqlite_schema.py`
- Create: `src/libs/vector_store/sqlite_image_repository.py`
- Create: `src/libs/vector_store/sqlite_record_repository.py`
- Create: `src/libs/vector_store/sqlite_search.py`
- Modify: `src/libs/vector_store/sqlite_store.py`

- [x] 将 schema DDL 和旧表迁移移入 schema manager。
- [x] 将记录 SQL 和 metadata 序列化移入 record repository。
- [x] 将图片 SQL 移入 image repository，并由 record repository 在同一事务中调用。
- [x] 将 dense/sparse 查询和 filter 编译移入 search service。
- [x] Facade 保留参数校验、连接加载、事务、公开返回值和异常包装。
- [x] 运行 SQLite 聚焦测试。

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_sqlite_store_components.py \
  tests/unit/test_sqlite_vector_store.py \
  tests/unit/test_get_chunks_by_parent_id.py \
  tests/unit/test_sparse_retriever.py
```

Expected: 全部通过。

- [x] 运行全量测试、编译和 diff 检查。
- [x] 提交：

```bash
git commit -m "refactor: 拆分 SQLite 向量存储职责"
```

### Task 5: 独立审查和阶段验收

**Files:**
- Review: 本计划新增和修改的全部生产、测试文件。

- [ ] 检查 Facade 公开签名和导入路径没有变化。
- [ ] 检查 SQLite transaction ownership 没有下沉到 repository。
- [ ] 检查 CRLF 测试文件仍保持 CRLF。
- [ ] 运行完整 pytest、`compileall`、`git diff --check` 和 CRLF 检查。
- [ ] 请求独立代码审查，修复 Critical 和 Important 问题。
- [ ] 更新项目整体加固进度，进入共享 ingest service 阶段。
