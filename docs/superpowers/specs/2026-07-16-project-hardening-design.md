# 项目整体加固设计

## 目标

在不改变 MCP 工具公开参数和响应结构的前提下，修复已确认的数据丢失、跨 collection 串数据、后端契约不一致、同步阻塞和发布不完整问题，并把高职责文件拆成稳定的内部组件。

## 实施策略

采用按风险排序的垂直批次，而不是一次性重写：

1. 数据正确性：原子文档替换、SQLite 联合主键、图片归属、Chroma metadata。
2. 存储契约：filters、删除计数、集合枚举、父子 metadata key、完整性 processing profile。
3. MCP 运行时：调用参数优先级、线程卸载、可生效的超时。
4. 结构拆分：Pipeline Facade、SQLite Store Facade、共享 ingest service。
5. 发布工程化：正式包命名空间、资源 API、CLI、依赖、CI、LICENSE。
6. 收尾：处理 skipped/warnings、清空测试运行数据、构建并安装 wheel 验证。

每一批先增加能够复现旧行为的失败测试，再实现最小修复，最后运行相关测试和完整测试后独立提交。

## 数据正确性

### 文档替换

`BaseVectorStore` 增加可选的文档级替换能力，默认抛出 `NotImplementedError`。SQLite 在一个事务内完成：

- 找出 collection 内指定 `source_doc_id` 的旧 rowid。
- 删除旧 vec、FTS、chunk 和不再引用的图片。
- 写入新 chunk、vec、FTS 和图片。
- 任一步失败则整体回滚。

`VectorUpserter` 只负责构造 `VectorRecord`，不再通过私有 `_write_fts` 判断后端能力。

### Collection 隔离

SQLite `chunks` 主键迁移为 `(id, collection_name)`。迁移保留原 rowid，避免破坏 `chunks_vec.chunk_rowid` 对应关系。所有按 id 的读写删除必须同时约束 collection。

### 图片隔离

Markdown 图片 ID 使用 `{doc_id}_image_{stem}_{index}`。查询响应必须获得明确的 effective collection；缺少 collection 时不再仅按 image id 猜测，宁可不返回图片也不返回错误 collection 的图片。

### Chroma Metadata

稀疏向量不再放入通用向量 metadata。BM25/FTS5 继续使用独立 sparse 链路。Chroma 对其他复合 metadata 使用明确的 JSON 标量编码，并在读取时恢复。

## 存储契约

所有后端统一以下语义：

- `query(filters=...)` 要么正确执行，要么明确拒绝，不得静默忽略。
- `delete()` 返回实际删除数。
- `list_collections()` 由当前后端实现提供。
- collection 是记录身份的一部分。
- 自定义 `parent_id_metadata_key` 在检索和存储查询中一致。

完整性历史主键扩展为 `(file_hash, collection_name, processing_profile)`。profile 由 loader、splitter、embedding、关键 ingestion 参数和处理版本生成稳定哈希。

## MCP 与结构

MCP 工具改为 async wrapper，通过 `asyncio.to_thread` 执行同步 Pipeline、SQLite、模型和网络调用。调用方显式传入的 `top_k` 优先于默认配置。Cross-Encoder 超时不再依赖会等待退出的 executor 上下文。

`IngestionPipeline` 保留 Facade，对外构造参数和公开方法不变，内部拆成 builder、transform runner、encoder 和 storage coordinator。`SQLiteVectorStore` 保留 Facade，内部拆成 schema、record repository、search、image repository。三个 ingest 工具改用共享 `IngestionService`，各自只负责 loader 选择和参数声明。

## 发布结构

正式 Python 包迁移到 `src/modular_rag_mcp_server/`，所有生产代码、测试和脚本改用 `modular_rag_mcp_server.*`。配置和 prompts 移入包资源并通过 `importlib.resources` 加载，外部配置路径仍可覆盖默认资源。

`pyproject.toml` 增加：

- `modular-rag-mcp-server` CLI entry point。
- package data 声明。
- 正确的运行依赖和可选 extras。
- pytest 仅保留在 dev extra。

CI 在支持的 Python 版本上运行测试、静态检查、wheel 构建和隔离安装 smoke test。

## 验收

- 所有已确认的回归场景均有自动测试。
- 完整测试无失败，已解释或消除 skipped 和第三方弃用警告。
- wheel 在仓库外安装后可启动 CLI、加载默认配置和 prompts。
- 测试结束后完整性历史和业务测试数据库为空。
- 真实业务库在备份后按用户要求清空，记录数验证为 0。
