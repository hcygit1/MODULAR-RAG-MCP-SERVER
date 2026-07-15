# 文件完整性测试隔离设计

## 目标

修复测试直接读写 `cache/processing/file_integrity.db` 的问题，确保测试不会污染真实运行时的文件入库历史。

## 问题

`FileIntegrityChecker` 默认使用仓库内的相对路径数据库，`IngestionPipeline` 又在构造函数中硬编码创建检查器。当前单元测试和集成测试因此会写入真实缓存库，测试之间也可能因历史记录残留而互相影响。部分 Pipeline 测试使用 `force=True` 绕过完整性检查，只是掩盖了污染问题。

## 方案

- 为 `IngestionPipeline` 增加可选的 `integrity_checker` 构造参数；未传入时继续创建默认检查器，保持生产行为兼容。
- 在 pytest 根级夹具中，将 `DEFAULT_DB_PATH` 重定向到每个测试独立的临时目录，并重置便捷函数使用的单例。
- Pipeline 集成测试显式注入临时检查器，移除 `force=True`，验证真实的完整性记录写入。
- 清空当前 `cache/processing/file_integrity.db`。不修改 `data/db/rag.sqlite` 和 Chroma 数据库中的业务数据。

## 验收标准

- 新增测试在旧构造函数下因不支持 `integrity_checker` 参数而失败。
- Pipeline 能使用注入的检查器完成入库并记录指定集合的成功状态。
- 完整测试套件运行后，`cache/processing/file_integrity.db` 不会被重新创建或写入。
- 现有业务数据库和向量数据保持不变。
