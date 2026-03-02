# 待办事项

## 优先级说明

- **P0**：严重（安全/数据错误/阻塞使用）
- **P1**：高（功能错误/影响正确性）
- **P2**：中（体验/架构）
- **P3**：低（优化/可选）

---

## P0 严重

- [x] **配置中 API Key 硬编码**（已完成）
  - 问题：settings.yaml 中敏感信息直接提交，存在泄露风险
  - 修复：settings.local.yaml 覆盖、.gitignore 排除

---

## P1 高

- [x] **list_collections 数据源错误**（已完成）
  - 问题：从错误路径列出集合，与 query、ingest 不一致
  - 修复：改为 settings.ingestion.bm25_base_path

- [x] **ingest 单文件失败时的 collection 一致性**（已完成）
  - 问题：BM25 失败时向量已写入，导致 collection 半写入状态
  - 修复：BM25 失败时回滚已写入的向量

---

## P2 中（功能）

- [ ] **按需查看任意文档**
  - 问题：缺少 list_documents，无法浏览集合下的文档列表
  - 待办：list_documents + get_document_summary 聚合

- [ ] **新增 MCP 上传工具**
  - 问题：无法通过 MCP 直接上传文件入库
  - 待办：实现 ingest_document，支持 Base64 上传

- [ ] **MinerU PDF 解析**
  - 问题：复杂 PDF 仅靠 MarkItDown 解析效果差
  - 待办：新增 MinerULoader，可选 MinerU 解析

- [x] **BM25 提前校验**（已完成）
  - 问题：索引不存在时错误在 SparseRetriever._get_indexer → BM25Indexer.load 深处抛出，导致 (1) Dense 检索已执行完毕造成向量检索资源浪费，(2) FileNotFoundError 提示不友好，用户不知需先 run ingest
  - 修复：SparseRetriever 新增 index_exists()，HybridSearch 入口在执行 Dense 前校验，不存在时立即抛出友好 ValueError

- [x] **MCP 工具路径统一**（已完成）
  - 问题：list_collections、get_document_summary 硬编码 base_path
  - 修复：从 settings 读取 bm25_base_path

- [ ] **Pipeline 仅支持 PDF**
  - 问题：Loader 固定 PdfLoader，无法 ingest txt、docx 等
  - 待办：LoaderFactory 或扩展 PdfLoader

- [ ] **list_collections 扩展**
  - 问题：仅返回集合名，无统计信息
  - 待办：附带 chunk 数、文档数、更新时间等

---

## P3 低（优化）

- [ ] **BM25 优化**
  - 问题：k1、b 等参数不可配置，分词器未优化
  - 待办：参数可配置化、分词器优化

- [ ] **配置热加载**
  - 问题：修改 settings.yaml 后需重启 MCP 才生效
  - 待办：支持运行时热加载

- [ ] **SparseRetriever 索引缓存**
  - 问题：_indexers 无限增长，多 collection 时占内存
  - 待办：LRU 或 max_collections 上限

- [ ] **向量库与 BM25 原子性**（暂不实现）
  - 问题：存储顺序与失败补偿复杂
  - 暂不实现

---

## 代码审查待办

### 已完成（归档）

| 项目 | 问题 | 修复 |
|------|------|------|
| query_knowledge_hub set_pipeline | 测试注入 mock 后 _cached_settings 未清空，images_base_path 用错配置 | 清空 _cached_settings |
| get_document_summary 异常静默 | except 吞掉 JSON/IO 错误，排查困难 | 添加 logger.warning |
| query_knowledge_hub 冗余调用 | 为取 settings 重复调用 _get_pipeline() | 直接用 _cached_settings |
| build_mcp_content_with_images 冗余 | 与 build_mcp_content 逻辑重复 | 删除，统一用 build_mcp_content |
| main.py / start_dashboard config | 硬编码路径，不支持 MODULAR_RAG_CONFIG_PATH | 改用 load_mcp_settings() |
| ingestion pipeline mark_success | 标记失败静默，可能导致重复处理 | 添加 logger.warning |
| qdrant_store / image_captioner | except pass 静默吞异常 | 补 logger.debug |
| pdf_loader print | 用 print 输出警告，污染 stdout | 3 处改为 logger.warning |
| batch_processor trace | 批次耗时未写入 trace，Dashboard 看不到 | 每批次 record_stage |
| reranker except 范围 | 捕获所有 Exception，无法区分错误类型 | 区分预期异常并打 logger.warning |
| start_dashboard except | 配置失败静默回退端口，掩盖配置错误 | logger.warning 后再回退 8501 |
