# 待办事项

## 优先级说明

- **P0**：严重（安全/数据错误/阻塞使用）
- **P1**：高（功能错误/影响正确性）
- **P2**：中（体验/架构）
- **P3**：低（优化/可选）

---

## P0 严重

- [x] **配置中 API Key 硬编码**（已完成）
  - 新增 `config/settings.local.yaml`（已加入 .gitignore），`load_settings` 会深度合并覆盖
  - `settings.yaml` 中敏感项改为空字符串，提供 `settings.local.yaml.example` 模板

---

## P1 高

- [x] **list_collections 数据源错误**（已完成）
  - 改为从 `settings.ingestion.bm25_base_path` 列出子目录，与 query、ingest 一致

- [x] **ingest 单文件失败时的 collection 一致性**（已完成）
  - BaseVectorStore 增加 delete 接口；ChromaStore/QdrantStore/FakeVectorStore 实现 delete
  - Pipeline._store_results：BM25 失败时回滚已写入的向量（按 chunk_id 删除）

---

## P2 中

### 功能

- [ ] **按需查看任意文档**：list_documents + get_document_summary 聚合，二者配合支撑文档浏览
  - list_documents(collection_name)：列出该集合下所有文档（doc_id + title），从 BM25 chunk_metadata 汇总
  - get_document_summary：聚合该 doc 下所有 chunk 的 title/summary/tags，返回文档级摘要（替代当前仅取首个 chunk）

- [ ] **新增 MCP 上传工具**：实现 `ingest_document` 工具，支持通过 MCP 上传文件并入库
  - 入参：`file_content`（Base64）、`file_name`、`collection_name`、`pdf_parser`（可选，markitdown | mineru）
  - 流程：解码 → 临时文件 → 调用 IngestionPipeline.process(pdf_parser=...) → 清理临时文件
  - 注意：大文件 Base64 体积大，建议限制文件大小（如 10MB）

- [ ] **MinerU PDF 解析**：新增 MinerULoader，复杂 PDF 可选 MinerU 解析，与现有 Pipeline 统一
  - 配置：`ingestion.pdf_loader`（markitdown | mineru）、`ingestion.mineru_api_url`
  - Pipeline：按 pdf_parser 选择 Loader；ImageCaptioner 对 mineru 已有 caption 时跳过 Vision LLM

  **方案 A：API 调用**
  - 实现：MinerULoader 通过 HTTP 调用 mineru-api 的 `/file_parse`
  - 优点：实现简单（80–120 行）、依赖少（仅 requests）、易维护
  - 缺点：需单独启动 mineru-api 服务；在 Cursor 等 IDE 中用户需手动开终端执行 `mineru-api --port 8000`

  **方案 B：SDK 调用**
  - 实现：MinerULoader 直接 import mineru/magic_pdf 模块，在进程内解析
  - 优点：无需单独服务，Cursor 配置后即可用
  - 缺点：实现复杂（150–250 行）、依赖多且重、需下载本地模型、随 MinerU 升级维护成本高

- [ ] **BM25 提前校验**：在 HybridSearch.search() 入口校验 collection 对应的 BM25 索引是否存在
  - SparseRetriever 新增 `index_exists(collection_name) -> bool`
  - 索引不存在时抛出友好错误

- [ ] **MCP 工具路径统一**：list_collections、get_document_summary 的 base_path 应从 settings 读取
  - 当前 list_collections 硬编码 data/documents，get_document_summary 用默认值

- [ ] **Pipeline 仅支持 PDF**：Loader 固定为 PdfLoader，无法 ingest 其他格式（txt、docx 等）
  - 可选：LoaderFactory 按扩展名路由，或扩展 PdfLoader 支持更多格式

- [ ] **list_collections工具修复与集合统计与简介**：当前仅返回集合名，扩展为附带每个集合的简单统计与介绍
  - 可含：chunk 数量、文档数量、索引更新时间等
  - 涉及：`list_collections.py`，需从 BM25/向量库或索引元数据读取统计

---

## P3 低

- [ ] **BM25 优化**：参数调优（k1、b）可配置化、分词器优化、性能优化

- [ ] **配置热加载**：MCP 启动后修改 settings.yaml 不生效，需重启

- [ ] **SparseRetriever 索引缓存无淘汰**：`_indexers` 无限增长，多 collection 时占内存
  - 可选：LRU 或 max_collections 上限

- [ ] **PdfLoader 使用 print 而非 logging**：多处 `# TODO: 使用 logging 记录警告`，当前用 print
  - 涉及：`pdf_loader.py`

- [ ] **BatchProcessor 未将批次耗时记录到 trace**：`# TODO: 将批次耗时记录到 trace`

- [ ] **向量库与 BM25 原子性**（暂不实现）
  - 存储顺序与异常处理、失败补偿需 VectorStore.delete 等，复杂度高

---

## 代码审查发现（按优先级）

### P1 高

- [x] **query_knowledge_hub：set_pipeline 未清理 _cached_settings**（已完成）
  - `set_pipeline(mock)` 注入后，`_cached_settings` 仍可能是旧值，解析 images_base_path 时用错配置
  - 修复：`set_pipeline` 中同时清空 `_cached_settings`

- [x] **get_document_summary：异常静默吞掉**（已完成）
  - `except Exception: return {}` 会隐藏 JSON 解析、权限、IO 等错误，排查困难
  - 修复：添加 `logger.warning("加载 chunk_metadata 失败 %s: %s", index_file, e)` 再 return

### P2 中

- [ ] **query_knowledge_hub：冗余 _get_pipeline() 调用**
  - L124–128 为取 settings 再次调用 `_get_pipeline()`，L115 已调用过
  - 建议：直接 `settings = _cached_settings`，为 None 时再 `load_mcp_settings()`

- [ ] **build_mcp_content_with_images 冗余**
  - `response_builder.build_mcp_content` 已通过 `assemble_content` 处理 image_refs
  - `multimodal_assembler.build_mcp_content_with_images` 逻辑重复，仅测试使用
  - 建议：删除或改为调用 `build_mcp_content`

- [x] **main.py / start_dashboard 硬编码 config 路径**（已完成）
  - 未使用 `MODULAR_RAG_CONFIG_PATH`，与 MCP 工具 config_utils 不一致
  - 修复：改用 `load_mcp_settings()`

- [ ] **get_document_summary：tags 为 dict 时处理不当**
  - `tags = [str(tags)]` 会把 `{"key":"val"}` 转成 `["{'key': 'val'}"]`
  - 建议：`tags` 非 list 时统一当作 `[]`，或加 `logger.warning` 提示非法类型

- [x] **ingestion pipeline：mark_success 失败静默**（已完成）
  - L268 `except Exception: pass`，标记成功失败可能导致重复处理
  - 修复：添加 logger.warning

- [x] **qdrant_store / image_captioner 的 except pass**（已完成）
  - `qdrant_store.py` L123、L303；`image_captioner.py` L313 静默吞异常
  - 修复：补 logger.debug

### P3 低

- [ ] **pdf_loader：TODO 使用 logging**
  - 多处 `# TODO: 使用 logging 记录警告`，当前用 print

- [ ] **batch_processor：TODO 批次耗时记录到 trace**
  - L116 已有 TODO，trace 已接入，可补充批次耗时打点

- [ ] **reranker：except Exception 范围过大**
  - 精排失败直接 fallback，可区分超时等异常类型单独处理

- [ ] **start_dashboard：except Exception 范围过大**
  - L40 捕获所有异常并回退端口 8501，可能掩盖配置错误，建议至少打日志
