# Data Correctness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove document-replacement data loss and cross-collection/image corruption while making Chroma metadata valid.

**Architecture:** Add a backend capability for atomic document replacement and implement it transactionally in SQLite. Migrate SQLite chunk identity to `(id, collection_name)`, make image IDs document-scoped, require an explicit collection for image loading, and keep sparse vectors out of generic vector metadata.

**Tech Stack:** Python 3.12, SQLite, sqlite-vec, ChromaDB, pytest

---

### Task 1: Reproduce atomic replacement failure

**Files:**
- Modify: `tests/unit/test_vector_upserter_idempotency.py`
- Modify: `tests/unit/test_sqlite_vector_store.py`

- [ ] Add a store test that inserts an old document, forces the new vec insert to fail, and asserts the old document remains.
- [ ] Add an upserter test asserting `replace_document_records()` is called once instead of delete followed by upsert.
- [ ] Run the focused tests and confirm they fail for the expected old behavior.

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_vector_upserter_idempotency.py tests/unit/test_sqlite_vector_store.py
```

Expected: new replacement tests fail because the capability does not exist and old data is deleted before the failed write.

### Task 2: Implement transactional SQLite replacement

**Files:**
- Modify: `src/libs/vector_store/base_vector_store.py`
- Modify: `src/libs/vector_store/sqlite_store.py`
- Modify: `src/ingestion/storage/vector_upserter.py`

- [ ] Add `replace_document_records(source_doc_id, records, collection_name, trace=None, chunks_for_images=None) -> int` to the base class with a default `NotImplementedError`.
- [ ] Extract SQLite record/image writes into transaction-aware private helpers that never commit independently.
- [ ] Implement SQLite replacement in one transaction and return the actual old chunk count.
- [ ] Make `VectorUpserter.replace_document_chunks()` build records once and delegate to the capability.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Reproduce and migrate collection identity

**Files:**
- Modify: `tests/unit/test_sqlite_vector_store.py`
- Modify: `src/libs/vector_store/sqlite_store.py`

- [ ] Add a test writing the same id to two collections and querying/deleting them independently.
- [ ] Add a migration test starting from the legacy global-primary-key schema and preserving rowids/vector rows.
- [ ] Confirm both tests fail before implementation.
- [ ] Add an idempotent schema migration to `PRIMARY KEY (id, collection_name)`.
- [ ] Run SQLite tests and confirm both fresh and migrated databases pass.

### Task 4: Reproduce and fix image ownership

**Files:**
- Modify: `tests/unit/test_ingest_document_normal.py`
- Modify: `tests/unit/test_multimodal_assembler.py`
- Modify: `src/mcp_server/tools/ingest_document_normal.py`
- Modify: `src/core/response/multimodal_assembler.py`
- Modify: `src/mcp_server/tools/query_knowledge_hub.py`

- [ ] Add tests for two Markdown documents containing the same image filename.
- [ ] Add a test proving missing collection never loads an arbitrary duplicate image.
- [ ] Confirm the tests fail.
- [ ] Prefix Markdown image IDs with document id.
- [ ] Pass the effective query collection into response building and remove image-id-only collection inference.
- [ ] Run the image and MCP query tests.

### Task 5: Reproduce and fix Chroma metadata

**Files:**
- Modify: `tests/integration/test_chroma_store_roundtrip.py`
- Modify: `tests/unit/test_vector_upserter_idempotency.py`
- Modify: `src/ingestion/storage/vector_upserter.py`
- Modify: `src/libs/vector_store/chroma_store.py`

- [ ] Add a real Chroma upsert test using Pipeline-style metadata and a sparse vector input.
- [ ] Confirm Chroma rejects the old nested metadata.
- [ ] Remove `sparse_vector` from generic `VectorRecord.metadata`.
- [ ] Encode remaining non-scalar Chroma metadata as tagged JSON strings and decode it on query.
- [ ] Run Chroma and upserter tests.

### Task 6: Verify and commit

**Files:**
- Verify all modified production and test files.

- [ ] Run focused storage, image and Chroma tests.
- [ ] Run the complete pytest suite.
- [ ] Run `git diff --check`.
- [ ] Review the diff for private-field coupling and accidental behavior changes.
- [ ] Commit with:

```bash
git commit -m "fix: 修复向量数据替换与集合隔离"
```
