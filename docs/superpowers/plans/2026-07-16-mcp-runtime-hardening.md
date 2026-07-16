# MCP Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make explicit MCP query limits take effect, prevent synchronous tools from blocking the FastMCP event loop, and make Cross-Encoder timeout return promptly without unbounded worker growth.

**Architecture:** Keep the existing synchronous `execute_*` functions as the reusable business layer. Expose async FastMCP wrappers that call them through `asyncio.to_thread`, treat request `top_k` as the final result limit while retaining configuration as candidate budgets, and allow at most one timed-out Cross-Encoder scoring task to continue in the background.

**Tech Stack:** Python 3.12, asyncio, FastMCP, concurrent.futures, pytest

---

### Task 1: Make request `top_k` authoritative

**Files:**
- Modify: `tests/integration/test_retrieval_pipeline.py`
- Modify: `src/core/query_engine/retrieval_pipeline.py`

- [ ] Add a test with `RetrievalConfig.top_k_final=10` and request `top_k=3`, asserting the final result and HybridSearch `top_k_final` are both 3.
- [ ] Add a test requesting more results than the configured dense/sparse candidate budgets and assert both budgets are raised to at least the request limit.
- [ ] Run the focused tests and confirm they fail because configuration currently overrides the request.
- [ ] Use request `top_k` as the final limit; use `max(configured_budget, top_k)` for non-parent dense/sparse candidates.
- [ ] In parent mode, keep `parent_aggregate_top_m` as the configured child budget but never below the requested parent count.
- [ ] Re-run retrieval tests and confirm they pass.

### Task 2: Offload all FastMCP tools

**Files:**
- Create: `tests/unit/test_mcp_async_wrappers.py`
- Modify: `src/mcp_server/tools/query_knowledge_hub.py`
- Modify: `src/mcp_server/tools/list_collections.py`
- Modify: `src/mcp_server/tools/ingest_document_normal.py`
- Modify: `src/mcp_server/tools/ingest_document_docling.py`
- Modify: `src/mcp_server/tools/ingest_document_mineru.py`
- Modify: `tests/integration/test_mcp_server.py`

- [ ] Add tests asserting every public FastMCP tool is an async callable.
- [ ] Patch each synchronous `execute_*` function with a blocking sentinel and prove an independent coroutine continues while the tool runs.
- [ ] Run the new tests and confirm they fail because all wrappers are synchronous.
- [ ] Convert each public wrapper to `async def` and call the existing executor through `await asyncio.to_thread(...)`.
- [ ] Update direct integration tests to await tools while preserving `CallToolResult` response shape.
- [ ] Run async wrapper, MCP integration, and E2E tests.

### Task 3: Make Cross-Encoder timeout prompt and bounded

**Files:**
- Modify: `tests/unit/test_cross_encoder_reranker.py`
- Modify: `src/libs/reranker/cross_encoder_reranker.py`

- [ ] Strengthen the timeout test to measure elapsed time and require return near the configured deadline rather than after scorer completion.
- [ ] Add a test that a second request while a timed-out scoring task is still running falls back immediately instead of starting another worker.
- [ ] Run the timeout tests and confirm the context-managed executor waits for the scorer and fails the elapsed-time assertion.
- [ ] Replace the executor context manager with one guarded active executor/future per reranker instance.
- [ ] On timeout, leave the uninterruptible worker to finish but return immediately; while it remains active, reject another scoring request immediately.
- [ ] Clean up a completed prior executor before accepting the next score operation.
- [ ] Run reranker and fallback tests.

### Task 4: Verify and commit

**Files:**
- Verify all files modified by Tasks 1-3.

- [ ] Run focused runtime tests.
- [ ] Run `.venv/bin/python -m compileall -q src scripts`.
- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Run `git diff --check`.
- [ ] Review the diff for public schema changes and accidental synchronous calls.
- [ ] Commit with a Chinese Conventional Commit message.
