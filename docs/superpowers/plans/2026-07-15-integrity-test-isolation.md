# File Integrity Test Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent pytest from reading or writing the runtime file-integrity database while allowing `IngestionPipeline` to use an explicitly supplied checker.

**Architecture:** Add one optional constructor dependency to `IngestionPipeline`, preserving its current default behavior. Add a root autouse pytest fixture that redirects the module default database path to a per-test temporary file and resets the convenience-function singleton.

**Tech Stack:** Python 3.12, pytest, SQLite

---

### Task 1: Add a failing injection regression test

**Files:**
- Modify: `tests/integration/test_ingestion_pipeline.py`

- [ ] **Step 1: Update the sample PDF test to construct a real temporary `FileIntegrityChecker` and pass it as `integrity_checker` to `IngestionPipeline`.**

- [ ] **Step 2: Remove `force=True` and assert the injected checker records the sample file hash for `test_collection`.**

- [ ] **Step 3: Run the single test.**

Run:

```bash
.venv/bin/python -m pytest -q tests/integration/test_ingestion_pipeline.py::TestIngestionPipelineWithFixtures::test_pipeline_with_sample_pdf
```

Expected: FAIL with `TypeError` because `IngestionPipeline.__init__` does not yet accept `integrity_checker`.

### Task 2: Implement dependency injection

**Files:**
- Modify: `src/ingestion/pipeline.py`

- [ ] **Step 1: Add `integrity_checker: Optional[FileIntegrityChecker] = None` to the constructor.**

- [ ] **Step 2: Use the supplied checker when it is not `None`; otherwise create the existing default checker.**

- [ ] **Step 3: Re-run the focused integration test.**

Expected: PASS and the temporary checker reports the file as successfully processed in `test_collection`.

### Task 3: Isolate all pytest integrity state

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/integration/test_ingestion_pipeline.py`

- [ ] **Step 1: Add an autouse fixture using `tmp_path` and `monkeypatch` to redirect `src.libs.loader.file_integrity.DEFAULT_DB_PATH`.**

- [ ] **Step 2: Reset `_default_checker` before and after every test so convenience functions cannot retain another test's connection target.**

- [ ] **Step 3: Remove the remaining `force=True` integrity bypass from the sample Pipeline tests.**

- [ ] **Step 4: Run file-integrity and Pipeline tests.**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_file_integrity.py tests/integration/test_ingestion_pipeline.py
```

Expected: all tests pass.

### Task 4: Clear polluted history and verify the repository

**Files:**
- Runtime cleanup only: `cache/processing/file_integrity.db`

- [ ] **Step 1: Delete the ignored runtime integrity database without touching `data/db/rag.sqlite` or `data/db/chroma/chroma.sqlite3`.**

- [ ] **Step 2: Run the complete pytest suite.**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Confirm `cache/processing/file_integrity.db` was not recreated and `git diff --check` is clean.**

- [ ] **Step 4: Commit the implementation with a Chinese Conventional Commit message.**

```bash
git commit -m "test: 隔离文件完整性测试数据库"
```
