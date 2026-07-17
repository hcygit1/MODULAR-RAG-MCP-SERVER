# Package Resources And CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将通用顶层包 `src` 迁移为正式 `modular_rag_mcp_server` 命名空间，使默认配置、prompts 和 CLI 在仓外安装 wheel 后可直接使用。

**Architecture:** 保留标准 `src` layout，生产代码位于 `src/modular_rag_mcp_server/`。统一资源模块使用 `importlib.resources` 读取包内 YAML/prompt；显式配置路径继续读取文件和同目录 local override，默认配置使用包资源并合并工作目录 local override。console script 调用包内 CLI，根 `main.py` 只保留兼容入口。

**Tech Stack:** Python 3.10+、setuptools、importlib.resources、argparse、pytest

---

### Task 1: 锁定安装态契约

**Files:**
- Modify: `tests/integration/test_distribution_package.py`
- Create: `tests/unit/test_package_resources.py`
- Create: `tests/unit/test_cli.py`

- [x] 将 wheel 内容断言改为 `modular_rag_mcp_server/*`，并要求包含 `resources/settings.yaml` 与四个 prompt。
- [x] 增加仓外 venv `pip install --no-deps <wheel>`，验证旧 `src` 不可导入、正式包可导入、console script `--check-config` 成功。
- [x] 增加资源读取测试，验证默认配置不依赖仓库 `config/settings.yaml`，显式路径和 `settings.local.yaml` 深度覆盖仍生效。
- [x] 增加 CLI 测试，验证 `--config` 优先级、`--check-config` 不启动 server、启动失败返回非零。
- [x] 运行新测试并确认因正式包、资源 API 和 CLI 不存在而 RED：

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_package_resources.py \
  tests/unit/test_cli.py \
  tests/integration/test_distribution_package.py
```

### Task 2: 迁移正式包命名空间

**Files:**
- Move: `src/{core,ingestion,libs,mcp_server,observability}` -> `src/modular_rag_mcp_server/`
- Delete: `src/__init__.py`
- Modify: all tracked `*.py` imports and patch target strings
- Modify: `pyproject.toml`

- [x] 使用 `git mv` 保留历史，将现有五个生产子包移入 `src/modular_rag_mcp_server/`。
- [x] 将 Python 源码、测试和脚本中的 `src.*` 机械替换为 `modular_rag_mcp_server.*`，保持原文件换行风格。
- [x] setuptools 改为标准 src layout：

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["modular_rag_mcp_server", "modular_rag_mcp_server.*"]
```

- [x] 刷新 editable install，并运行 `rg` 确认可执行 Python 文件无 `src.*` 导入残留。
- [x] 运行 smoke imports 和全部单元测试，修复任何真实导入边界问题。

### Task 3: 打包配置与 prompt 资源

**Files:**
- Create: `src/modular_rag_mcp_server/package_resources.py`
- Create: `src/modular_rag_mcp_server/resources/__init__.py`
- Move: `config/settings.yaml` -> `src/modular_rag_mcp_server/resources/settings.yaml`
- Move: `config/prompts/*.txt` -> `src/modular_rag_mcp_server/resources/prompts/*.txt`
- Modify: `src/modular_rag_mcp_server/core/settings.py`
- Modify: `src/modular_rag_mcp_server/mcp_server/tools/config_utils.py`
- Modify: prompt consumers under `ingestion/transform/` and `libs/reranker/`
- Modify: `pyproject.toml`

- [x] 实现 `read_resource_text(name)` 和 `read_prompt(name)`，拒绝绝对路径与 `..` 路径穿越。
- [x] `load_settings(path=None)` 默认解析包内 YAML；显式 path 保持原语义；默认分支可合并 `config/settings.local.yaml`。
- [x] MCP 配置优先级为测试 override、环境变量、包内默认配置。
- [x] prompt consumers 默认使用 `read_prompt`；显式 prompt path 继续读取调用方文件。
- [x] package-data 声明包含 `resources/*.yaml` 和 `resources/prompts/*.txt`。
- [x] 运行配置、prompt、processing profile 和 ingestion 相关测试。

### Task 4: 正式 CLI 与兼容入口

**Files:**
- Create: `src/modular_rag_mcp_server/cli.py`
- Modify: `main.py`
- Modify: `pyproject.toml`
- Modify: `scripts/ingest_mineru.py`
- Modify: `scripts/test_mineru_connection.py`
- Modify: `README.md`

- [x] 包内 CLI 使用 `argparse` 提供 `--config PATH` 和 `--check-config`。
- [x] CLI 始终把日志写入 stderr；配置/校验/启动失败返回 `1`，成功配置检查返回 `0`。
- [x] 增加 entry point：

```toml
[project.scripts]
modular-rag-mcp-server = "modular_rag_mcp_server.cli:main"
```

- [x] 根 `main.py` 只导入并调用包内 `main`；删除脚本中的 `sys.path.insert`。
- [x] README 更新安装、配置覆盖和 CLI 示例，目录树不再把 `src` 描述为包名。
- [x] 运行 CLI、MCP server、async wrapper 和 e2e 聚焦测试。

### Task 5: 分发与回归验收

**Files:**
- Modify: `.planning/project-hardening/{findings,progress,task_plan}.md`

- [x] 运行 Ruff、定向 mypy、compileall、`git diff --check` 和 CRLF 检查。
- [x] 构建 wheel，检查归档中只有正式包且资源/entry point 完整。
- [x] 在仓库外 venv 安装 wheel，运行正式 import、资源加载和 `modular-rag-mcp-server --check-config`。
- [x] 运行全量 pytest，并确认业务数据库与完整性历史计数未变化。
- [x] 独立规格/质量审查通过后提交并推送：

```bash
git commit -m "refactor: 迁移正式包与资源入口"
```
