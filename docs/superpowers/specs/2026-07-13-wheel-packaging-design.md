# Wheel 子包打包修复设计

## 目标

修复项目构建的 wheel 仅包含 `src/__init__.py`、缺少业务子包的问题，使正式安装后的包能够导入现有 `src.*` 模块。

## 范围

本次保留当前 `src` 顶层包名和所有现有导入路径，仅修改 setuptools 包发现配置，并增加针对实际 wheel 内容的回归测试。

本次不处理以下事项：

- 不迁移到 `src/modular_rag_mcp_server/` 标准包结构。
- 不修改 `config/settings.yaml` 的加载方式。
- 不打包或迁移 `config/prompts/`。
- 不增加命令行 entry point。

这些事项分别属于包命名、资源路径和命令入口问题，后续单独处理。

## 实现方案

将 `pyproject.toml` 中显式声明单个包：

```toml
[tool.setuptools]
packages = ["src"]
```

改为 setuptools 自动发现 `src` 及其子包。发现范围只包含 `src` 和 `src.*`，避免误打包测试、脚本或本地目录。

## 回归测试

新增一个集成测试，在临时目录中执行真实 wheel 构建，然后使用 `zipfile` 检查 wheel 内容。

测试至少验证以下代表性模块存在：

- `src/core/settings.py`
- `src/ingestion/pipeline.py`
- `src/libs/vector_store/base_vector_store.py`
- `src/mcp_server/server.py`
- `src/observability/logger.py`

测试使用当前解释器调用 `python -m pip wheel`，避免硬编码 `python` 命令，并使用 `--no-deps --no-build-isolation`，防止测试下载依赖。

## 验收标准

- 新增测试在旧配置下因 wheel 缺少业务子包而失败。
- 修改包发现配置后新增测试通过。
- 构建出的 wheel 包含全部现有 Python 子包。
- 在仓库目录之外，可以从 wheel 解压目录导入 `src.mcp_server.server`。
- 现有核心导入、检索管线和 MCP 集成测试继续通过。

## 风险控制

自动发现仅匹配 `src` 和 `src.*`，不会包含 `tests`、`scripts`、`data`、`model` 等目录。保留原有包名可以避免本次修改触发全仓库导入迁移。
