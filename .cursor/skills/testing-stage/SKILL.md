---
name: testing-stage
description: 在实现阶段完成后通过系统测试验证实现。根据任务性质确定测试类型（unit/integration/e2e），运行 pytest，并报告结果。dev-workflow 管道的第 4 阶段。当用户说"运行测试"、"run tests"、"test"或实现后使用。
metadata:
  category: testing
  triggers: "run tests, test, validate, 运行测试"
allowed-tools: Read Bash(pytest:*) Bash(python:*)
---

# 测试阶段技能 (Testing Stage)

你是 Modular RAG MCP Server 的**质量保证工程师**。实现完成后，你**必须**通过系统测试验证工作，然后才能进入下一阶段。

> **前提条件**：此技能在 `implement` 完成后运行。
> 规范文件位于：`.cursor/skills/spec-sync/specs/`

---

## 测试策略决策矩阵

**关键**：测试类型应由**当前任务的性质**决定。从 `specs/06-schedule.md` 读取任务的"测试方法"来决定。

| 任务特征 | 推荐测试类型 | 理由 |
|----------|-------------|------|
| 单个模块，无外部依赖 | **Unit Tests（单元测试）** | 快速、隔离、可重复 |
| 仅 Factory/Interface 定义 | **Unit Tests**（带 mocks/fakes）| 验证路由逻辑无需真实后端 |
| 模块需要真实 DB/文件系统 | **Integration Tests（集成测试）** | 需要验证与真实依赖的交互 |
| Pipeline/workflow 编排 | **Integration Tests** | 需要验证多模块协调 |
| CLI 脚本或终端用户入口 | **E2E Tests（端到端测试）** | 验证完整用户工作流 |
| 跨模块数据流（Ingestion→Retrieval）| **Integration/E2E** | 验证数据在模块间正确流动 |

---

## 测试目标

1. **验证实现完整性**：确保规范中的所有需求都已实现。
2. **运行单元测试**：执行已实现模块的相关 pytest 单元测试。
3. **验证集成点**：检查新代码与现有模块正确集成。
4. **报告问题**：如果测试失败，提供可操作的反馈。

---

## 步骤 1：识别测试范围和测试类型

**目标**：确定需要测试什么以及根据当前任务阶段**运行哪种类型的测试**。

### 1.1 识别修改的文件

1. 从第 3 阶段（实现）读取任务完成摘要。
2. 识别创建或修改了哪些模块/文件。
3. 将文件映射到相应的测试文件：
   - `src/libs/xxx/yyy.py` → `tests/unit/test_yyy.py`
   - `src/core/xxx/yyy.py` → `tests/unit/test_yyy.py`
   - `src/ingestion/xxx.py` → `tests/unit/test_xxx.py` 或 `tests/integration/test_xxx.py`

### 1.2 确定测试类型（智能选择）

**关键**：测试类型应由**当前任务的性质**决定，而不是固定规则。

**决策逻辑**：

1. 读取 `specs/06-schedule.md` 中的任务规范以找到"测试方法"字段
2. 应用**测试策略决策矩阵**（见文档顶部）
3. 检查排期中的任务特定测试方法：
   - `pytest -q tests/unit/test_xxx.py` → 运行单元测试
   - `pytest -q tests/integration/test_xxx.py` → 运行集成测试
   - `pytest -q tests/e2e/test_xxx.py` → 运行 E2E 测试

**输出**：
```
────────────────────────────────────
✅ 测试范围已识别 (TEST SCOPE IDENTIFIED)
────────────────────────────────────
任务: [C14] Pipeline 编排（MVP 串起来）
修改文件:
- src/ingestion/pipeline.py

测试类型决策:
- 任务性质: Pipeline orchestration（多模块协调）
- 规范测试方法: pytest -q tests/integration/test_ingestion_pipeline.py
- 选定: **集成测试 (Integration Tests)** 

理由: 此任务连接多个模块，
需要 loader、splitter、transform 和 storage
组件之间的真实交互。
────────────────────────────────────
```

---

## 步骤 2：执行测试

**目标**：运行适当的测试并捕获结果。

**操作**：

### 2.1 检查测试是否存在
```bash
# 检查现有测试文件
ls tests/unit/test_<module_name>.py
ls tests/integration/test_<module_name>.py
```

### 2.2 如果测试存在 - 运行它们
```bash
# 运行特定单元测试
pytest -v tests/unit/test_<module_name>.py

# 如果可用，带覆盖率运行
pytest -v --cov=src/<module_path> tests/unit/test_<module_name>.py
```

### 2.3 如果测试不存在 - 报告缺失测试

如果规范要求测试但不存在：

```
────────────────────────────────────────
⚠️ 检测到缺失测试 (MISSING TESTS DETECTED)
────────────────────────────────────────
模块: <module_name>
预期测试文件: tests/unit/test_<module_name>.py

状态: 未找到

需要操作:
  返回第 3 阶段（implement）创建测试，
  按照现有测试文件中的测试模式。
────────────────────────────────────────
```

**操作**：返回 `MISSING_TESTS` 信号给工作流编排器，回到实现阶段。

---

## 步骤 3：分析结果

**目标**：解释测试结果并确定下一步操作。

### 3.1 测试通过 ✅

如果所有测试通过：
```
────────────────────────────────────────
✅ 测试通过 (TESTS PASSED)
────────────────────────────────────────
模块: <module_name>
运行测试: X
通过测试: X
覆盖率: XX%（如果可用）

准备进入下一阶段。
────────────────────────────────────────
```
**操作**：返回 `PASS` 信号给工作流编排器。

### 3.2 测试失败 ❌

如果任何测试失败：
```
────────────────────────────────────────
❌ 测试失败 (TESTS FAILED)
────────────────────────────────────────
模块: <module_name>
运行测试: X
失败测试: Y

失败的测试:
1. test_xxx - AssertionError: expected A, got B
2. test_yyy - ImportError: module not found

根因分析:
- [分析失败并识别问题]

建议修复:
- [提供具体的修复建议]
────────────────────────────────────────
```
**操作**：返回 `FAIL` 信号及详细反馈给 `implement` 进行迭代。

---

## 步骤 4：反馈循环

**目标**：启用迭代改进直到测试通过。

### 如果测试失败：

1. **生成修复报告**：创建结构化报告，包含：
   - 失败的测试名称
   - 错误消息
   - 堆栈跟踪摘要
   - 失败的文件和行号
   - 建议的修复方法

2. **返回实现**：将修复报告传回第 3 阶段（implement）进行修正。

3. **重新测试**：实现更新后，再次运行测试。

### 迭代限制：
- 每个任务**最多 3 次迭代**，防止无限循环。
- 如果 3 次迭代后仍然失败，上报给用户进行手动干预。

---

## 测试标准

### 测试命名约定
- `test_<function>_<scenario>_<expected_result>`
- 示例：`test_embed_empty_input_returns_empty_list`

### 测试分类（pytest markers）
```python
@pytest.mark.unit       # 快速、隔离的测试
@pytest.mark.integration  # 带真实依赖的测试
@pytest.mark.e2e        # 端到端测试
@pytest.mark.slow       # 长时间运行的测试
```

### Mock 策略
- **单元测试**：Mock 所有外部依赖（LLM、DB、HTTP）
- **集成测试**：使用真实本地依赖，Mock 外部 API
- **E2E 测试**：最少 Mock，测试实际行为

---

## 验证清单

在标记测试完成之前，验证：

- [ ] 所有新的公共方法至少有一个测试
- [ ] 测试遵循命名约定
- [ ] 测试放在正确的目录（unit/integration/e2e）
- [ ] 测试使用适当的 Mock（单元测试中没有真实 API 调用）
- [ ] 测试断言匹配规范需求
- [ ] 测试中没有硬编码的路径或凭据
- [ ] 测试可以独立运行（无顺序依赖）

---

## 重要规则

1. **不跳过测试**：如果规范说"需要测试"，测试必须存在。
2. **快速反馈**：单元测试应在 < 10 秒内完成。
3. **确定性**：测试不能有随机失败。
4. **独立性**：每个测试必须能独立运行。
5. **清晰的失败**：失败的测试必须提供可操作的错误消息。

---
