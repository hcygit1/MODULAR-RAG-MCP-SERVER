---
name: implement
description: 按照规范驱动的工作流实现功能。先读取规范，提取设计原则，规划文件策略，然后编写带类型提示和文档字符串的生产级代码。当用户要求实现功能、编写代码或构建模块时使用。依赖 spec-sync 访问规范文档。
metadata:
  category: implementation
  triggers: "implement, write code, build module, 实现, 写代码"
allowed-tools: Read Write Bash(python:*) Bash(pytest:*)
---

# 标准操作流程：基于规范实现 (Implement from Spec)

你是 Modular RAG MCP Server 的首席架构师。当用户要求实现功能时，你**必须**遵循这个严格定义的工作流程。

> **前提条件**：此技能依赖 `spec-sync` 来访问规范文档。
> 规范文件位于：`.cursor/skills/spec-sync/specs/`

---

## 步骤 1：规范检索与分析

**目标**：使用渐进式披露的方式，将工作建立在权威的规范文档之上。

### 1.1 智能导航

不要阅读整个 `DEV_SPEC.md`，而是使用模块化方法：
- **首先**，读取 `.cursor/skills/spec-sync/SPEC_INDEX.md` 定位相关章节。
- **然后**，仅从 `.cursor/skills/spec-sync/specs/` 读取特定的章节文件。

### 1.2 提取任务特定需求

从目标章节中识别关键需求：
* **输入/输出**：期望什么数据类型？
* **依赖项**：是否需要特定的库？
* **修改文件**：此任务应创建或修改哪些文件？
* **验收标准**：此任务的验收条件是什么？

### 1.3 提取设计原则

**关键**：从规范中识别并提取与当前任务相关的设计原则。

**操作**：
1. 在 `specs/06-schedule.md` 中定位任务
2. 交叉参考 `specs/03-tech-stack.md` 或 `specs/05-architecture.md`
3. 提取适用的原则（Pluggable 可插拔、Config-Driven 配置驱动、Fallback 回退、Idempotent 幂等、Observable 可观测）
4. 在编码前记录原则

**输出模板**：
```
────────────────────────────────────
此任务的设计原则 (DESIGN PRINCIPLES)
────────────────────────────────────
任务: [Task ID] [Task Name]

适用原则:
1. [原则] - [实现要求]
2. [原则] - [实现要求]

来源: specs/XX-xxx.md 第 X.X 节
────────────────────────────────────
```

### 1.4 确认

明确告诉用户你查阅了哪个章节以及哪些原则适用。示例：
> *"我已审阅 `specs/03-tech-stack.md` 第 3.3.2 节。对于任务 B1（LLM Factory），适用的设计原则是：可插拔架构（抽象基类 + 工厂）、配置驱动（provider 来自 settings.yaml）、优雅错误处理。"*

**章节参考快速指南**（`.cursor/skills/spec-sync/specs/` 中的文件）：
- **架构问题** → `05-architecture.md`
- **技术实现细节** → `03-tech-stack.md`
- **测试需求** → `04-testing.md`
- **排期/进度跟踪** → `06-schedule.md`

---

## 步骤 2：技术规划

**目标**：在编写任何代码之前，确保模块化和设计原则合规。

1. **文件策略**：列出要创建或修改的文件（与排期中任务的"修改文件"字段交叉核对）。
2. **接口设计**：基于步骤 1.3 中提取的设计原则：
   - 如果适用 **Pluggable（可插拔）** 原则 → 先定义抽象基类
   - 如果适用 **Factory Pattern（工厂模式）** → 规划工厂函数签名
   - 如果适用 **Config-Driven（配置驱动）** → 识别 settings.yaml 需要的字段
3. **依赖检查**：如果需要新库，计划更新 `pyproject.toml` 或 `requirements.txt`。
4. **设计原则清单**：继续之前，验证你的计划解决了步骤 1.3 中的每个原则。

---

## 步骤 3：实现

**目标**：编写生产级、合规的代码。

1. **编码标准**：
   * **类型提示 (Type Hinting)**：所有函数签名必须有类型提示。
   * **文档字符串 (Docstrings)**：所有类和方法使用 Google 风格的文档字符串。
   * **禁止硬编码 (No Hardcoding)**：使用配置或依赖注入。
   * **Clean Code 原则**：
     - **单一职责 (Single Responsibility)**：每个函数/类只做一件事并做好
     - **短小精悍 (Short & Focused)**：函数应小（理想 < 20 行），类应内聚
     - **有意义的命名 (Meaningful Names)**：变量/函数名揭示意图（`getUserById` 而非 `getData`）
     - **无副作用 (No Side Effects)**：函数只做其名称所说的事，没有隐藏行为
     - **DRY**：抽象公共模式，避免重复
     - **快速失败 (Fail Fast)**：尽早验证输入，抛出清晰的异常
2. **错误处理**：为外部集成（LLM、数据库）实现健壮的 try/except 块。

---

## 步骤 4：自我验证（测试前）

**目标**：在移交给 testing-stage 之前进行自我修正和设计原则合规检查。

> **范围**：这是静态验证（代码审查，不是执行）。实际测试执行在第 4 阶段（testing-stage）进行。

1. **规范合规检查**：生成的代码是否违反了步骤 1 中发现的任何约束？
2. **设计原则合规检查**：验证步骤 1.3 中的每个原则是否已实现：
   - [ ] 如果 **Pluggable** → 是否有抽象基类？实现可以替换吗？
   - [ ] 如果 **Factory Pattern** → 工厂是否根据配置正确路由？
   - [ ] 如果 **Config-Driven** → 所有魔法值是否移到 settings.yaml？
   - [ ] 如果 **Fallback** → 失败时是否有优雅降级？
   - [ ] 如果 **Idempotent** → 操作是否可安全重复？
3. **测试文件验证**：确保测试文件已创建，具有正确的结构（导入、测试用例）
4. **改进**：如果使用了像 `pass` 这样的占位符，用工作逻辑或带 TODO 注释的清晰 `NotImplementedError` 替换。
5. **最终输出**：总结应用了哪些设计原则：
   ```
   ────────────────────────────────────
   ✅ 已应用的设计原则 (DESIGN PRINCIPLES APPLIED)
   ────────────────────────────────────
   [x] Pluggable: 定义了 BaseLLM 抽象类
   [x] Factory: LLMFactory.create() 根据 provider 路由
   [x] Config-Driven: Provider 从 settings.llm.provider 读取
   [x] Error Handling: 未知 provider 抛出 ValueError
   ────────────────────────────────────
   ```

---
