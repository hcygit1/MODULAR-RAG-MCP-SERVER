---
name: spec-sync
description: 同步 DEV_SPEC.md 并将其拆分为章节文件到 specs/ 目录。运行 sync_spec.py 进行更新，然后读取 SPEC_INDEX.md 进行导航。这是所有基于规范操作的基础。当用户说"同步规范"、"sync spec"时使用，或在任何依赖规范的任务之前使用。
metadata:
  category: documentation
  triggers: "sync spec, update spec, 同步规范"
allowed-tools: Bash(python:*) Read
---

# 规范同步 (Spec Sync)

此技能同步主规范文档（`DEV_SPEC.md`）并将其拆分为较小的章节文件，存储在 `specs/` 目录中。

> **这是所有基于规范操作的前提条件。** 其他技能依赖拆分后的规范文件来执行任务。

---

## 使用方式

### 在 dev-workflow 中使用（自动）

当你触发 dev-workflow（例如"下一阶段"或"继续开发"）时，**spec-sync 会自动作为第 1 阶段运行**。无需手动操作。

### 手动同步（仅边缘情况）

仅在以下情况手动运行：
- 你在工作流外部编辑了 `DEV_SPEC.md`
- 规范文件损坏或丢失
- 单独测试某个技能

```bash
# 普通同步
python .cursor/skills/spec-sync/sync_spec.py

# 强制重新生成（即使未检测到变更）
python .cursor/skills/spec-sync/sync_spec.py --force
```

---

### 同步脚本的工作原理

脚本执行以下操作：
1. 从项目根目录读取 `DEV_SPEC.md`
2. 计算哈希值以检测变更
3. 将文档拆分为 `specs/` 下的章节文件
4. 生成 `SPEC_INDEX.md` 作为导航索引

---

### 同步后：使用 SPEC_INDEX.md 导航

**使用 `SPEC_INDEX.md` 作为入口点**，了解每个规范文件包含的内容：

```
Read: .cursor/skills/spec-sync/SPEC_INDEX.md
```

该索引文件提供：
- 每个章节内容的摘要
- 快速定位所需规范的参考

然后从 `specs/` 目录读取你需要的具体规范文件：

```
Read: .cursor/skills/spec-sync/specs/05-architecture.md
```

---

## 目录结构

```
.cursor/skills/spec-sync/
├── SKILL.md              ← 此文件
├── SPEC_INDEX.md         ← 自动生成的索引（导航索引）
├── sync_spec.py          ← 同步脚本
├── .spec_hash            ← 用于变更检测的哈希文件
└── specs/                ← 拆分后的规范文件（章节文件）
    ├── 01-overview.md
    ├── 02-features.md
    ├── 03-tech-stack.md
    ├── 04-testing.md
    ├── 05-architecture.md
    ├── 06-schedule.md
    └── 07-future.md
```

---

## 重要提示

- **永远不要直接编辑 `specs/` 中的文件** — 它们是自动生成的
- **始终编辑 `DEV_SPEC.md`** 然后重新运行同步脚本
- 使用 `--force` 标志即使未检测到变更也强制重新生成：
  ```bash
  python .cursor/skills/spec-sync/sync_spec.py --force
  ```
