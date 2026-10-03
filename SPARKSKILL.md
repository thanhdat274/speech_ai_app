# SparkSkill

> Structure + Automation + Quality = Developer Happiness

SparkSkill là bộ skill vượt trội kết hợp best-of-breed từ **OpenSpec**, **spec-kit**, và **superpowers**.

## 🌟 Core Philosophy

SparkSkill thay thế "vibe coding" bằng quy trình có cấu trúc:

```
brainstorm → constitution → spec → plan → tasks → implement → review → test
```

## 📚 Skills Reference

### Creation Skills

| Skill | Command | Description |
|-------|---------|-------------|
| `spark-brainstorm` | `/spark.brainstorm` | Refine ideas through Socratic questioning |
| `spark-constitution` | `/spark.constitution` | Create project principles |
| `spark-specify` | `/spark.specify` | Describe WHAT to build |
| `spark-plan` | `/spark.plan` | Technical implementation plan |

### Implementation Skills

| Skill | Command | Description |
|-------|---------|-------------|
| `spark-tasks` | `/spark.tasks` | Break into bite-sized tasks |
| `spark-implement` | `/spark.implement` | Execute with TDD (RED-GREEN-REFACTOR) |
| `spark-review` | `/spark.review` | Two-stage code review |
| `spark-debug` | `/spark.debug` | Systematic root cause analysis |

### Workflow Skills

| Skill | Command | Description |
|-------|---------|-------------|
| `spark-worktree` | `/spark.worktree` | Git worktree management |
| `spark-init` | `/spark.init` | Initialize new project |

## 🚀 Quick Start

### 1. Initialize Project

```bash
/spark.init my-project
```

### 2. Create Constitution

```bash
/spark.constitution
```

Establish principles for code quality, testing, and architecture.

### 3. Specify Requirements

```bash
/spark.specify
```

Describe what to build (focus on WHAT, not HOW).

### 4. Plan Implementation

```bash
/spark.plan
```

Specify tech stack and architecture.

### 5. Break Into Tasks

```bash
/spark.tasks
```

Create actionable task list.

### 6. Implement

```bash
/spark.implement
```

Execute tasks with TDD cycle.

## 📁 Project Structure

```
project/
├── SPARKSKILL.md          # This file
├── CONSTITUTION.md        # Project principles
├── SPEC.md                # Requirements
├── PLAN.md                # Technical plan
├── TASKS.md               # Implementation tasks
└── src/                   # Source code
```

## 🔧 Workflow

### Phase 1: Discovery

1. **Brainstorm** - Explore requirements
2. **Constitution** - Set principles
3. **Specify** - Document requirements

### Phase 2: Planning

4. **Plan** - Technical architecture
5. **Tasks** - Break down work

### Phase 3: Implementation

6. **Implement** - TDD cycle
7. **Review** - Code review
8. **Debug** - Fix issues if any

## 💡 Key Features

### Constitution System

Project principles guide all development decisions.

### TDD-First

Every task follows RED-GREEN-REFACTOR:
1. Write failing test
2. Write minimal code
3. Refactor safely

### Systematic Debugging

4-phase process:
1. Observation
2. Isolation
3. Root cause
4. Fix & prevent

### Extension Ready

Future support for community extensions like spec-kit.

## 🆚 Comparison

| Feature | OpenSpec | spec-kit | superpowers | **SparkSkill** |
|---------|----------|----------|-------------|----------------|
| Artifact workflow | ✅ | ✅ | ❌ | ✅ |
| Constitution | ❌ | ✅ | ❌ | ✅ |
| TDD enforcement | ❌ | ❌ | ✅ | ✅ |
| Auto-trigger skills | ❌ | ❌ | ✅ | ✅ |
| CLI commands | ⚠️ Manual | ✅ | ⚠️ Manual | ✅ |
| Extension system | ❌ | ✅ | ❌ | 🚧 Coming |

## 🎯 Best Practices

1. **Start with constitution** - Set principles first
2. **Small tasks** - 2-5 minutes each
3. **TDD always** - Tests first
4. **Review every change** - Catch issues early
5. **Document decisions** - Keep PLAN.md updated

## 📖 References

- [spec-kit](https://github.com/github/spec-kit) - Spec-Driven Development
- [superpowers](https://github.com/obra/superpowers) - Agentic skills framework
