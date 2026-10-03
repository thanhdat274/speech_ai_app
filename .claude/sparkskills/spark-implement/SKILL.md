---
name: spark-implement
description: Execute tasks with TDD cycle
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Implement

## Purpose
Implement tasks following Test-Driven Development: RED-GREEN-REFACTOR cycle.

## When to Use
- After tasks are created
- When ready to write code
- After all planning is complete

## Process

### TDD Cycle (per task)

1. **RED** - Write failing test first
2. **GREEN** - Write minimal code to pass
3. **REFACTOR** - Improve while tests still pass

### Execution

1. **Pick next task** - From TASKS.md
2. **Write test** - Define expected behavior
3. **Run test** - Watch it fail
4. **Write code** - Just enough to pass
5. **Run test** - Watch it pass
6. **Commit** - Clear commit message
7. **Move to next** - Repeat until done

## Output
- Implemented features
- Passing tests
- Updated TASKS.md (check completed tasks)

## Usage Example
```
/spark.implement
```

**Claude will:**
- Pick next unchecked task from TASKS.md
- Execute TDD cycle
- Commit changes
- Repeat until all done

## Guardrails

**Always:**
- Write tests first
- Keep commits small
- Update TASKS.md

**Never:**
- Write code without test
- Skip refactoring
- Forget to commit

## Integration
Requires: spark-tasks
Produces: Implemented code + tests
Updates: TASKS.md

## TDD Anti-Patterns to Avoid

- Testing implementation, not behavior
- Complex setup before test
- Multiple assertions per test
- Tests that depend on execution order
