---
name: spark-worktree
description: Automatic git worktree management
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Worktree

## Purpose
Automatically manage git worktrees for parallel development without context switching.

## When to Use
- Before starting new feature
- When you need to switch contexts
- When testing multiple branches

## Process

1. **Create worktree** - New branch for feature
2. **Verify baseline** - Tests pass on clean slate
3. **Work isolated** - Develop without affecting main
4. **Clean finish** - Merge or discard when done

## Benefits

- Parallel development without switch
- Clean isolation per feature
- Easy experimentation
- Safe to break without risk

## Usage Example
```
/spark.worktree
```

**Claude will:**
- Create new branch from main
- Set up worktree
- Run baseline tests
- Ready for development

## Guardrails

**Always:**
- Start from clean baseline
- Keep worktrees organized
- Clean up after finishing

**Never:**
- Commit to wrong branch
- Leave worktrees lying around
- Mix unrelated changes

## Commands

- `spark worktree new <name>` - Create new feature worktree
- `spark worktree list` - List active worktrees
- `spark worktree cleanup` - Remove old worktrees

## Integration
Triggered before: spark-implement
Finished with: spark-branch (merge/PR)
