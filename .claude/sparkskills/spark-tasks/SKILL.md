---
name: spark-tasks
description: Break plan into bite-sized tasks (2-5 min each)
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Tasks

## Purpose
Break implementation plan into small, actionable tasks that can be completed in 2-5 minutes each.

## When to Use
- After plan is approved
- Before starting implementation
- When need to manage scope

## Process

1. **Analyze plan** - Understand all components
2. **Decompose tasks** - Break into smallest meaningful units
3. **Prioritize** - Order by dependencies
4. **Estimate** - Ensure each is 2-5 minutes
5. **Document** - Create tasks.md with checkboxes

## Output
Creates/updates `TASKS.md` with:
```markdown
## Implementation Tasks

- [ ] Task 1: Brief description
- [ ] Task 2: Brief description
- [ ] Task 3: Brief description
```

## Usage Example
```
/spark.tasks
```

## Best Practices

**Good task:**
- "Add input validation to email field"
- "Write unit test for calculateTotal function"
- "Update README with installation steps"

**Bad task:**
- "Build authentication" (too big)
- "Fix bugs" (vague)
- "Improve performance" (no clear goal)

## Integration
Requires: spark-plan
Produces: TASKS.md
Feeds into: spark-implement
