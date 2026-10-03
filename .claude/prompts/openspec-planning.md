# OpenSpec Skill Planning Guide

When executing OpenSpec skills, follow this pattern for creating plans and tasks:

## Pattern for Implementation Skills (e.g., `/opsx:apply`)

1. **Read and analyze** the change artifacts:
   - Create Task for reading proposal.md
   - Create Task for reading design.md
   - Create Task for reading tasks.md

2. **Break down tasks** from tasks.md:
   - Create sub-tasks for each item in tasks.md
   - Mark task as in_progress when starting
   - Mark as completed when done

3. **Update tasks.md** after implementation:
   - Check off completed items: `- [ ]` → `- [x]`

## Pattern for Creation Skills (e.g., `/opsx:new`, `/opsx:propose`)

1. **Analyze schema requirements** - what artifacts are needed
2. **Get instructions** for each artifact
3. **Create artifacts** in dependency order
4. **Verify** all artifacts are complete

## Task Naming Convention

Use clear, actionable names:
- "Read change artifacts: proposal, design, tasks"
- "Implement authentication flow"
- "Update tasks.md with completed items"

## When to Use Plan vs Direct Execution

- Use **EnterPlanMode** for complex multi-file changes
- Use **direct execution** for simple, well-defined tasks
