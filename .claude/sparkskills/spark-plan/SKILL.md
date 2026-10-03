---
name: spark-plan
description: Create technical implementation plan
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Plan

## Purpose
Convert the spec into a detailed technical implementation plan with architecture, tech stack, and task breakdown.

## When to Use
- After spec is approved
- Before breaking down into tasks
- When starting implementation phase

## Process

1. **Review spec** - Understand requirements and constraints
2. **Choose tech stack** - Select frameworks, libraries, databases
3. **Design architecture** - Define components and their relationships
4. **Estimate effort** - Rough time estimates for each component
5. **Identify risks** - Potential blockers and mitigation

## Output
Creates `PLAN.md` with:
- Architecture diagram (Mermaid)
- Tech stack decisions
- Component breakdown
- API design (if applicable)
- Database schema (if applicable)
- Implementation milestones

## Usage Example
```
/spark.plan Use React with Vite, SQLite for storage, no external APIs
```

## Integration
Requires: spark-specify
Produces: PLAN.md
Feeds into: spark-tasks

## Guardrails
- Follow constitution principles
- Keep it simple (YAGNI principle)
- Consider testing strategy
- Plan for scalability if needed
