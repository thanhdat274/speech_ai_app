---
name: spark-specify
description: Describe what to build (what/why, not how)
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Specify

## Purpose
Create a clear specification focusing on **what** to build and **why**, not technical implementation details.

## When to Use
- After constitution is established
- Before creating technical plan
- When starting a new feature

## Process

1. **Capture requirements** - What problem are we solving?
2. **Define user stories** - Who is this for and how will they use it?
3. **Identify constraints** - What are the limitations?
4. **Document acceptance criteria** - How do we know it's done?

## Output
Creates `SPEC.md` with:
- Problem statement
- User stories
- Functional requirements
- Non-functional requirements
- Acceptance criteria

## Usage Example
```
/spark.specify Build a photo album organizer that groups photos by date
```

## Integration
Requires: spark-constitution
Produces: SPEC.md
Feeds into: spark-plan

## Guardrails
- Focus on WHAT and WHY
- Avoid tech stack details (that's for spark.plan)
- Be specific about user experience
- Include edge cases
