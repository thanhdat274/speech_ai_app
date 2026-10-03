---
name: spark-debug
description: Systematic 4-phase root cause analysis
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Debug

## Purpose
Systematic debugging process to find and fix root causes, not just symptoms.

## When to Use
- When code doesn't work
- When bugs appear unexpectedly
- When debugging is taking too long

## Process: 4 Phases

### Phase 1: Observation
- What is happening?
- What is expected?
- Collect evidence (logs, errors, behavior)

### Phase 2: Isolation
- Narrow down the scope
- Binary search the problem
- Identify the minimal failing case

### Phase 3: Root Cause
- Why did this happen?
- What condition triggered it?
- Trace back to origin

### Phase 4: Fix & Prevent
- Fix the root cause
- Add tests to prevent regression
- Update constitution if process gap found

## Output
Debug report with:
- Problem description
- Root cause analysis
- Fix applied
- Prevention measures

## Usage Example
```
/spark.debug
```

## Techniques

### Defense in Depth
- Don't rely on single verification
- Multiple checks prevent failures

### Condition-Based Waiting
- Wait for correct conditions, not arbitrary time
- Prevents race conditions

### Root Cause Tracing
- Follow the chain of causation
- Ask "why" 5 times

## Integration
Triggered after: spark-implement (if issues)
Can feed back to: spark-tasks (new task)
