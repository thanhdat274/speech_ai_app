---
name: spark-review
description: Two-stage code review (spec compliance → quality)
version: "1.0"
author: SparkSkill
compatibility: Claude Code
---

# Spark Review

## Purpose
Systematic code review with two stages: spec compliance first, then code quality.

## When to Use
- After implementation task
- Before merging changes
- When requesting code review

## Process

### Stage 1: Spec Compliance

- [ ] Code meets requirements in SPEC.md
- [ ] All acceptance criteria satisfied
- [ ] No missing features
- [ ] Edge cases handled

### Stage 2: Code Quality

- [ ] Follows constitution standards
- [ ] Tests pass and have good coverage
- [ ] Code is readable and maintainable
- [ ] No unnecessary complexity
- [ ] Performance is acceptable

## Output
Creates review report with:
- Issues found (critical, major, minor)
- Suggestions for improvement
- Approval decision

## Usage Example
```
/spark.review
```

## Review Criteria

**Blocking (must fix):**
- Tests failing
- Spec not implemented
- Security issues

**Non-blocking (nice to have):**
- Style improvements
- Refactoring suggestions
- Performance optimizations

## Integration
Can be invoked after: spark-implement
Feeds into: spark-debug (if issues found)

## Checklist Template
```markdown
## Code Review

### Spec Compliance
- [ ] All features implemented
- [ ] Acceptance criteria met

### Code Quality
- [ ] Tests pass
- [ ] Code follows standards
- [ ] No code smells

### Decision
[ ] Approved [ ] Needs fixes
```
