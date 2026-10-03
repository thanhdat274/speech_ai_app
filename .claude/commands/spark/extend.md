# spark extend

Create new SparkSkill extension.

## Usage

```
/spark.extend
```

## Output

Creates new extension in `.spark/extensions/` with:
- SKILL.md template
- catalog.json entry
- Documentation

## Example

```
/spark.extend my-extension
```

Creates `extensions/my-extension/` with all templates.

## Extension Types

- `@spark/docs` - Read/write spec artifacts
- `@spark/code` - Review/modify source
- `@spark/process` - Orchestrate workflow
- `@spark/integration` - External sync
- `@spark/visibility` - Health reports
