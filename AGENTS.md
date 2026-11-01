# Repository Guidelines

## Repository language

All repository content must be written in English, including source code,
comments, docstrings, configuration files, documentation, test names, error
messages, commit messages, and pull request descriptions.

## Development workflow

- Use `uv` for Python environments and dependency management.
- Run `uv run pre-commit run --all-files` before requesting a commit.
- Do not commit changes unless the user explicitly requests a commit.
- Keep changes limited to the current approved work unit.
- Stop and ask the user before making an unresolved design decision.
