---
name: release
description: Bumps the project patch version, commits the version changes, and tags the commit with an informative release message.
---

# Release Skill

This skill automates the release process for the `syntx` package.

## Procedure

1. **Verify State:** Ensures the git status is clean before proceeding.
2. **Read & Bump Version:** Reads the current version from `pyproject.toml`, parses it, and increments the patch version (e.g. `0.1.1` -> `0.1.2`).
3. **Write Changes:** Updates the version string in:
   - `pyproject.toml`
   - src/syntx/__init__.py
4. **Git Commit & Tag:**
   - Stages the updated files.
   - Commits with the message: `release: bump version to v<new_version>`.
   - Creates an annotated Git tag: `v<new_version>` with the message `Release version v<new_version>`.

## Usage

Run the release script:
```bash
python .agents/skills/release/scripts/release.py
```
