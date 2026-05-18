"""Git utilities for simulation metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit(repo_root: Path | str | None = None) -> str | None:
    """Return HEAD commit hash (40-char SHA) or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root) if repo_root else None,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def git_dirty(repo_root: Path | str | None = None) -> bool:
    """Check if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            capture_output=True,
            cwd=str(repo_root) if repo_root else None,
            timeout=5,
        )
        return result.returncode != 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
