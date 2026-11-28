"""Isolated git-worktree/temp-copy lifecycle shared by race workers."""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
from pathlib import Path

from .models import BranchSpec

_IGNORED_COPY_PARTS = shutil.ignore_patterns(
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return cleaned or "branch"


def _git_output(root: Path, *args: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _clean_git_source(root: Path) -> bool:
    try:
        inside = _git_output(root, "rev-parse", "--is-inside-work-tree")
        top = _git_output(root, "rev-parse", "--show-toplevel")
        status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        return False
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return False
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        return False
    return not status.stdout.strip()


def _materialise_sandbox(spec: BranchSpec) -> tuple[Path, str]:
    parent = Path(spec.sandbox_parent)
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / f"{_safe_name(spec.run_id)}-{_safe_name(spec.branch_id)}"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    source = Path(spec.source_root)
    if _clean_git_source(source):
        added = _git_output(source, "worktree", "add", "--detach", str(dest.resolve()), "HEAD")
        if added.returncode == 0 and dest.is_dir():
            return dest, "git_worktree"
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(source, dest, ignore=_IGNORED_COPY_PARTS)
    return dest, "temp_copy"


def _cleanup_sandbox(spec: BranchSpec, dest: Path, mode: str) -> None:
    if mode == "git_worktree":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _git_output(
                Path(spec.source_root),
                "worktree",
                "remove",
                "--force",
                str(dest.resolve()),
            )
    shutil.rmtree(dest, ignore_errors=True)
