"""Lightweight `.env` loading for local assignment-agent runs."""

from __future__ import annotations

from pathlib import Path
import os


def load_project_env(start_path: Path | None = None) -> list[Path]:
    """Load `.env` files from likely project locations without overriding shell vars."""
    loaded_paths = []
    seen_paths = set()
    for root in _candidate_roots(start_path):
        for env_path in (root / ".env", root / ".venv" / ".env"):
            resolved_path = env_path.resolve()
            if resolved_path in seen_paths or not resolved_path.is_file():
                continue
            seen_paths.add(resolved_path)
            _load_env_file(resolved_path)
            loaded_paths.append(resolved_path)
    return loaded_paths


def _candidate_roots(start_path: Path | None) -> list[Path]:
    """Return likely project roots, preferring the current working directory."""
    roots = []
    seen_paths = set()
    for path in _expand_candidate_paths(start_path):
        for candidate in (path, *path.parents):
            resolved_path = candidate.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            roots.append(resolved_path)
    return roots


def _expand_candidate_paths(start_path: Path | None) -> list[Path]:
    """Build the list of paths whose parents should be searched."""
    if start_path is not None:
        resolved_start = start_path.resolve()
        paths = [resolved_start.parent if resolved_start.is_file() else resolved_start, Path.cwd()]
        return paths
    paths = [Path.cwd()]
    return paths


def _load_env_file(env_path: Path) -> None:
    """Parse one `.env` file and set missing variables in `os.environ`."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)
