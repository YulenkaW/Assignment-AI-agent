"""Shared workspace-owned paths for assignment agent artifacts."""

from __future__ import annotations

from pathlib import Path
import hashlib
import shutil


class WorkspacePaths:
    """Create stable paths for build and cache artifacts outside target repos."""

    def __init__(self, repository_path: Path, workspace_root: Path) -> None:
        self.repository_path = repository_path
        self.workspace_root = workspace_root
        self.base_directory = workspace_root / ".assignment_agent_work" / self._build_repository_key(repository_path)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def get_build_directory(self, variant_name: str = "default") -> Path:
        """Return the external build directory for the repository and backend variant."""
        normalized_variant = variant_name.strip().lower() or "default"
        build_directory = self.base_directory / normalized_variant / "build"
        build_directory.mkdir(parents=True, exist_ok=True)
        return build_directory

    def reset_build_directory(self, variant_name: str = "default") -> Path:
        """Delete and recreate one workspace-owned build directory."""
        build_directory = self.get_build_directory(variant_name)
        if build_directory.exists():
            shutil.rmtree(build_directory, ignore_errors=True)
        build_directory.mkdir(parents=True, exist_ok=True)
        return build_directory

    def get_index_cache_path(self) -> Path:
        """Return the cache path for the repository index."""
        cache_directory = self.base_directory / "cache"
        cache_directory.mkdir(parents=True, exist_ok=True)
        return cache_directory / "repository_index.pkl"

    def _build_repository_key(self, repository_path: Path) -> str:
        """Build a stable per-repository key."""
        digest = hashlib.sha1(str(repository_path).encode("utf-8")).hexdigest()[:12]
        return f"repo_{digest}"
