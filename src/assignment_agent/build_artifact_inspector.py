"""Inspect generated build artifacts to answer read-only build questions."""

from __future__ import annotations

from pathlib import Path
import re


class BuildArtifactInspector:
    """Read generator outputs when command-based target listing is not portable."""

    TARGET_EXCLUSIONS = {
        "ZERO_CHECK",
    }

    # CTest emits lines such as `add_test(test_name ...)` in generated CMake files.
    TEST_PATTERN = re.compile(r"add_test\(([^)\s]+)")

    def __init__(self) -> None:
        self._build_target_cache = {}
        self._ctest_cache = {}

    def list_build_targets(self, build_directory: Path) -> list[str]:
        """Return available build targets from generated build artifacts."""
        build_directory = build_directory.resolve()
        signature = self._build_target_signature(build_directory)
        cached_targets = self._load_cached_targets(build_directory, signature)
        if cached_targets is not None:
            return cached_targets
        targets = self._list_targets_from_solution(build_directory)
        if targets:
            self._store_cached_targets(build_directory, signature, targets)
            return targets
        targets = self._list_targets_from_vcxproj_files(build_directory)
        if targets:
            self._store_cached_targets(build_directory, signature, targets)
            return targets
        self._store_cached_targets(build_directory, signature, [])
        return []

    def list_ctest_tests(self, build_directory: Path) -> list[str]:
        """Return configured CTest names from generated CTest files."""
        build_directory = build_directory.resolve()
        signature = self._ctest_signature(build_directory)
        cached_tests = self._load_cached_tests(build_directory, signature)
        if cached_tests is not None:
            return cached_tests
        tests = []
        seen_tests = set()
        for test_file in self._collect_test_files(build_directory):
            try:
                test_text = test_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw_line in test_text.splitlines():
                stripped_line = raw_line.strip()
                if not stripped_line.startswith("add_test("):
                    continue
                match = self.TEST_PATTERN.match(stripped_line)
                if match is None:
                    continue
                test_name = match.group(1).strip()
                if test_name == "NOT_AVAILABLE" or test_name in seen_tests:
                    continue
                seen_tests.add(test_name)
                tests.append(test_name)
        self._store_cached_tests(build_directory, signature, tests)
        return tests

    def _load_cached_targets(self, build_directory: Path, signature: tuple[tuple[str, int, int], ...]) -> list[str] | None:
        """Return cached targets when the underlying project files are unchanged."""
        cached_entry = self._build_target_cache.get(build_directory)
        if cached_entry is None:
            return None
        cached_signature, cached_targets = cached_entry
        if cached_signature != signature:
            return None
        return list(cached_targets)

    def _store_cached_targets(
        self,
        build_directory: Path,
        signature: tuple[tuple[str, int, int], ...],
        targets: list[str],
    ) -> None:
        """Store build targets with the current artifact signature."""
        self._build_target_cache[build_directory] = (signature, list(targets))

    def _load_cached_tests(self, build_directory: Path, signature: tuple[tuple[str, int, int], ...]) -> list[str] | None:
        """Return cached test names when generated CTest files are unchanged."""
        cached_entry = self._ctest_cache.get(build_directory)
        if cached_entry is None:
            return None
        cached_signature, cached_tests = cached_entry
        if cached_signature != signature:
            return None
        return list(cached_tests)

    def _store_cached_tests(
        self,
        build_directory: Path,
        signature: tuple[tuple[str, int, int], ...],
        tests: list[str],
    ) -> None:
        """Store discovered test names with the current artifact signature."""
        self._ctest_cache[build_directory] = (signature, list(tests))

    def _build_target_signature(self, build_directory: Path) -> tuple[tuple[str, int, int], ...]:
        """Return a stable signature for build-target artifacts."""
        metadata_files = list(sorted(build_directory.glob("*.sln")))
        metadata_files.extend(self._collect_project_files(build_directory))
        return self._build_file_signature(metadata_files)

    def _ctest_signature(self, build_directory: Path) -> tuple[tuple[str, int, int], ...]:
        """Return a stable signature for generated CTest metadata files."""
        return self._build_file_signature(self._collect_test_files(build_directory))

    def _collect_project_files(self, build_directory: Path) -> list[Path]:
        """Collect generated project files used for target discovery."""
        project_files = []
        for project_file in sorted(build_directory.rglob("*.vcxproj")):
            if "CMakeFiles" in project_file.parts:
                continue
            project_files.append(project_file)
        return project_files

    def _collect_test_files(self, build_directory: Path) -> list[Path]:
        """Collect generated CTest metadata files."""
        return list(sorted(build_directory.rglob("CTestTestfile.cmake")))

    def _build_file_signature(self, file_paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
        """Return one stable signature for the given generated artifact files."""
        signature = []
        for file_path in file_paths:
            try:
                stat_result = file_path.stat()
            except OSError:
                continue
            signature.append((str(file_path), int(stat_result.st_mtime_ns), int(stat_result.st_size)))
        return tuple(signature)

    def _list_targets_from_solution(self, build_directory: Path) -> list[str]:
        """Parse target names from a Visual Studio solution file."""
        solution_files = sorted(build_directory.glob("*.sln"))
        if not solution_files:
            return []
        solution_text = solution_files[0].read_text(encoding="utf-8", errors="ignore")
        targets = []
        seen_targets = set()
        for raw_line in solution_text.splitlines():
            match = re.match(r'Project\("\{[^"]+\}"\)\s*=\s*"([^"]+)",\s*"([^"]+)"', raw_line)
            if match is None:
                continue
            target_name = match.group(1).strip()
            project_path = match.group(2).strip()
            lowered_project_path = project_path.lower()
            if not lowered_project_path.endswith((".vcxproj", ".proj", ".csproj")):
                continue
            if target_name in self.TARGET_EXCLUSIONS or target_name in seen_targets:
                continue
            seen_targets.add(target_name)
            targets.append(target_name)
        return targets

    def _list_targets_from_vcxproj_files(self, build_directory: Path) -> list[str]:
        """Fallback to project file names when no solution file is present."""
        targets = []
        seen_targets = set()
        for project_file in sorted(build_directory.rglob("*.vcxproj")):
            if "CMakeFiles" in project_file.parts:
                continue
            target_name = project_file.stem
            if target_name in self.TARGET_EXCLUSIONS or target_name in seen_targets:
                continue
            seen_targets.add(target_name)
            targets.append(target_name)
        return targets
