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

    def list_build_targets(self, build_directory: Path) -> list[str]:
        """Return available build targets from generated build artifacts."""
        build_directory = build_directory.resolve()
        targets = self._list_targets_from_solution(build_directory)
        if targets:
            return targets
        targets = self._list_targets_from_vcxproj_files(build_directory)
        if targets:
            return targets
        return []

    def list_ctest_tests(self, build_directory: Path) -> list[str]:
        """Return configured CTest names from generated CTest files."""
        build_directory = build_directory.resolve()
        tests = []
        seen_tests = set()
        for test_file in sorted(build_directory.rglob("CTestTestfile.cmake")):
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
        return tests

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
