"""Test runner for local repository tests."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import logging
import os

from .command_executor import CommandExecutor
from .contracts import ExecutionBatch
from .error_accumulator import ErrorAccumulator
from .execution_intent import ExecutionIntent
from .workspace_paths import WorkspacePaths


class TestRunner:
    """Run repository tests after a successful build.

    Policy:
    - Prefer `ctest --test-dir <build>` for CMake projects.
    - Store test artifacts outside the target repository.
    - Prefer matching the default Release build configuration on Windows multi-config generators.
    - Allow backend-specific `make test` only when it targets the isolated build directory.
    """

    def __init__(
        self,
        repository_path: Path,
        workspace_paths: WorkspacePaths,
        error_accumulator: ErrorAccumulator | None = None,
    ) -> None:
        self.repository_path = repository_path
        self.workspace_paths = workspace_paths
        self.error_accumulator = error_accumulator
        self.executor = CommandExecutor(
            repository_path,
            error_accumulator,
            [self.workspace_paths.base_directory],
        )
        self.logger = logging.getLogger(__name__)

    def should_run_tests(self, query_text: str) -> bool:
        """Return True when the query requires test execution."""
        execution_intent = ExecutionIntent.from_query(query_text)
        return execution_intent.requests_test_execution()

    def run(self, query_text: str = "") -> ExecutionBatch:
        """Run the default test flow selected for the repository."""
        return self.run_with_progress(query_text)

    def run_with_progress(self, query_text: str = "", progress_callback: Callable[[str], None] | None = None) -> ExecutionBatch:
        """Run the test flow while reporting command progress."""
        commands = self._build_commands(query_text)
        self.logger.debug("test commands=%s", commands)
        results = []
        for command_parts in commands:
            result = self.executor.run(command_parts, "test", progress_callback=progress_callback)
            results.append(result)
            if result.exit_code != 0:
                break
        return ExecutionBatch("test", results)

    def _build_commands(self, query_text: str = "") -> list[list[str]]:
        """Choose test commands from the repository layout and explicit query hints."""
        execution_intent = ExecutionIntent.from_query(query_text)
        if (self.repository_path / "CMakeLists.txt").exists():
            build_directory = self.workspace_paths.get_build_directory(self._build_variant_name(execution_intent))
            if execution_intent.requests_list_tests():
                return [self._build_ctest_command(build_directory, execution_intent, list_only=True)]
            filtered_pattern = execution_intent.extract_test_filter()
            if filtered_pattern:
                return [self._build_ctest_command(build_directory, execution_intent, filtered_pattern=filtered_pattern)]
            if execution_intent.direct_tool_name() == "make" and self._has_make():
                return [["make", "-C", str(build_directory), "test"]]
            return [self._build_ctest_command(build_directory, execution_intent)]

        if self.error_accumulator is not None:
            self.error_accumulator.add(
                "test_runner",
                "No supported test entrypoint was detected",
                f"repo={self.repository_path}",
                "warning",
            )
        return [["ctest", "--test-dir", str(self.workspace_paths.get_build_directory()), "--output-on-failure"]]

    def _has_make(self) -> bool:
        """Return True when a raw make backend is available."""
        return CommandExecutor.resolve_command_path("make") is not None

    def _build_ctest_command(
        self,
        build_directory: Path,
        execution_intent: ExecutionIntent,
        list_only: bool = False,
        filtered_pattern: str = "",
    ) -> list[str]:
        """Build the default CTest command for the current platform."""
        command = ["ctest", "--test-dir", str(build_directory)]
        if os.name == "nt" and not self._uses_single_config_backend(execution_intent):
            command.extend(["-C", "Release"])
        if list_only:
            command.append("-N")
            return command
        if filtered_pattern:
            command.extend(["-R", filtered_pattern])
        command.append("--output-on-failure")
        return command

    def _uses_single_config_backend(self, execution_intent: ExecutionIntent) -> bool:
        """Return True when the query explicitly configures a single-config generator."""
        return execution_intent.wants_make_backend() or execution_intent.wants_ninja_backend()

    def _build_variant_name(self, execution_intent: ExecutionIntent) -> str:
        """Return the workspace build-directory variant for the requested backend."""
        if execution_intent.wants_make_backend():
            return "make"
        if execution_intent.wants_ninja_backend():
            return "ninja"
        return "default"
