"""Build runner for local repository builds."""

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


class BuildRunner:
    """Run configure and build commands against the local repository.

    Policy:
    - Prefer CMake when a `CMakeLists.txt` file exists.
    - Use `cmake --build` as the default build command.
    - Prefer a parallel Release build on Windows multi-config generators.
    - Store build artifacts outside the target repository.
    - Allow backend-specific `make` or `ninja` only when they target the isolated build directory.
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

    def run(self, query_text: str) -> ExecutionBatch:
        """Run the default build flow selected for the repository."""
        return self.run_with_progress(query_text)

    def run_with_progress(self, query_text: str, progress_callback: Callable[[str], None] | None = None) -> ExecutionBatch:
        """Run the build flow while reporting command progress."""
        commands = self._build_commands(query_text)
        self.logger.debug("build query=%r commands=%s", query_text, commands)
        results = []
        for command_parts in commands:
            result = self.executor.run(command_parts, "build", progress_callback=progress_callback)
            results.append(result)
            if result.exit_code != 0:
                break
        return ExecutionBatch("build", results)

    def _build_commands(self, query_text: str = "") -> list[list[str]]:
        """Choose build commands from the repository layout and explicit query hints."""
        execution_intent = ExecutionIntent.from_query(query_text)
        explicit_makefile = execution_intent.extract_makefile_name()
        if explicit_makefile:
            return [["make", "-f", str(self._resolve_repo_file(explicit_makefile))]]
        if (self.repository_path / "CMakeLists.txt").exists():
            build_directory = self.workspace_paths.get_build_directory(self._build_variant_name(execution_intent))
            configure_command = self._build_configure_command(build_directory, execution_intent)
            if execution_intent.requests_show_cmake_options():
                return [configure_command, ["cmake", "-LAH", "-N", str(build_directory)]]
            # These read-only questions are answered from generated build artifacts later.
            if execution_intent.requests_build_targets():
                return [configure_command]
            if execution_intent.requests_list_tests():
                return [configure_command]
            if execution_intent.requests_configure_only():
                return [configure_command]
            explicit_target = execution_intent.extract_build_target_name()
            if explicit_target:
                return [configure_command, self._build_target_command(build_directory, execution_intent, explicit_target)]
            build_command = self._build_backend_command(build_directory, execution_intent)
            return [configure_command, build_command]

        if self.error_accumulator is not None:
            self.error_accumulator.add(
                "build_runner",
                "No supported build entrypoint was detected",
                f"repo={self.repository_path}",
                "warning",
            )
        return [["cmake", "--build", str(self.workspace_paths.get_build_directory())]]

    def _build_configure_command(self, build_directory: Path, execution_intent: ExecutionIntent) -> list[str]:
        """Choose the CMake configure command, including optional generator hints."""
        command = [
            "cmake",
            "-S",
            str(self.repository_path),
            "-B",
            str(build_directory),
            "-DJSON_BuildTests=ON",
            "-DBUILD_TESTING=ON",
        ]
        if execution_intent.wants_valgrind():
            command.append("-DJSON_Valgrind=ON")
        if execution_intent.wants_fast_tests():
            command.append("-DJSON_FastTests=ON")
        if execution_intent.wants_make_backend():
            make_path = CommandExecutor.resolve_command_path("make")
            command.extend(["-G", "Unix Makefiles"])
            command.append("-DCMAKE_BUILD_TYPE=Release")
            if make_path:
                command.append(f"-DCMAKE_MAKE_PROGRAM={make_path}")
        elif execution_intent.wants_ninja_backend():
            ninja_path = CommandExecutor.resolve_command_path("ninja")
            command.extend(["-G", "Ninja"])
            command.append("-DCMAKE_BUILD_TYPE=Release")
            if ninja_path:
                command.append(f"-DCMAKE_MAKE_PROGRAM={ninja_path}")
        elif os.name != "nt":
            command.append("-DCMAKE_BUILD_TYPE=Release")
        return command

    def _build_backend_command(self, build_directory: Path, execution_intent: ExecutionIntent) -> list[str]:
        """Choose the actual backend build command for the current query."""
        if execution_intent.wants_make_backend():
            return ["make", "-C", str(build_directory)]
        if execution_intent.wants_ninja_backend():
            return ["ninja", "-C", str(build_directory)]
        command = ["cmake", "--build", str(build_directory)]
        if os.name == "nt" and not self._uses_single_config_backend(execution_intent):
            command.extend(["--config", "Release"])
        command.append("--parallel")
        return command

    def _build_target_command(self, build_directory: Path, execution_intent: ExecutionIntent, target_name: str) -> list[str]:
        """Choose the actual backend command for one explicit build target."""
        if execution_intent.wants_make_backend():
            return ["make", "-C", str(build_directory), target_name]
        if execution_intent.wants_ninja_backend():
            return ["ninja", "-C", str(build_directory), target_name]
        command = ["cmake", "--build", str(build_directory), "--target", target_name]
        if os.name == "nt" and not self._uses_single_config_backend(execution_intent):
            command.extend(["--config", "Release"])
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

    def _resolve_repo_file(self, file_name: str) -> Path:
        """Resolve a user-named file relative to the repository root when needed."""
        candidate_path = Path(file_name)
        if candidate_path.is_absolute():
            return candidate_path
        return self.repository_path / candidate_path
