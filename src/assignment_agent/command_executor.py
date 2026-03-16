"""Shared command execution with consistent logging and error capture."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import logging
import os
import queue
import shutil
import subprocess
import threading
import time

from .command_safety import CommandSafetyPolicy
from .contracts import CommandExecutionResult
from .error_accumulator import ErrorAccumulator


class CommandExecutor:
    """Run local commands and normalize runtime failures."""

    DEFAULT_TIMEOUT_SECONDS = 300
    BUILD_TIMEOUT_SECONDS = 600
    READ_ONLY_TIMEOUT_SECONDS = 120

    # Map the user-facing tool name to the executable names we are willing to try.
    COMMAND_ALIASES = {
        "cmake": ("cmake",),
        "ctest": ("ctest",),
        "make": ("make", "mingw32-make", "gmake"),
        "ninja": ("ninja",),
    }

    # Windows installs often miss PATH setup, so keep a small set of discovery hints.
    WINDOWS_DISCOVERY_HINTS = {
        "cmake": ("cmake",),
        "ctest": ("cmake",),
        "make": ("make", "mingw", "msys", "android", "ndk", "gnu"),
        "ninja": ("ninja", "cmake", "visual studio"),
    }
    WINDOWS_FALLBACK_DIRECTORIES = (
        Path(r"C:\ProgramData\chocolatey\bin"),
        Path(r"C:\Program Files\CMake\bin"),
        Path(r"C:\Program Files (x86)\CMake\bin"),
    )
    def __init__(
        self,
        repository_path: Path,
        error_accumulator: ErrorAccumulator | None = None,
        allowed_artifact_directories: list[Path] | None = None,
    ) -> None:
        self.repository_path = repository_path
        self.error_accumulator = error_accumulator
        self.logger = logging.getLogger(__name__)
        self.safety_policy = CommandSafetyPolicy(repository_path, allowed_artifact_directories)

    def run(
        self,
        command_parts: list[str],
        phase_name: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CommandExecutionResult:
        """Execute one command against the target repository."""
        self.logger.debug("phase=%s command=%s", phase_name, " ".join(command_parts))
        safety_decision = self.safety_policy.validate(command_parts)
        if not safety_decision.allowed:
            return self._build_blocked_result(command_parts, phase_name, safety_decision.summary_text, progress_callback)

        resolved_command_parts = self._resolve_command_parts(command_parts)
        try:
            timeout_seconds = self._select_timeout_seconds(command_parts, phase_name)
            self._report_progress(progress_callback, f"Running {phase_name} command: {' '.join(command_parts)}")
            exit_code, stdout_text, stderr_text = self._run_streaming_command(
                resolved_command_parts,
                timeout_seconds,
                progress_callback,
            )
            self._record_nonzero_exit(command_parts, phase_name, exit_code)
            self._report_progress(
                progress_callback,
                f"Finished {phase_name} command with exit code {exit_code}: {' '.join(command_parts)}",
            )
            return CommandExecutionResult(command_parts, exit_code, stdout_text, stderr_text, phase_name)
        except subprocess.TimeoutExpired as error:
            return self._build_timeout_result(command_parts, phase_name, error, progress_callback)
        except FileNotFoundError as error:
            return self._build_missing_command_result(command_parts, phase_name, error, progress_callback)
        except OSError as error:
            return self._build_os_error_result(command_parts, phase_name, error, progress_callback)

    def _select_timeout_seconds(self, command_parts: list[str], phase_name: str) -> int:
        """Choose a timeout that fits the command shape."""
        if self._is_read_only_command(command_parts):
            return self.READ_ONLY_TIMEOUT_SECONDS
        if phase_name in {"build", "test"}:
            return self.BUILD_TIMEOUT_SECONDS
        return self.DEFAULT_TIMEOUT_SECONDS

    def _is_read_only_command(self, command_parts: list[str]) -> bool:
        """Return True for inspection-style commands that should finish quickly."""
        if not command_parts:
            return False
        command_name = Path(command_parts[0]).name.lower().removesuffix(".exe")
        if command_name == "ctest" and "-N" in command_parts:
            return True
        if command_name != "cmake":
            return False
        if "-LAH" in command_parts and "-N" in command_parts:
            return True
        return "--target" in command_parts and "help" in command_parts

    def _resolve_command_parts(self, command_parts: list[str]) -> list[str]:
        """Replace the executable with a resolved absolute path when one is found."""
        resolved_command_parts = list(command_parts)
        resolved_executable = self.resolve_command_path(command_parts[0])
        if resolved_executable is not None:
            resolved_command_parts[0] = resolved_executable
        return resolved_command_parts

    def _build_blocked_result(
        self,
        command_parts: list[str],
        phase_name: str,
        summary_text: str,
        progress_callback: Callable[[str], None] | None,
    ) -> CommandExecutionResult:
        """Create a consistent result for commands rejected by policy."""
        if self.error_accumulator is not None:
            self.error_accumulator.add(
                "command_executor",
                "Command blocked by safety policy",
                f"command={' '.join(command_parts)} reason={summary_text}",
                "warning",
            )
        self._report_progress(progress_callback, f"Blocked {phase_name} command: {' '.join(command_parts)}")
        return CommandExecutionResult(command_parts, 126, "", summary_text, phase_name)

    def _record_nonzero_exit(self, command_parts: list[str], phase_name: str, exit_code: int) -> None:
        """Persist one warning when a launched command returns a failure code."""
        if exit_code == 0 or self.error_accumulator is None:
            return
        self.error_accumulator.add(
            "command_executor",
            f"{phase_name} command failed",
            f"command={' '.join(command_parts)} exit_code={exit_code}",
            "warning",
        )

    def _build_timeout_result(
        self,
        command_parts: list[str],
        phase_name: str,
        error: subprocess.TimeoutExpired,
        progress_callback: Callable[[str], None] | None,
    ) -> CommandExecutionResult:
        """Create a consistent timeout result that preserves partial output."""
        if self.error_accumulator is not None:
            self.error_accumulator.add_exception(
                "command_executor",
                f"{phase_name} command timed out",
                error,
            )
        stdout_text = error.stdout or ""
        stderr_text = error.stderr or ""
        self._report_progress(progress_callback, f"Timed out {phase_name} command: {' '.join(command_parts)}")
        return CommandExecutionResult(command_parts, 124, stdout_text, stderr_text or str(error), phase_name)

    def _build_missing_command_result(
        self,
        command_parts: list[str],
        phase_name: str,
        error: FileNotFoundError,
        progress_callback: Callable[[str], None] | None,
    ) -> CommandExecutionResult:
        """Create a consistent result when the executable cannot be found."""
        if self.error_accumulator is not None:
            self.error_accumulator.add_exception(
                "command_executor",
                f"{phase_name} command is unavailable",
                error,
            )
        self._report_progress(progress_callback, f"Missing executable for {phase_name} command: {' '.join(command_parts)}")
        return CommandExecutionResult(command_parts, 127, "", str(error), phase_name)

    def _build_os_error_result(
        self,
        command_parts: list[str],
        phase_name: str,
        error: OSError,
        progress_callback: Callable[[str], None] | None,
    ) -> CommandExecutionResult:
        """Create a consistent result when the process cannot be started."""
        if self.error_accumulator is not None:
            self.error_accumulator.add_exception(
                "command_executor",
                f"{phase_name} command could not start",
                error,
            )
        self._report_progress(progress_callback, f"Failed to start {phase_name} command: {' '.join(command_parts)}")
        return CommandExecutionResult(command_parts, 1, "", str(error), phase_name)

    @staticmethod
    def _report_progress(progress_callback: Callable[[str], None] | None, message_text: str) -> None:
        """Send one progress update when the caller provided a callback."""
        if progress_callback is not None:
            progress_callback(message_text)

    @classmethod
    def resolve_command_path(cls, command_name: str) -> str | None:
        """Resolve an executable from PATH or common Windows install locations."""
        if not command_name:
            return None
        normalized_name = command_name.lower()
        command_path = Path(command_name)
        if command_path.is_absolute() or command_path.parent != Path("."):
            if command_path.exists():
                return str(command_path)
            return None

        candidate_names = cls.COMMAND_ALIASES.get(normalized_name, (command_name,))
        for candidate_name in candidate_names:
            resolved_path = shutil.which(candidate_name)
            if resolved_path is not None:
                return resolved_path
        if os.name != "nt":
            return None

        executable_candidates = []
        for candidate_name in candidate_names:
            executable_candidates.append(candidate_name)
            if not candidate_name.lower().endswith(".exe"):
                executable_candidates.append(f"{candidate_name}.exe")

        for directory in cls._windows_fallback_directories(normalized_name):
            for candidate_name in executable_candidates:
                candidate_path = directory / candidate_name
                if candidate_path.exists():
                    return str(candidate_path)
        discovered_executable = cls._discover_windows_executable(normalized_name, executable_candidates)
        if discovered_executable is not None:
            return discovered_executable
        return None

    @classmethod
    def _windows_fallback_directories(cls, command_name: str) -> tuple[Path, ...]:
        """Return common Windows install directories for build executables."""
        directories = list(cls.WINDOWS_FALLBACK_DIRECTORIES)
        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_name)
            if not root:
                continue
            directories.append(Path(root) / "CMake" / "bin")
        directories.extend(cls._discover_windows_tool_directories(command_name))
        unique_directories = []
        seen_paths = set()
        for directory in directories:
            if directory in seen_paths:
                continue
            seen_paths.add(directory)
            unique_directories.append(directory)
        return tuple(unique_directories)

    @classmethod
    def _discover_windows_tool_directories(cls, command_name: str) -> list[Path]:
        """Probe likely Windows install roots for build tools that are missing from PATH."""
        hints = cls.WINDOWS_DISCOVERY_HINTS.get(command_name, ())
        if not hints:
            return []

        directories = []
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData", "LocalAppData"):
            root_value = os.environ.get(env_name)
            if not root_value:
                continue
            root_path = Path(root_value)
            directories.extend(cls._find_candidate_directories(root_path, hints))
        return directories

    @classmethod
    def _discover_windows_executable(cls, command_name: str, candidate_names: list[str]) -> str | None:
        """Search likely install roots for one executable when directory probing is not enough."""
        hints = cls.WINDOWS_DISCOVERY_HINTS.get(command_name, ())
        if not hints:
            return None

        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData", "LocalAppData"):
            root_value = os.environ.get(env_name)
            if not root_value:
                continue
            root_path = Path(root_value)
            for child_path in cls._iter_hint_directories(root_path, hints):
                for candidate_name in candidate_names:
                    try:
                        matches = child_path.rglob(candidate_name)
                    except OSError:
                        continue
                    for match_path in matches:
                        if match_path.is_file():
                            return str(match_path)
        return None

    @classmethod
    def _find_candidate_directories(cls, root_path: Path, hints: tuple[str, ...]) -> list[Path]:
        """Search a small set of installation folders for directories that likely contain the tool."""
        if not root_path.exists():
            return []

        candidate_directories = []
        for child_path in cls._iter_hint_directories(root_path, hints):
            candidate_directories.append(child_path)
            try:
                nested_directories = child_path.rglob("bin")
            except OSError:
                continue
            for nested_directory in nested_directories:
                if nested_directory.is_dir():
                    candidate_directories.append(nested_directory)
        return candidate_directories

    @staticmethod
    def _iter_hint_directories(root_path: Path, hints: tuple[str, ...]):
        """Yield top-level install directories whose names match the discovery hints."""
        try:
            children = list(root_path.iterdir())
        except OSError:
            return

        for child_path in children:
            try:
                is_directory = child_path.is_dir()
            except OSError:
                continue
            if not is_directory:
                continue
            child_name = child_path.name.lower()
            if any(hint in child_name for hint in hints):
                yield child_path

    def _run_streaming_command(
        self,
        resolved_command_parts: list[str],
        timeout_seconds: int,
        progress_callback: Callable[[str], None] | None,
    ) -> tuple[int, str, str]:
        """Run one command while streaming merged output back to the caller."""
        process = subprocess.Popen(
            resolved_command_parts,
            cwd=str(self.repository_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            bufsize=1,
        )
        line_queue: queue.Queue[str] = queue.Queue()
        reader_thread = threading.Thread(
            target=self._read_stream_lines,
            args=(process.stdout, line_queue),
            daemon=True,
        )
        reader_thread.start()

        output_lines: list[str] = []
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                process.kill()
                process.wait()
                self._drain_stream_queue(line_queue, output_lines, progress_callback)
                raise subprocess.TimeoutExpired(
                    resolved_command_parts,
                    timeout_seconds,
                    output="".join(output_lines),
                    stderr="",
                )
            try:
                line_text = line_queue.get(timeout=min(0.1, remaining_seconds))
                output_lines.append(line_text)
                if progress_callback is not None and line_text.strip():
                    progress_callback(line_text.rstrip())
            except queue.Empty:
                if process.poll() is None:
                    continue
                if not reader_thread.is_alive() and line_queue.empty():
                    break

        reader_thread.join(timeout=1.0)
        self._drain_stream_queue(line_queue, output_lines, progress_callback)
        exit_code = process.wait(timeout=1.0)
        return exit_code, "".join(output_lines), ""

    @staticmethod
    def _read_stream_lines(stream, line_queue: queue.Queue[str]) -> None:
        """Read lines from a subprocess stream into a queue."""
        if stream is None:
            return
        try:
            for line_text in iter(stream.readline, ""):
                line_queue.put(line_text)
        finally:
            stream.close()

    @staticmethod
    def _drain_stream_queue(
        line_queue: queue.Queue[str],
        output_lines: list[str],
        progress_callback: Callable[[str], None] | None,
    ) -> None:
        """Flush any queued output lines after the process exits or times out."""
        while True:
            try:
                line_text = line_queue.get_nowait()
            except queue.Empty:
                return
            output_lines.append(line_text)
            if progress_callback is not None and line_text.strip():
                progress_callback(line_text.rstrip())
