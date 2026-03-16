"""Command execution helpers for build and test workflows."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from .agent_models import CommandExecutionResult


class BuildCommandExecutor:
    """Runs build commands against the local repository and parses failures."""

    ERROR_PATTERNS = [
        re.compile(r"(?P<file>[A-Za-z0-9_./\\:-]+\.(?:h|hpp|hh|hxx|c|cc|cpp|cxx))[:(](?P<line>\d+)(?:[:),](?P<column>\d+))?"),
    ]

    def __init__(self, repository_path: Path) -> None:
        self.repository_path = repository_path

    def run_command(self, command_parts: list[str], working_directory: Path | None = None) -> CommandExecutionResult:
        """Execute one command and capture its output.

        The method catches missing binaries so the agent can explain environment
        problems instead of crashing.
        """
        try:
            completed = subprocess.run(
                command_parts,
                cwd=str(working_directory or self.repository_path),
                text=True,
                capture_output=True,
                shell=False,
            )
            return CommandExecutionResult(command_parts, completed.returncode, completed.stdout, completed.stderr)
        except FileNotFoundError as error:
            return CommandExecutionResult(command_parts, 127, "", str(error))

    def build_happy_path_commands(self) -> list[list[str]]:
        """Return the default CMake happy-path commands for a local repository."""
        build_directory = self.repository_path / ".agent-build"
        return [
            ["cmake", "-S", str(self.repository_path), "-B", str(build_directory)],
            ["cmake", "--build", str(build_directory), "--parallel"],
            ["ctest", "--test-dir", str(build_directory), "--output-on-failure"],
        ]

    def find_error_location(self, output_text: str) -> tuple[str, int] | None:
        """Extract the first file and line reference from compiler output."""
        for pattern in self.ERROR_PATTERNS:
            match = pattern.search(output_text)
            if match:
                return match.group("file"), int(match.group("line"))
        return None
