"""Command safety policy for controlled local execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CommandSafetyLevel:
    """Safety levels for local command execution."""

    READ_ONLY = "read_only"
    BUILD_ARTIFACT_WRITE = "build_artifact_write"
    SOURCE_MODIFY = "source_modify"


@dataclass(frozen=True)
class CommandSafetyDecision:
    """Result of validating one command against the safety policy."""

    allowed: bool
    safety_level: str
    summary_text: str


class CommandSafetyPolicy:
    """Allow only read-only inspection or build-artifact writes by default."""

    READ_ONLY_COMMANDS = {"ls", "dir", "find", "grep", "rg", "cat", "sed", "head", "tail"}
    SOURCE_MODIFY_COMMANDS = {"git", "rm", "rmdir", "del", "erase", "mv", "move", "cp", "copy", "patch"}
    SAFE_ARTIFACT_ROOTS = {"build", "out", "dist", "tmp", ".tmp", ".assignment_agent_work"}

    def __init__(self, repository_path: Path, allowed_artifact_directories: list[Path] | None = None) -> None:
        self.repository_path = repository_path.resolve()
        self.allowed_artifact_directories = [directory.resolve() for directory in (allowed_artifact_directories or [])]

    def validate(self, command_parts: list[str]) -> CommandSafetyDecision:
        """Validate one command against the repository execution policy."""
        if not command_parts:
            return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "Empty commands are not allowed.")

        command_name = self._normalize_command_name(command_parts[0])
        if command_name in self.SOURCE_MODIFY_COMMANDS:
            return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, f"`{command_name}` is a source-modifying command and is blocked by policy.")
        if command_name in self.READ_ONLY_COMMANDS:
            return CommandSafetyDecision(True, CommandSafetyLevel.READ_ONLY, f"`{command_name}` is classified as read-only.")
        if command_name == "cmake":
            return self._validate_cmake(command_parts)
        if command_name == "ctest":
            return self._validate_ctest(command_parts)
        if command_name in {"make", "ninja"}:
            return self._validate_backend_build(command_parts, command_name)
        return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, f"`{command_name}` is not on the allowed command whitelist.")

    def _validate_cmake(self, command_parts: list[str]) -> CommandSafetyDecision:
        """Allow only explicitly whitelisted CMake command forms."""
        if "-S" in command_parts and "-B" in command_parts:
            source_directory = self._resolve_option_path(command_parts, "-S")
            build_directory = self._resolve_option_path(command_parts, "-B")
            if source_directory is None or not self._is_within(self.repository_path, source_directory):
                return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "CMake configure must read sources from the target repository.")
            if build_directory is None or not self._is_allowed_artifact_path(build_directory):
                return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "CMake configure may write only to the build artifact directory.")
            return CommandSafetyDecision(True, CommandSafetyLevel.BUILD_ARTIFACT_WRITE, "CMake configure writes only to generated build artifacts.")

        if "--build" in command_parts:
            build_directory = self._resolve_option_path(command_parts, "--build")
            if build_directory is None or not self._is_allowed_artifact_path(build_directory):
                return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "CMake build may write only to the build artifact directory.")
            if "--target" in command_parts and self._get_option_value(command_parts, "--target") == "help":
                return CommandSafetyDecision(True, CommandSafetyLevel.READ_ONLY, "CMake target help is read-only.")
            return CommandSafetyDecision(True, CommandSafetyLevel.BUILD_ARTIFACT_WRITE, "CMake build is allowed because writes stay under build artifacts.")

        if "-LAH" in command_parts and "-N" in command_parts:
            build_directory = self._resolve_last_path_argument(command_parts)
            if build_directory is None or not self._is_allowed_artifact_path(build_directory):
                return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "CMake cache inspection must point at the build artifact directory.")
            return CommandSafetyDecision(True, CommandSafetyLevel.READ_ONLY, "CMake cache inspection is read-only.")

        return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "Only whitelisted CMake configure, build, and cache-inspection forms are allowed.")

    def _validate_ctest(self, command_parts: list[str]) -> CommandSafetyDecision:
        """Allow CTest only when scoped to the build artifact directory."""
        test_directory = self._resolve_option_path(command_parts, "--test-dir")
        if test_directory is None or not self._is_allowed_artifact_path(test_directory):
            return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "CTest must run against the generated build directory.")
        if "-N" in command_parts:
            return CommandSafetyDecision(True, CommandSafetyLevel.READ_ONLY, "CTest listing is read-only.")
        return CommandSafetyDecision(True, CommandSafetyLevel.BUILD_ARTIFACT_WRITE, "CTest is allowed as a contained build/test command.")

    def _validate_backend_build(self, command_parts: list[str], command_name: str) -> CommandSafetyDecision:
        """Allow make or ninja only when explicitly confined to the build directory."""
        if command_name == "make" and "-f" in command_parts:
            makefile_path = self._resolve_option_path(command_parts, "-f")
            if makefile_path is None or not self._is_within(self.repository_path, makefile_path):
                return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, "Named makefiles must live inside the repository.")
            return CommandSafetyDecision(True, CommandSafetyLevel.BUILD_ARTIFACT_WRITE, "Explicit repo makefiles are allowed as controlled build/test commands.")
        build_directory = self._resolve_option_path(command_parts, "-C")
        if build_directory is None or not self._is_allowed_artifact_path(build_directory):
            return CommandSafetyDecision(False, CommandSafetyLevel.SOURCE_MODIFY, f"`{command_name}` must be scoped to the build artifact directory with `-C`.")
        return CommandSafetyDecision(True, CommandSafetyLevel.BUILD_ARTIFACT_WRITE, f"`{command_name}` is allowed because writes stay under build artifacts.")

    def _resolve_option_path(self, command_parts: list[str], option_name: str) -> Path | None:
        """Resolve the path value associated with one option."""
        value = self._get_option_value(command_parts, option_name)
        if value is None:
            return None
        return self._resolve_path(value)

    def _resolve_last_path_argument(self, command_parts: list[str]) -> Path | None:
        """Resolve the last path-like argument in a command."""
        for value in reversed(command_parts[1:]):
            if value.startswith("-"):
                continue
            return self._resolve_path(value)
        return None

    def _get_option_value(self, command_parts: list[str], option_name: str) -> str | None:
        """Return the argument value that follows one option."""
        try:
            option_index = command_parts.index(option_name)
        except ValueError:
            return None
        if option_index + 1 >= len(command_parts):
            return None
        return command_parts[option_index + 1]

    def _resolve_path(self, value: str) -> Path:
        """Resolve a command path relative to the repository when needed."""
        candidate_path = Path(value)
        if candidate_path.is_absolute():
            return candidate_path.resolve()
        return (self.repository_path / candidate_path).resolve()

    def _normalize_command_name(self, value: str) -> str:
        """Normalize an executable name for policy matching."""
        command_name = Path(value).name.lower()
        if command_name.endswith(".exe"):
            return command_name[:-4]
        return command_name

    def _is_allowed_artifact_path(self, candidate_path: Path) -> bool:
        """Return True when a path is inside the allowed artifact area."""
        if any(self._is_within(directory, candidate_path) for directory in self.allowed_artifact_directories):
            return True
        if not self._is_within(self.repository_path, candidate_path):
            return False
        try:
            relative_path = candidate_path.relative_to(self.repository_path)
        except ValueError:
            return False
        if not relative_path.parts:
            return False
        return relative_path.parts[0].lower() in self.SAFE_ARTIFACT_ROOTS

    def _is_within(self, root_path: Path, candidate_path: Path) -> bool:
        """Return True when a path is the same as or nested inside another path."""
        try:
            candidate_path.relative_to(root_path)
            return True
        except ValueError:
            return False
