"""Build failure parsing and retrieval-query extraction.

This module turns raw build output into a focused retrieval query. It improves
accuracy when the user prompt is generic and the compiler output introduces the real
identifiers that matter.
"""

from __future__ import annotations

import re


class BuildFailureAnalysis:
    """Stores the actionable parts of one build failure."""

    def __init__(
        self,
        error_line: str,
        identifiers: list[str],
        file_path: str = "",
        line_number: int | None = None,
        failure_kind: str = "unknown",
        should_retry_with_happy_path: bool = False,
        stop_reason: str = "",
    ) -> None:
        self.error_line = error_line
        self.identifiers = identifiers
        self.file_path = file_path
        self.line_number = line_number
        self.failure_kind = failure_kind
        self.should_retry_with_happy_path = should_retry_with_happy_path
        self.stop_reason = stop_reason

    def build_retrieval_query(self, original_query: str) -> str:
        """Create a retrieval query that blends the user request with build evidence."""
        query_parts = [original_query]
        if self.file_path:
            query_parts.append(self.file_path)
        if self.error_line:
            query_parts.append(self.error_line)
        if self.identifiers:
            query_parts.append(" ".join(self.identifiers))
        return "\n".join(query_parts)


class BuildFailureAnalyzer:
    """Extracts the most useful build-failure evidence from command output."""

    IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_:<>]*")
    ERROR_LINE_PATTERN = re.compile(r"error[: ].*", re.IGNORECASE)
    MISSING_COMMAND_PATTERN = re.compile(r"(winerror 2|no such file or directory|not found|is not recognized)", re.IGNORECASE)
    BUILD_SETUP_PATTERN = re.compile(r"(could not load cache|not a build directory|cannot find test manifest|no tests were found)", re.IGNORECASE)
    ENVIRONMENT_PATTERN = re.compile(
        r"(generator|toolchain|compiler|sdk|permission denied|access is denied|cmake error|could not find|not recognized)",
        re.IGNORECASE,
    )

    def analyze_output(
        self,
        output_text: str,
        file_path: str = "",
        line_number: int | None = None,
        command_text: str = "",
    ) -> BuildFailureAnalysis:
        """Parse command output into a retrieval-focused failure analysis."""
        error_line = self._extract_actionable_error_line(output_text)
        identifiers = self._extract_identifiers(error_line or output_text)
        failure_kind = self._classify_failure_kind(output_text, file_path)
        should_retry = self._should_retry_with_happy_path(command_text, output_text)
        stop_reason = self._build_stop_reason(failure_kind, should_retry, file_path)
        return BuildFailureAnalysis(
            error_line,
            identifiers,
            file_path,
            line_number,
            failure_kind,
            should_retry,
            stop_reason,
        )

    def _extract_actionable_error_line(self, output_text: str) -> str:
        """Return the first useful error line from the build output."""
        for raw_line in output_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.ERROR_LINE_PATTERN.search(line):
                return line
            if "error" in line.lower() or "undefined reference" in line.lower() or "failed" in line.lower():
                return line
        return ""

    def _extract_identifiers(self, text: str) -> list[str]:
        """Extract likely identifiers introduced by the build failure."""
        identifiers = []
        seen = set()
        for raw_identifier in self.IDENTIFIER_PATTERN.findall(text):
            lowered_identifier = raw_identifier.lower()
            if lowered_identifier in seen:
                continue
            seen.add(lowered_identifier)
            identifiers.append(raw_identifier)
            if len(identifiers) >= 12:
                break
        return identifiers

    def _classify_failure_kind(self, output_text: str, file_path: str) -> str:
        """Classify the failure as code-related, environmental, or unknown."""
        if file_path:
            return "code"
        if self.MISSING_COMMAND_PATTERN.search(output_text):
            return "environment"
        if self.ENVIRONMENT_PATTERN.search(output_text):
            return "environment"
        if self.ERROR_LINE_PATTERN.search(output_text):
            return "unknown"
        return "unknown"

    def _should_retry_with_happy_path(self, command_text: str, output_text: str) -> bool:
        """Return True when a single retry with configure/build/test is justified."""
        lowered_command = command_text.lower()
        if lowered_command.startswith("ctest") and self.BUILD_SETUP_PATTERN.search(output_text):
            return True
        if lowered_command.startswith("cmake --build") and self.BUILD_SETUP_PATTERN.search(output_text):
            return True
        return False

    def _build_stop_reason(self, failure_kind: str, should_retry: bool, file_path: str) -> str:
        """Explain why the agent stopped instead of continuing execution blindly."""
        if should_retry:
            return "The initial command skipped required setup, so one retry with the default configure/build/test flow is justified."
        if failure_kind == "environment":
            return "Execution stopped because the failure is environmental, so rerunning the same commands would likely repeat the same toolchain problem."
        if failure_kind == "code" and file_path:
            return f"Execution stopped at the first source error in {file_path} because later commands would not add better evidence until that code issue is fixed."
        return "Execution stopped after the first failure because the next actionable source location was unclear."
