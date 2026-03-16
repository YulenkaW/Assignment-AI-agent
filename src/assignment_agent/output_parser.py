"""Output parser for deterministic command evidence extraction."""

from __future__ import annotations

import logging
import re

from .contracts import CapturedCommandOutput, ParsedCommandOutput
from .error_accumulator import ErrorAccumulator


def _compile_patterns(*expressions: str) -> tuple[re.Pattern[str], ...]:
    """Compile a small tuple of case-insensitive regex patterns."""
    return tuple(re.compile(expression, re.IGNORECASE) for expression in expressions)


class OutputParser:
    """Parse raw command output into deterministic evidence."""

    # Match compiler-style file and line references such as file.cpp:42 or file.cpp(42).
    FILE_LINE_PATTERN = re.compile(r"(?P<file>[A-Za-z0-9_./\\:-]+\.(?:h|hpp|hh|hxx|c|cc|cpp|cxx))[:(](?P<line>\d+)")

    # These patterns capture hard failure signals that are stable across toolchains.
    EXPLICIT_ERROR_PATTERNS = _compile_patterns(
        r"\bCMake Error\b",
        r"\bfatal error\b",
        r"\berror:\b",
        r"\berror\s+[A-Za-z]+\d+:",
        r"\bundefined reference\b",
        r"\bFAILED:\b",
    )

    # Keep missing-command detection narrow so normal compiler failures do not get mislabeled.
    COMMAND_UNAVAILABLE_PATTERNS = _compile_patterns(
        r"\[winerror 2\]",
        r"the system cannot find the file specified",
        r"\bcommand not found\b",
        r"is not recognized as the name of a cmdlet",
        r"no such file or directory",
        r"required command is unavailable",
    )

    def __init__(self, error_accumulator: ErrorAccumulator | None = None) -> None:
        self.error_accumulator = error_accumulator
        self.logger = logging.getLogger(__name__)

    def parse_outputs(self, captured_outputs: list[CapturedCommandOutput]) -> list[ParsedCommandOutput]:
        """Parse all captured outputs."""
        parsed_outputs = []
        for captured_output in captured_outputs:
            parsed_outputs.append(self._parse_one(captured_output))
        return parsed_outputs

    def _parse_one(self, captured_output: CapturedCommandOutput) -> ParsedCommandOutput:
        """Parse a single command output."""
        error_lines = []
        file_references = []
        for raw_line in captured_output.combined_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered_line = line.lower()
            if self._is_error_line(line, lowered_line, captured_output.exit_code):
                error_lines.append(line)
            file_match = self.FILE_LINE_PATTERN.search(line)
            if file_match:
                file_references.append((file_match.group("file"), int(file_match.group("line"))))

        combined_output = captured_output.combined_output
        missing_command = self._matches_any(combined_output, self.COMMAND_UNAVAILABLE_PATTERNS)
        if missing_command and self.error_accumulator is not None:
            self.error_accumulator.add(
                "output_parser",
                "Required command appears to be unavailable",
                captured_output.command_text,
            )
        if captured_output.exit_code != 0 and not error_lines and self.error_accumulator is not None:
            self.error_accumulator.add(
                "output_parser",
                "Command failed without parsable error lines",
                captured_output.command_text,
                "warning",
            )
        self.logger.debug(
            "parsed command=%s exit_code=%s errors=%d refs=%d missing_command=%s",
            captured_output.command_text,
            captured_output.exit_code,
            len(error_lines),
            len(file_references),
            missing_command,
        )
        return ParsedCommandOutput(
            captured_output.command_text,
            captured_output.exit_code,
            error_lines,
            file_references,
            missing_command,
        )

    def _is_error_line(self, line: str, lowered_line: str, exit_code: int) -> bool:
        """Return True when the output line should be treated as an error signal."""
        if self._matches_any(line, self.EXPLICIT_ERROR_PATTERNS):
            return True
        if exit_code == 0:
            return False
        return "error" in lowered_line or "failed" in lowered_line or "undefined reference" in lowered_line

    @staticmethod
    def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
        """Return True when any compiled regex matches the given text."""
        return any(pattern.search(text) for pattern in patterns)
