"""Model and record classes used across the agent.

The project intentionally uses classic Python classes instead of dataclasses so each
object can carry behavior and remain easy to extend during the assignment.
"""

from __future__ import annotations


class PromptSection:
    """Represents a prompt fragment that competes for context budget.

    Each section has a label, body text, and priority. The budget manager keeps the
    highest-priority sections that fit inside the current request budget.
    """

    def __init__(self, label: str, text: str, priority: int) -> None:
        self.label = label
        self.text = text
        self.priority = priority


class CodeChunkRecord:
    """Stores a retrieved slice of a source file.

    The record contains path and line metadata so the agent can cite exact locations
    when answering questions or explaining compiler failures.
    """

    def __init__(self, path: str, start_line: int, end_line: int, summary: str, content: str, symbols: list[str]) -> None:
        self.path = path
        self.start_line = start_line
        self.end_line = end_line
        self.summary = summary
        self.content = content
        self.symbols = symbols

    def get_line_span_size(self) -> int:
        """Return the size of this chunk in source lines."""
        return self.end_line - self.start_line + 1


class IndexedFileRecord:
    """Stores the derived representation of one file in the repository."""

    def __init__(self, path: str, summary: str, symbols: list[str], chunks: list[CodeChunkRecord]) -> None:
        self.path = path
        self.summary = summary
        self.symbols = symbols
        self.chunks = chunks

    def get_smallest_available_chunk(self) -> CodeChunkRecord | None:
        """Return the smallest indexed chunk for this file."""
        if not self.chunks:
            return None
        smallest_chunk = self.chunks[0]
        for chunk in self.chunks[1:]:
            if chunk.get_line_span_size() < smallest_chunk.get_line_span_size():
                smallest_chunk = chunk
        return smallest_chunk


class RetrievedChunkMatch:
    """Stores one retrieval result and why it was selected."""

    def __init__(self, score: int, reason: str, chunk: CodeChunkRecord) -> None:
        self.score = score
        self.reason = reason
        self.chunk = chunk


class RetrievedFileMatch:
    """Stores one file-level retrieval result before chunk expansion."""

    def __init__(self, score: int, reason: str, indexed_file: IndexedFileRecord) -> None:
        self.score = score
        self.reason = reason
        self.indexed_file = indexed_file


class CommandExecutionResult:
    """Stores the outcome of an executed shell command."""

    def __init__(self, command_parts: list[str], return_code: int, stdout_text: str, stderr_text: str) -> None:
        self.command_parts = command_parts
        self.return_code = return_code
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text

    def get_command_text(self) -> str:
        """Return the command in printable form for reports and logs."""
        return " ".join(self.command_parts)

    def get_combined_output(self) -> str:
        """Return merged stdout and stderr while keeping empty sections out."""
        parts = []
        if self.stdout_text.strip():
            parts.append(self.stdout_text)
        if self.stderr_text.strip():
            parts.append(self.stderr_text)
        return "\n".join(parts)


class AgentAnswer:
    """Stores the final answer returned by the agent."""

    def __init__(self, task_type: str, answer_text: str, command_results: list[CommandExecutionResult] | None = None, prompt_report=None) -> None:
        self.task_type = task_type
        self.answer_text = answer_text
        self.command_results = command_results or []
        self.prompt_report = prompt_report
