"""Repository indexing and code chunk generation.

This module scans a local C++ repository, extracts lightweight symbol information,
and prepares chunk records that the retrieval layer can rank later.
"""

from __future__ import annotations

from pathlib import Path
import re

from .agent_models import CodeChunkRecord, IndexedFileRecord


class CodebaseIndexer:
    """Builds a lightweight searchable representation of the repository."""

    SOURCE_EXTENSIONS = {".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".ipp", ".inl", ".cmake", ".txt"}
    SYMBOL_PATTERN = re.compile(
        r"^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct|namespace|enum|using)\s+([A-Za-z_]\w*)|"
        r"^\s*(?:inline\s+|constexpr\s+|static\s+|virtual\s+)?(?:[\w:<>,~*&\s]+)\s+([A-Za-z_]\w*)\s*\(",
        re.MULTILINE,
    )

    def __init__(self, repository_path: Path) -> None:
        self.repository_path = repository_path
        self.indexed_files = {}
        self.symbol_to_paths = {}

    def build_index(self) -> None:
        """Scan the repository and build file records for supported source files."""
        for file_path in self.repository_path.rglob("*"):
            if not file_path.is_file():
                continue
            if not self._is_supported_file(file_path):
                continue
            relative_path = file_path.relative_to(self.repository_path).as_posix()
            file_text = file_path.read_text(encoding="utf-8", errors="ignore")
            indexed_file = self._create_indexed_file(relative_path, file_text)
            self.indexed_files[relative_path] = indexed_file
            self._register_symbols(indexed_file)

    def _is_supported_file(self, file_path: Path) -> bool:
        """Return True when the file should participate in retrieval."""
        if file_path.name == "CMakeLists.txt":
            return True
        return file_path.suffix.lower() in self.SOURCE_EXTENSIONS

    def _create_indexed_file(self, relative_path: str, file_text: str) -> IndexedFileRecord:
        """Create one indexed file record from raw file text."""
        file_lines = file_text.splitlines()
        symbols = self._extract_symbols(file_text)
        summary_text = self._build_summary(relative_path, file_lines, symbols)
        chunk_records = self._build_chunks(relative_path, file_lines, symbols)
        return IndexedFileRecord(relative_path, summary_text, symbols, chunk_records)

    def _extract_symbols(self, file_text: str) -> list[str]:
        """Extract likely symbol names using lightweight regular expressions."""
        symbols = []
        for match in self.SYMBOL_PATTERN.finditer(file_text):
            symbol_name = match.group(1) or match.group(2)
            if symbol_name and symbol_name not in symbols:
                symbols.append(symbol_name)
        return symbols[:50]

    def _build_summary(self, relative_path: str, file_lines: list[str], symbols: list[str]) -> str:
        """Create a short summary for file-level retrieval and memory."""
        non_empty_lines = []
        for line in file_lines:
            stripped_line = line.strip()
            if stripped_line:
                non_empty_lines.append(stripped_line)
        preview_text = " ".join(non_empty_lines[:3])[:220]
        if symbols:
            symbol_preview = ", ".join(symbols[:8])
        else:
            symbol_preview = "no obvious exported symbols"
        return f"{relative_path}: symbols [{symbol_preview}]. preview: {preview_text}"

    def _build_chunks(self, relative_path: str, file_lines: list[str], symbols: list[str], window_size: int = 60, overlap_size: int = 10) -> list[CodeChunkRecord]:
        """Split files into bounded chunks before retrieval happens.

        The implementation always creates bounded chunks, even for short files. This
        prevents retrieval from passing full files by default and keeps prompt costs
        predictable.
        """
        chunk_records = []
        current_start = 0
        while current_start < len(file_lines):
            current_end = min(len(file_lines), current_start + window_size)
            chunk_lines = file_lines[current_start:current_end]
            chunk_content = "\n".join(chunk_lines)
            chunk_symbols = self._symbols_for_chunk(symbols, chunk_content)
            chunk_summary = f"{relative_path}:{current_start + 1}-{current_end}"
            chunk_record = CodeChunkRecord(
                relative_path,
                current_start + 1,
                current_end,
                chunk_summary,
                chunk_content,
                chunk_symbols,
            )
            chunk_records.append(chunk_record)
            if current_end == len(file_lines):
                break
            current_start = current_end - overlap_size
        return chunk_records

    def _symbols_for_chunk(self, symbols: list[str], chunk_content: str) -> list[str]:
        """Return only the symbols that appear in this specific chunk."""
        chunk_symbols = []
        for symbol_name in symbols:
            if symbol_name in chunk_content:
                chunk_symbols.append(symbol_name)
            if len(chunk_symbols) >= 12:
                break
        return chunk_symbols

    def _register_symbols(self, indexed_file: IndexedFileRecord) -> None:
        """Update the symbol-to-file lookup table for direct symbol recall."""
        for symbol_name in indexed_file.symbols:
            lowered_name = symbol_name.lower()
            if lowered_name not in self.symbol_to_paths:
                self.symbol_to_paths[lowered_name] = set()
            self.symbol_to_paths[lowered_name].add(indexed_file.path)
