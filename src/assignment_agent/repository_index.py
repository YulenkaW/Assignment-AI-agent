"""Repository indexing and code-structure chunking.

This module builds the repository index used by path search, symbol lookup, and
targeted chunk retrieval. Chunks follow code structure when possible instead of fixed
text windows only.
"""

from __future__ import annotations

from pathlib import Path
import re

from .contracts import IndexedFile, RepositoryChunk


class RepositoryIndex:
    """Build and query a lightweight repository index."""

    SUPPORTED_EXTENSIONS = {".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".ipp", ".inl", ".cmake", ".txt"}
    # Match likely type, namespace, alias, and function declarations for symbol extraction.
    SYMBOL_PATTERN = re.compile(
        r"^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct|namespace|enum|using)\s+([A-Za-z_]\w*)|"
        r"^\s*(?:inline\s+|constexpr\s+|static\s+|virtual\s+)?(?:[\w:<>,~*&\s]+)\s+([A-Za-z_]\w*)\s*\(",
        re.MULTILINE,
    )
    # Match code structure boundaries used to start structural chunks.
    STRUCTURE_PATTERN = re.compile(
        r"^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct|namespace|enum)\s+[A-Za-z_]\w*|"
        r"^\s*(?:inline\s+|constexpr\s+|static\s+|virtual\s+)?(?:[\w:<>,~*&\s]+)\s+[A-Za-z_]\w*\s*\(",
        re.MULTILINE,
    )
    # Match block-style declarations that should be extended to their closing brace.
    BLOCK_DECLARATION_PATTERN = re.compile(r"\b(?:class|struct|namespace|enum)\b")

    def __init__(self, repository_path: Path) -> None:
        self.repository_path = repository_path
        self.files = {}
        self.symbol_map = {}
        self.chunk_map = {}

    def build(self) -> None:
        """Scan the repository and populate the index."""
        for file_path in self.repository_path.rglob("*"):
            if not file_path.is_file():
                continue
            if not self._is_supported_file(file_path):
                continue
            relative_path = file_path.relative_to(self.repository_path).as_posix()
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            indexed_file = self._index_file(relative_path, text)
            self.files[relative_path] = indexed_file
            self._register_symbols(indexed_file)
            self._register_chunks(indexed_file)

    def find_by_path_suffix(self, path_text: str) -> IndexedFile | None:
        """Return the indexed file that ends with the given path."""
        normalized_path = Path(path_text).as_posix()
        for indexed_path, indexed_file in self.files.items():
            if indexed_path.endswith(normalized_path):
                return indexed_file
        return None

    def _is_supported_file(self, file_path: Path) -> bool:
        """Return True when the file should be indexed."""
        if file_path.name == "CMakeLists.txt":
            return True
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _index_file(self, relative_path: str, text: str) -> IndexedFile:
        """Create the indexed representation for one file."""
        lines = text.splitlines()
        symbols = self._extract_symbols(text)
        chunks = self._build_chunks(relative_path, lines, symbols)
        summary = self._build_summary(relative_path, lines, symbols)
        file_type = Path(relative_path).suffix.lower() or Path(relative_path).name
        return IndexedFile(relative_path, file_type, symbols, summary, chunks)

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract likely exported symbols."""
        symbols = []
        for match in self.SYMBOL_PATTERN.finditer(text):
            symbol_name = match.group(1) or match.group(2)
            if symbol_name and symbol_name not in symbols:
                symbols.append(symbol_name)
        return symbols[:80]

    def _build_summary(self, relative_path: str, lines: list[str], symbols: list[str]) -> str:
        """Build the file summary used as external memory."""
        preview_lines = []
        for line in lines:
            if line.strip():
                preview_lines.append(line.strip())
            if len(preview_lines) >= 3:
                break
        symbol_text = ", ".join(symbols[:8]) if symbols else "no obvious symbols"
        preview_text = " ".join(preview_lines)[:180]
        return f"{relative_path}: symbols [{symbol_text}] preview [{preview_text}]"

    def _build_chunks(self, relative_path: str, lines: list[str], symbols: list[str]) -> list[RepositoryChunk]:
        """Chunk a file by code structure first and by window fallback second."""
        structure_boundaries = self._find_structure_boundaries(lines)
        if not structure_boundaries:
            return self._build_window_chunks(relative_path, lines, symbols)

        chunks = []
        for boundary_index in range(len(structure_boundaries)):
            start_line = structure_boundaries[boundary_index]
            if self._starts_block_declaration(lines, start_line):
                end_line = self._find_block_end(lines, start_line)
            elif boundary_index + 1 < len(structure_boundaries):
                end_line = structure_boundaries[boundary_index + 1] - 1
            else:
                end_line = min(len(lines), start_line + 79)
            if end_line < start_line:
                end_line = start_line
            content = "\n".join(lines[start_line - 1:end_line])
            chunk_symbols = self._symbols_for_content(symbols, content)
            chunk = RepositoryChunk(
                relative_path,
                start_line,
                end_line,
                "structure",
                f"{relative_path}:{start_line}-{end_line}",
                content,
                chunk_symbols,
            )
            chunks.append(chunk)

        if not chunks:
            return self._build_window_chunks(relative_path, lines, symbols)
        return chunks

    def _find_structure_boundaries(self, lines: list[str]) -> list[int]:
        """Return 1-based line numbers that start a code structure chunk."""
        boundaries = []
        for index, line in enumerate(lines, start=1):
            if self.STRUCTURE_PATTERN.search(line):
                boundaries.append(self._expand_boundary_start(lines, index))
        return boundaries

    def _expand_boundary_start(self, lines: list[str], start_line: int) -> int:
        """Include nearby doc comments or template lines before a declaration."""
        current_line = start_line
        while current_line > 1:
            previous_line = lines[current_line - 2].strip()
            if not previous_line:
                break
            if previous_line.startswith(("///", "//", "/*", "*", "*/", "template")):
                current_line -= 1
                continue
            break
        return current_line

    def _starts_block_declaration(self, lines: list[str], start_line: int) -> bool:
        """Return True when the chunk begins with a class/struct/namespace/enum block."""
        for offset in range(0, 6):
            line_index = start_line - 1 + offset
            if line_index >= len(lines):
                break
            if self.BLOCK_DECLARATION_PATTERN.search(lines[line_index]):
                return True
        return False

    def _find_block_end(self, lines: list[str], start_line: int) -> int:
        """Find the closing brace for a class-like block when possible."""
        brace_depth = 0
        saw_open_brace = False
        max_end_line = min(len(lines), start_line + 159)
        for line_number in range(start_line, max_end_line + 1):
            line = lines[line_number - 1]
            brace_depth += line.count("{")
            if "{" in line:
                saw_open_brace = True
            brace_depth -= line.count("}")
            if saw_open_brace and brace_depth <= 0:
                return line_number
        return max_end_line

    def _build_window_chunks(self, relative_path: str, lines: list[str], symbols: list[str]) -> list[RepositoryChunk]:
        """Fallback chunking for files without clear structure anchors."""
        chunks = []
        start = 0
        window_size = 60
        overlap = 10
        while start < len(lines):
            end = min(len(lines), start + window_size)
            content = "\n".join(lines[start:end])
            chunk_symbols = self._symbols_for_content(symbols, content)
            chunk = RepositoryChunk(
                relative_path,
                start + 1,
                end,
                "window",
                f"{relative_path}:{start + 1}-{end}",
                content,
                chunk_symbols,
            )
            chunks.append(chunk)
            if end == len(lines):
                break
            start = end - overlap
        return chunks

    def _symbols_for_content(self, symbols: list[str], content: str) -> list[str]:
        """Return the symbols that appear inside the chunk."""
        matching_symbols = []
        for symbol_name in symbols:
            if symbol_name in content:
                matching_symbols.append(symbol_name)
            if len(matching_symbols) >= 12:
                break
        return matching_symbols

    def _register_symbols(self, indexed_file: IndexedFile) -> None:
        """Populate the symbol lookup map."""
        for symbol_name in indexed_file.symbols:
            lowered_name = symbol_name.lower()
            if lowered_name not in self.symbol_map:
                self.symbol_map[lowered_name] = []
            self.symbol_map[lowered_name].append(indexed_file.file_path)

    def _register_chunks(self, indexed_file: IndexedFile) -> None:
        """Populate the chunk lookup map."""
        for chunk in indexed_file.chunks:
            self.chunk_map[chunk.get_location_text()] = chunk
