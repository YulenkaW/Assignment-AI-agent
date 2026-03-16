"""Retrieval logic for source files and code chunks.

The retrieval strategy stays intentionally simple: lexical matching, symbol hints,
and grounded entity recall. This is the most reliable approach for compiler errors
and C++ source navigation in a moderately sized local repository.
"""

from __future__ import annotations

from pathlib import Path
import re

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    Document = None

from .agent_models import RetrievedChunkMatch, RetrievedFileMatch
from .codebase_indexer import CodebaseIndexer
from .conversation_memory_store import SummaryMemoryStore
from .entity_fact_store import EntityFactStore


class RepositoryRetrievalEngine:
    """Ranks relevant files and code chunks for the active user query."""

    WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_:/.-]*")

    def __init__(self, indexer: CodebaseIndexer, summary_memory: SummaryMemoryStore, entity_memory: EntityFactStore) -> None:
        self.indexer = indexer
        self.summary_memory = summary_memory
        self.entity_memory = entity_memory

    def search(self, query_text: str, limit: int = 6, preferred_path: str = "") -> list[RetrievedChunkMatch]:
        """Return top chunk matches using a staged retrieval policy."""
        file_matches = self.search_files(query_text, limit=max(limit, 4), preferred_path=preferred_path)
        return self.expand_files_to_chunks(query_text, file_matches, limit)

    def search_files(self, query_text: str, limit: int = 6, preferred_path: str = "") -> list[RetrievedFileMatch]:
        """Return top file matches based on summaries, symbols, paths, and entity recall."""
        lowered_terms = self._extract_terms(query_text)
        exact_file_terms = self._extract_file_like_terms(lowered_terms)
        exact_symbol_terms = self._extract_exact_symbol_terms(lowered_terms)
        recalled_entities = self.entity_memory.recall_entities_with_sources(lowered_terms, preferred_path=preferred_path)
        recalled_names = set()
        for entity_fact in recalled_entities:
            recalled_names.add(entity_fact.entity_name.lower())

        matches = []
        for indexed_file in self.indexer.indexed_files.values():
            score, reason_parts = self._score_file(indexed_file, lowered_terms, exact_file_terms, exact_symbol_terms, recalled_names)
            if score > 0:
                reason_text = ", ".join(reason_parts)
                matches.append(RetrievedFileMatch(score, reason_text, indexed_file))

        matches.sort(key=self._sort_file_match, reverse=True)
        return matches[:limit]

    def expand_files_to_chunks(self, query_text: str, file_matches: list[RetrievedFileMatch], limit: int = 6) -> list[RetrievedChunkMatch]:
        """Expand file matches into bounded chunk matches."""
        lowered_terms = self._extract_terms(query_text)
        chunk_matches = []

        for file_match in file_matches:
            best_chunk_match = self._find_best_chunk_in_file(file_match, lowered_terms)
            if best_chunk_match is not None:
                chunk_matches.append(best_chunk_match)

        chunk_matches.sort(key=self._sort_chunk_match, reverse=True)
        trimmed_matches = chunk_matches[:limit]
        if len(trimmed_matches) >= limit:
            return trimmed_matches

        expanded_matches = list(trimmed_matches)
        used_keys = set()
        for match in expanded_matches:
            used_keys.add(self._chunk_key(match.chunk))

        for file_match in file_matches:
            neighboring_matches = self._collect_additional_chunks(file_match, lowered_terms)
            for neighboring_match in neighboring_matches:
                chunk_key = self._chunk_key(neighboring_match.chunk)
                if chunk_key in used_keys:
                    continue
                expanded_matches.append(neighboring_match)
                used_keys.add(chunk_key)
                if len(expanded_matches) >= limit:
                    expanded_matches.sort(key=self._sort_chunk_match, reverse=True)
                    return expanded_matches[:limit]

        expanded_matches.sort(key=self._sort_chunk_match, reverse=True)
        return expanded_matches[:limit]

    def find_chunk_for_file_reference(self, file_reference: str, line_number: int | None = None):
        """Return the chunk that best covers a compiler-reported file reference."""
        normalized_reference = Path(file_reference).as_posix()
        for indexed_path, indexed_file in self.indexer.indexed_files.items():
            if indexed_path.endswith(normalized_reference):
                return self._find_line_chunk(indexed_file.chunks, line_number)
        return None

    def build_langchain_documents(self, matches: list[RetrievedChunkMatch]) -> list:
        """Convert matches into LangChain Document objects when available."""
        if Document is None:
            return []
        documents = []
        for match in matches:
            metadata = {
                "path": match.chunk.path,
                "start_line": match.chunk.start_line,
                "end_line": match.chunk.end_line,
                "reason": match.reason,
            }
            documents.append(Document(page_content=match.chunk.content, metadata=metadata))
        return documents

    def _find_best_chunk_in_file(self, file_match: RetrievedFileMatch, lowered_terms: list[str]) -> RetrievedChunkMatch | None:
        """Choose the strongest bounded chunk within a matched file."""
        best_match = None
        for chunk in file_match.indexed_file.chunks:
            chunk_score = file_match.score + self._score_chunk(chunk.content, lowered_terms) + self._score_exact_symbol_hit(chunk.symbols, lowered_terms)
            if chunk_score <= 0:
                continue
            reason_text = f"{file_match.reason}, best bounded chunk"
            candidate_match = RetrievedChunkMatch(chunk_score, reason_text, chunk)
            if best_match is None or candidate_match.score > best_match.score:
                best_match = candidate_match
        return best_match

    def _collect_additional_chunks(self, file_match: RetrievedFileMatch, lowered_terms: list[str]) -> list[RetrievedChunkMatch]:
        """Collect extra bounded chunks only when first-pass evidence is insufficient."""
        additional_matches = []
        for chunk in file_match.indexed_file.chunks:
            chunk_score = file_match.score + self._score_chunk(chunk.content, lowered_terms)
            if chunk_score <= 0:
                continue
            reason_text = f"{file_match.reason}, expanded nearby chunk"
            additional_matches.append(RetrievedChunkMatch(chunk_score, reason_text, chunk))
        additional_matches.sort(key=self._sort_chunk_match, reverse=True)
        return additional_matches

    def _extract_terms(self, query_text: str) -> list[str]:
        """Extract normalized query terms from the user request."""
        terms = []
        for raw_term in self.WORD_PATTERN.findall(query_text):
            terms.append(raw_term.lower())
        return terms

    def _extract_file_like_terms(self, lowered_terms: list[str]) -> list[str]:
        """Return query terms that look like file references."""
        file_terms = []
        for term in lowered_terms:
            if "/" in term or "\\" in term or term.endswith((".hpp", ".h", ".cpp", ".cc", ".cxx", ".cmake", ".txt")):
                file_terms.append(term)
        return file_terms

    def _extract_exact_symbol_terms(self, lowered_terms: list[str]) -> list[str]:
        """Return query terms that exactly match indexed symbols."""
        symbol_terms = []
        indexed_symbols = self.indexer.symbol_to_paths
        for term in lowered_terms:
            if term in indexed_symbols:
                symbol_terms.append(term)
        return symbol_terms

    def _score_file(self, indexed_file, lowered_terms: list[str], exact_file_terms: list[str], exact_symbol_terms: list[str], recalled_names: set[str]) -> tuple[int, list[str]]:
        """Score a file by evidence value, not just generic overlap."""
        path_text = indexed_file.path.lower()
        summary_text = indexed_file.summary.lower()
        symbol_text = " ".join(indexed_file.symbols).lower()
        score = 0
        reason_parts = []

        for term in exact_file_terms:
            if term in path_text:
                score += 20
                reason_parts.append("exact file path match")

        for term in exact_symbol_terms:
            if term in symbol_text:
                score += 16
                reason_parts.append("exact symbol match")

        for recalled_name in recalled_names:
            if recalled_name in symbol_text:
                score += 8
                reason_parts.append("entity recall match")

        for term in lowered_terms:
            if term in summary_text or term in symbol_text or term in path_text:
                score += 2

        if score > 0 and not reason_parts:
            reason_parts.append("generic lexical match")
        return score, reason_parts

    def _score_chunk(self, chunk_content: str, lowered_terms: list[str]) -> int:
        """Add score when exact query terms appear in the chunk body."""
        lowered_chunk = chunk_content.lower()
        score = 0
        for term in lowered_terms:
            if term in lowered_chunk:
                score += 1
        return score

    def _score_exact_symbol_hit(self, chunk_symbols: list[str], lowered_terms: list[str]) -> int:
        """Boost chunks that contain exact symbol hits from the user query."""
        score = 0
        lowered_symbol_set = set()
        for symbol_name in chunk_symbols:
            lowered_symbol_set.add(symbol_name.lower())
        for term in lowered_terms:
            if term in lowered_symbol_set:
                score += 8
        return score

    def _find_line_chunk(self, chunks, line_number: int | None):
        """Choose the chunk that best covers the requested line number."""
        if not chunks:
            return None
        if line_number is None:
            return chunks[0]
        for chunk in chunks:
            if chunk.start_line <= line_number <= chunk.end_line:
                return chunk
        return chunks[0]

    def _chunk_key(self, chunk) -> str:
        """Build a stable key for deduplicating chunk matches."""
        return f"{chunk.path}:{chunk.start_line}:{chunk.end_line}"

    def _sort_file_match(self, match: RetrievedFileMatch) -> int:
        """Return the numeric score used for deterministic file ordering."""
        return match.score

    def _sort_chunk_match(self, match: RetrievedChunkMatch) -> int:
        """Return the numeric score used for deterministic chunk ordering."""
        return match.score
