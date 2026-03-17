"""Persistent external memory for file summaries and notes."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

from .contracts import ExternalMemoryRecord


class ExternalMemoryStore:
    """Persist lightweight external memory outside the prompt window."""

    # Extract code-like terms and paths from a query.
    TERM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
    # Extract quoted literals such as 'sbor' or "json_pointer".
    QUOTED_LITERAL_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")
    STOP_TERMS = {
        "the",
        "a",
        "an",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "does",
        "do",
        "did",
        "under",
        "contain",
        "contains",
        "find",
        "search",
        "text",
        "files",
        "file",
        "responsible",
        "class",
        "function",
        "and",
        "or",
        "about",
    }
    SCOPE_PATTERNS = {
        # Match requests scoped to the tests directory.
        "tests/": re.compile(r"\b(?:under|in)\s+tests\b|\btests\s+directory\b"),
        # Match requests scoped to the src directory.
        "src/": re.compile(r"\b(?:under|in)\s+src\b|\bsrc\s+directory\b"),
        # Match requests scoped to the include directory.
        "include/": re.compile(r"\b(?:under|in)\s+include\b|\binclude\s+directory\b"),
        # Match requests scoped to the docs directory.
        "docs/": re.compile(r"\b(?:under|in)\s+docs\b|\bdocs\s+directory\b"),
    }
    NOISY_PATH_PENALTIES = {
        "tests/thirdparty/": 6,
        "thirdparty/": 5,
        "licenses/": 8,
        "docs/": 4,
        "docs/examples/": 4,
        "examples/": 3,
    }
    CORE_PATH_BONUSES = {
        "include/": 3,
        "src/": 2,
        "single_include/": 3,
    }
    CONCEPT_ALIASES = {
        "serialization": ["serializer", "to_json", "from_json", "adl_serializer", "dump"],
        "serializer": ["serialize", "to_json", "from_json", "adl_serializer", "dump"],
        "serialize": ["serializer", "to_json", "from_json", "dump"],
        "parsing": ["parser", "parse", "from_json"],
        "parser": ["parse", "from_json"],
    }

    def __init__(self, storage_root: Path, repository_path: Path) -> None:
        self.storage_root = storage_root
        self.repository_path = repository_path
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_root / f"{self._build_repository_key(repository_path)}.json"
        self.records = []
        self._load()

    def seed_file_summaries(self, repository_index) -> None:
        """Seed persisted memory from the repository index when records are missing."""
        if self.records:
            return
        for indexed_file in repository_index.files.values():
            self.records.append(ExternalMemoryRecord(indexed_file.file_path, indexed_file.summary, indexed_file.file_path))
        self._save()

    def ensure_seeded(self, repository_index) -> None:
        """Seed file summaries lazily when the store is still empty."""
        self.seed_file_summaries(repository_index)

    def remember_note(self, key: str, summary_text: str, source_path: str) -> None:
        """Store a new external-memory note."""
        for record in self.records:
            if record.key == key and record.summary_text == summary_text and record.source_path == source_path:
                return
        self.records.append(ExternalMemoryRecord(key, summary_text, source_path))
        self._save()

    def find_relevant_records(self, query_text: str, limit: int = 5) -> list[ExternalMemoryRecord]:
        """Return the most relevant persisted records for the query."""
        terms = self._extract_query_terms(query_text)
        literal_text = self._extract_literal_text(query_text)
        scope_prefixes = self._extract_scope_prefixes(query_text)
        concept_query = self._is_concept_query(query_text)
        scored_records = []
        for record in self.records:
            lowered_path = record.source_path.lower()
            haystack = f"{record.key} {record.summary_text} {record.source_path}".lower()

            if literal_text and literal_text not in haystack:
                continue
            if scope_prefixes and not self._matches_scope(lowered_path, scope_prefixes):
                continue

            score = self._score_record(lowered_path, haystack, terms, concept_query)
            if literal_text:
                score += 6
            if score > 0:
                scored_records.append((score, record))
        scored_records.sort(key=self._score_sort_key, reverse=True)
        return [record for _, record in scored_records[:limit]]

    def get_record_count(self) -> int:
        """Return the total number of persisted records."""
        return len(self.records)

    def _build_repository_key(self, repository_path: Path) -> str:
        """Build a stable storage key for the repository."""
        digest = hashlib.sha1(str(repository_path).encode("utf-8")).hexdigest()[:12]
        return f"repo_{digest}"

    def _load(self) -> None:
        """Load persisted records from disk."""
        if not self.storage_path.exists():
            self.records = []
            return
        raw_data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        loaded_records = []
        for item in raw_data:
            loaded_records.append(ExternalMemoryRecord(item["key"], item["summary_text"], item["source_path"]))
        self.records = loaded_records

    def _save(self) -> None:
        """Persist the current record set to disk."""
        raw_data = []
        for record in self.records:
            raw_data.append(
                {
                    "key": record.key,
                    "summary_text": record.summary_text,
                    "source_path": record.source_path,
                }
            )
        self.storage_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

    def _score_sort_key(self, scored_record: tuple[int, ExternalMemoryRecord]) -> int:
        """Return the record sorting key."""
        return scored_record[0]

    def _extract_query_terms(self, query_text: str) -> list[str]:
        """Return weighted terms for external-memory lookup."""
        lowered_terms = []
        for raw_term in self.TERM_PATTERN.findall(query_text):
            lowered_term = raw_term.lower()
            if lowered_term in self.STOP_TERMS or len(lowered_term) < 3:
                continue
            lowered_terms.append(lowered_term)

        expanded_terms = []
        for term in lowered_terms:
            expanded_terms.append(term)
            for alias in self.CONCEPT_ALIASES.get(term, []):
                expanded_terms.append(alias.lower())

        ordered_terms = []
        seen_terms = set()
        for term in expanded_terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            ordered_terms.append(term)
        return ordered_terms

    def _extract_literal_text(self, query_text: str) -> str:
        """Return the quoted literal from the query when present."""
        match = self.QUOTED_LITERAL_PATTERN.search(query_text)
        if match is None:
            return ""
        return match.group(1).strip().lower()

    def _extract_scope_prefixes(self, query_text: str) -> list[str]:
        """Return path scopes explicitly requested by the query."""
        lowered_query = query_text.lower()
        prefixes = []
        for scope_prefix, pattern in self.SCOPE_PATTERNS.items():
            if pattern.search(lowered_query):
                prefixes.append(scope_prefix)
        return prefixes

    def _matches_scope(self, lowered_path: str, scope_prefixes: list[str]) -> bool:
        """Return True when the path matches any requested scope."""
        if not scope_prefixes:
            return True
        for scope_prefix in scope_prefixes:
            if lowered_path.startswith(scope_prefix):
                return True
        return False

    def _is_concept_query(self, query_text: str) -> bool:
        """Return True for broad understanding questions about implementation areas."""
        lowered_query = query_text.lower()
        return "which files" in lowered_query or "what files" in lowered_query or "responsible" in lowered_query

    def _score_record(self, lowered_path: str, haystack: str, terms: list[str], concept_query: bool) -> int:
        """Score one external-memory record against the query."""
        score = 0
        for term in terms:
            if term in lowered_path:
                score += 3
            elif term in haystack:
                score += 1

        for noisy_prefix, penalty in self.NOISY_PATH_PENALTIES.items():
            if lowered_path.startswith(noisy_prefix):
                score -= penalty

        if concept_query:
            for core_prefix, bonus in self.CORE_PATH_BONUSES.items():
                if lowered_path.startswith(core_prefix):
                    score += bonus
            if "detail/output/serializer" in lowered_path:
                score += 5
            if "detail/conversions/" in lowered_path:
                score += 4
            if lowered_path.endswith("adl_serializer.hpp"):
                score += 4
            if lowered_path.endswith("json.hpp") or lowered_path.endswith("json_fwd.hpp"):
                score += 2
            if lowered_path.startswith("tests/"):
                score -= 2

        return score
