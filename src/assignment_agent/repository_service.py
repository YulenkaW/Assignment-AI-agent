"""Repository retrieval services for path, symbol, keyword, and chunk lookup."""

from __future__ import annotations

from pathlib import Path
import logging
import re

from .contracts import RetrievalBatch, RetrievalCandidate, RetrievalPlan
from .repository_index import RepositoryIndex


class QueryTermSet:
    """Store query terms separated by retrieval importance."""

    def __init__(
        self,
        query_text: str,
        exact_identifiers: list[str],
        compound_identifiers: list[str],
        broad_terms: list[str],
        action_terms: list[str],
        penalized_broad_terms: list[str],
        concept_terms: list[str],
        asks_for_files: bool,
    ) -> None:
        self.query_text = query_text
        self.exact_identifiers = exact_identifiers
        self.compound_identifiers = compound_identifiers
        self.broad_terms = broad_terms
        self.action_terms = action_terms
        self.penalized_broad_terms = penalized_broad_terms
        self.concept_terms = concept_terms
        self.asks_for_files = asks_for_files


class RepositoryService:
    """Provide hybrid retrieval for codebase questions."""

    # Extract identifiers, namespaces, and path-like query terms.
    TERM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_:/.\-]*")
    # Split CamelCase names into searchable subterms.
    CAMEL_SPLIT_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
    ACTION_TERMS = {"build", "fail", "failure", "explain", "where", "what", "why", "run", "test", "debug", "find", "defined"}
    STOP_TERMS = {
        "does",
        "is",
        "it",
        "class",
        "function",
        "file",
        "files",
        "the",
        "a",
        "an",
        "for",
        "about",
        "which",
        "under",
        "directory",
        "contain",
        "contains",
        "match",
        "matches",
        "search",
        "text",
        "json",
        "responsible",
    }
    # Match questions asking which files own or implement a concept.
    FILE_REQUEST_PATTERN = re.compile(r"\b(?:which|what)\s+files?\b|\bresponsible\b")
    FILE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_./-]+\.(?:hpp|h|cpp|cc|cxx|ipp|inl)$")
    CONCEPT_ALIASES = {
        "serialization": ["serialize", "serializer", "to_json", "from_json", "adl_serializer", "dump"],
        "serializer": ["serialize", "to_json", "from_json", "adl_serializer", "dump"],
        "serialize": ["serializer", "to_json", "from_json", "dump"],
        "parsing": ["parser", "parse", "sax_parse", "from_json"],
        "parser": ["parse", "sax_parse", "from_json"],
    }

    def __init__(self, repository_index: RepositoryIndex) -> None:
        self.repository_index = repository_index
        self.logger = logging.getLogger(__name__)

    def retrieve(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Run the requested retrieval plan."""
        if retrieval_plan.retrieval_steps:
            return self._retrieve_from_steps(retrieval_plan)
        if retrieval_plan.search_type == "execution_guided":
            return self._execution_guided_search(retrieval_plan)
        if retrieval_plan.search_type == "literal_text":
            return self._literal_text_search(retrieval_plan)
        if retrieval_plan.search_type == "path":
            batch = self._path_search(retrieval_plan)
            if batch.candidates:
                return batch
            return self._keyword_search(retrieval_plan)
        if retrieval_plan.search_type == "symbol":
            batch = self._symbol_search(retrieval_plan)
            if batch.candidates:
                return batch
            return self._keyword_search(retrieval_plan)
        return self._keyword_search(retrieval_plan)

    def _retrieve_from_steps(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Run the ordered fallback chain until the evidence target is met."""
        merged_candidates = []
        merged_search_type = retrieval_plan.search_type
        for retrieval_step in retrieval_plan.retrieval_steps:
            batch = self._run_step(retrieval_step.operator_name, retrieval_plan)
            if not batch.candidates:
                continue
            expanded_candidates = self._expand_neighbors(batch.candidates, retrieval_plan)
            merged_candidates = self._merge_candidates(merged_candidates, expanded_candidates, retrieval_plan.limit)
            merged_search_type = retrieval_step.operator_name
            if self._meets_evidence_target(merged_candidates, retrieval_plan):
                break
        return self._finalize_batch(merged_candidates, merged_search_type, retrieval_plan)

    def _run_step(self, operator_name: str, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Execute one retrieval operator from the fallback chain."""
        if operator_name == "execution_guided":
            return self._execution_guided_search(retrieval_plan)
        if operator_name == "literal_text":
            return self._literal_text_search(retrieval_plan)
        if operator_name == "path":
            return self._path_search(retrieval_plan)
        if operator_name == "symbol":
            return self._symbol_search(retrieval_plan)
        return self._keyword_search(retrieval_plan)

    def _execution_guided_search(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Retrieve chunks around execution evidence."""
        candidates = []
        dropped_candidates = []
        for file_index, file_path in enumerate(retrieval_plan.preferred_files):
            indexed_file = self.repository_index.find_by_path_suffix(file_path)
            if indexed_file is None:
                continue
            line_number = None
            if file_index < len(retrieval_plan.preferred_lines):
                line_number = retrieval_plan.preferred_lines[file_index]
            chunk = self._find_chunk_for_line(indexed_file, line_number)
            candidate = RetrievalCandidate(
                indexed_file.file_path,
                chunk.get_location_text(),
                100 - file_index,
                "execution evidence location",
                chunk,
            )
            if len(candidates) < retrieval_plan.limit:
                candidates.append(candidate)
            else:
                dropped_candidates.append(candidate)
            self._maybe_add_related_implementation(indexed_file.file_path, candidates, dropped_candidates, retrieval_plan.limit, retrieval_plan)
        return RetrievalBatch(
            candidates,
            "execution_guided",
            dropped_candidates,
            literal_text=retrieval_plan.literal_text,
            scope_prefixes=retrieval_plan.scope_prefixes,
        )

    def _literal_text_search(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Retrieve files by exact or near-exact text matches."""
        literal_text = retrieval_plan.literal_text.strip()
        if not literal_text:
            return self._keyword_search(retrieval_plan)

        lowered_literal = literal_text.lower()
        candidates = []
        term_set = self._parse_query(retrieval_plan.query_text)
        for indexed_file in self._iter_indexed_files(retrieval_plan):
            best_candidate = None
            for chunk in indexed_file.chunks:
                lowered_content = chunk.content.lower()
                if lowered_literal not in lowered_content:
                    continue
                score = 180 + lowered_content.count(lowered_literal) * 5
                score += self._score_repository_area(indexed_file.file_path.lower(), term_set, [])
                candidate = RetrievalCandidate(
                    indexed_file.file_path,
                    chunk.get_location_text(),
                    score,
                    "literal text match",
                    chunk,
                )
                if best_candidate is None or candidate.relevance_score > best_candidate.relevance_score:
                    best_candidate = candidate
            if best_candidate is not None:
                candidates.append(best_candidate)
        self.logger.debug("literal_text_search query=%r literal=%r matches=%d", retrieval_plan.query_text, literal_text, len(candidates))
        return self._finalize_batch(candidates, "literal_text", retrieval_plan)

    def _path_search(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Retrieve by path using exact identifiers first, broad terms second."""
        term_set = self._parse_query(retrieval_plan.query_text)
        candidates = []
        for indexed_file in self._iter_indexed_files(retrieval_plan):
            reasons = []
            score = self._score_candidate(indexed_file, term_set, reasons)
            if score <= 0 or not indexed_file.chunks:
                continue
            candidates.append(
                RetrievalCandidate(
                    indexed_file.file_path,
                    indexed_file.chunks[0].get_location_text(),
                    score,
                    ", ".join(reasons),
                    indexed_file.chunks[0],
                )
            )
        self.logger.debug(
            "path_search query=%r identifiers=%s broad_terms=%s matches=%d",
            retrieval_plan.query_text,
            term_set.exact_identifiers,
            term_set.broad_terms,
            len(candidates),
        )
        return self._finalize_batch(candidates, "path", retrieval_plan)

    def _symbol_search(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Retrieve files and chunks by exact symbol names."""
        term_set = self._parse_query(retrieval_plan.query_text)
        candidates = []
        seen_paths = set()
        prioritized_terms = term_set.exact_identifiers + term_set.compound_identifiers + term_set.broad_terms + term_set.penalized_broad_terms
        for term in prioritized_terms:
            symbol_paths = self.repository_index.symbol_map.get(term.lower(), [])
            for file_path in symbol_paths:
                if file_path in seen_paths:
                    continue
                if not self._path_is_in_scope(file_path, retrieval_plan.scope_prefixes):
                    continue
                seen_paths.add(file_path)
                indexed_file = self.repository_index.files[file_path]
                best_chunk = self._find_symbol_chunk(indexed_file, term)
                score = 20
                reason = "symbol match"
                if term in term_set.exact_identifiers:
                    score = 95
                    reason = "exact identifier symbol match"
                elif term in term_set.compound_identifiers:
                    score = 60
                    reason = "compound identifier symbol match"
                elif term in term_set.penalized_broad_terms:
                    score = 6
                    reason = "fallback broad symbol match"
                score += self._score_repository_area(indexed_file.file_path.lower(), term_set, [])
                candidates.append(
                    RetrievalCandidate(
                        file_path,
                        best_chunk.get_location_text(),
                        score,
                        reason,
                        best_chunk,
                    )
                )
                self._maybe_add_related_implementation(file_path, candidates, [], retrieval_plan.limit, retrieval_plan)
        self.logger.debug("symbol_search query=%r identifiers=%s matches=%d", retrieval_plan.query_text, term_set.exact_identifiers, len(candidates))
        return self._finalize_batch(candidates, "symbol", retrieval_plan)

    def _keyword_search(self, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Retrieve chunks by content overlap with identifier-first weights."""
        term_set = self._parse_query(retrieval_plan.query_text)
        candidates = []
        for indexed_file in self._iter_indexed_files(retrieval_plan):
            best_candidate = None
            for chunk in indexed_file.chunks:
                score = self._score_chunk(chunk, indexed_file, term_set)
                score += self._score_repository_area(chunk.file_path.lower(), term_set, [])
                if score <= 0:
                    continue
                candidate = RetrievalCandidate(
                    indexed_file.file_path,
                    chunk.get_location_text(),
                    score,
                    "keyword overlap",
                    chunk,
                )
                if best_candidate is None or candidate.relevance_score > best_candidate.relevance_score:
                    best_candidate = candidate
            if best_candidate is not None:
                candidates.append(best_candidate)
        self.logger.debug("keyword_search query=%r identifiers=%s matches=%d", retrieval_plan.query_text, term_set.exact_identifiers, len(candidates))
        return self._finalize_batch(candidates, "keyword", retrieval_plan)

    def _parse_query(self, query_text: str) -> QueryTermSet:
        """Split the query into exact identifiers, broad terms, and action terms."""
        exact_identifiers = []
        compound_identifiers = []
        broad_terms = []
        action_terms = []
        seen_terms = set()

        lowered_query = query_text.lower()
        asks_for_files = self.FILE_REQUEST_PATTERN.search(lowered_query) is not None

        for raw_term in self.TERM_PATTERN.findall(query_text):
            lowered_term = raw_term.lower()
            if lowered_term in seen_terms or lowered_term in self.STOP_TERMS:
                continue
            seen_terms.add(lowered_term)

            if lowered_term in self.ACTION_TERMS:
                action_terms.append(lowered_term)
                continue

            if self._looks_like_code_identifier(raw_term):
                exact_identifiers.append(lowered_term)
                continue

            variants = self._expand_token_variants(raw_term)
            if any("_" in variant or "::" in variant for variant in variants):
                compound_identifiers.extend([variant.lower() for variant in variants if variant.lower() not in self.STOP_TERMS])
                continue

            if any(character.isupper() for character in raw_term) or "::" in raw_term:
                compound_identifiers.extend([variant.lower() for variant in variants if len(variant) >= 3 and variant.lower() not in self.STOP_TERMS])
                continue

            if len(lowered_term) >= 4:
                broad_terms.append(lowered_term)

        compound_identifiers = self._unique_preserve_order(compound_identifiers)
        broad_terms = self._unique_preserve_order(broad_terms)
        penalized_broad_terms = []
        concept_terms = self._expand_concept_terms(exact_identifiers, compound_identifiers, broad_terms)

        if exact_identifiers:
            remaining_broad_terms = []
            for broad_term in broad_terms:
                if self._is_parent_term_of_identifier(broad_term, exact_identifiers):
                    penalized_broad_terms.append(broad_term)
                else:
                    remaining_broad_terms.append(broad_term)
            broad_terms = remaining_broad_terms

        for concept_term in concept_terms:
            if "_" in concept_term or "::" in concept_term:
                if concept_term not in compound_identifiers:
                    compound_identifiers.append(concept_term)
            elif concept_term not in broad_terms and concept_term not in self.STOP_TERMS:
                broad_terms.append(concept_term)

        return QueryTermSet(
            lowered_query,
            self._unique_preserve_order(exact_identifiers),
            self._unique_preserve_order(compound_identifiers),
            self._unique_preserve_order(broad_terms),
            self._unique_preserve_order(action_terms),
            self._unique_preserve_order(penalized_broad_terms),
            self._unique_preserve_order(concept_terms),
            asks_for_files,
        )

    def _expand_concept_terms(
        self,
        exact_identifiers: list[str],
        compound_identifiers: list[str],
        broad_terms: list[str],
    ) -> list[str]:
        """Expand broad repository concepts into implementation-oriented terms."""
        expanded_terms = []
        for term in exact_identifiers + compound_identifiers + broad_terms:
            for alias in self.CONCEPT_ALIASES.get(term, []):
                expanded_terms.append(alias.lower())
        return self._unique_preserve_order(expanded_terms)

    def _looks_like_code_identifier(self, raw_term: str) -> bool:
        """Return True when the raw term should dominate retrieval."""
        lowered_term = raw_term.lower()
        if "_" in raw_term and len(raw_term) >= 5:
            return True
        if "::" in raw_term:
            return True
        if any(character.isupper() for character in raw_term) and len(raw_term) >= 5:
            return True
        if lowered_term.endswith((".hpp", ".h", ".cpp", ".cc", ".cxx")):
            return True
        return False

    def _looks_like_file_identifier(self, identifier: str) -> bool:
        """Return True when an identifier names a concrete source file."""
        return self.FILE_NAME_PATTERN.fullmatch(identifier) is not None

    def _expand_token_variants(self, raw_term: str) -> list[str]:
        """Preserve the full code token plus weaker split variants."""
        variants = [raw_term]
        namespace_parts = raw_term.split("::")
        if len(namespace_parts) > 1:
            for part in namespace_parts:
                if part:
                    variants.append(part)
        underscore_parts = []
        for part in namespace_parts:
            underscore_parts.extend(part.split("_"))
        for part in underscore_parts:
            if part:
                variants.append(part)
            camel_parts = self.CAMEL_SPLIT_PATTERN.findall(part)
            for camel_part in camel_parts:
                if camel_part:
                    variants.append(camel_part)
        return self._unique_preserve_order([variant for variant in variants if variant])

    def _score_candidate(self, indexed_file, term_set: QueryTermSet, reasons: list[str]) -> int:
        """Score a file-level candidate with identifier-first weighting."""
        lowered_path = indexed_file.file_path.lower()
        file_name = Path(indexed_file.file_path).name.lower()
        file_stem = Path(file_name).stem
        symbol_set = set(symbol_name.lower() for symbol_name in indexed_file.symbols)
        combined_content = " ".join(chunk.content.lower() for chunk in indexed_file.chunks[:2])
        score = 0

        for identifier in term_set.exact_identifiers:
            if identifier == file_stem:
                score += 120
                reasons.append("exact identifier filename match")
            score += self._score_identifier_path_match(identifier, lowered_path, file_name, reasons)
            if identifier in symbol_set:
                score += 95
                reasons.append("identifier symbol match")
            if identifier in combined_content:
                score += 80
                reasons.append("identifier content match")
            split_terms = self._split_identifier(identifier)
            if split_terms and self._all_subtokens_in_text(split_terms, lowered_path):
                score += 25
                reasons.append("identifier subtoken path support")

        for identifier in term_set.compound_identifiers:
            if identifier in lowered_path:
                score += 45
                reasons.append("compound path match")
            if identifier in symbol_set:
                score += 35
                reasons.append("compound symbol match")
            if identifier in combined_content:
                score += 20
                reasons.append("compound content match")

        if term_set.exact_identifiers:
            score += self._score_broad_terms(term_set.penalized_broad_terms, lowered_path, combined_content, 2, reasons, "fallback")
        else:
            score += self._score_broad_terms(term_set.broad_terms, lowered_path, combined_content, 10, reasons, "broad")

        score += self._score_repository_area(lowered_path, term_set, reasons)
        return score

    def _score_identifier_path_match(
        self,
        identifier: str,
        lowered_path: str,
        file_name: str,
        reasons: list[str] | None = None,
    ) -> int:
        """Score exact file names strongly without rewarding loose filename substrings."""
        if self._looks_like_file_identifier(identifier):
            if identifier == file_name:
                if reasons is not None:
                    reasons.append("exact file name match")
                return 180
            if lowered_path.endswith(f"/{identifier}") or lowered_path.endswith(f"\\{identifier}"):
                if reasons is not None:
                    reasons.append("exact path suffix match")
                return 150
            return 0

        if identifier in lowered_path:
            if reasons is not None:
                reasons.append("identifier path match")
            return 100
        return 0

    def _score_repository_area(self, lowered_path: str, term_set: QueryTermSet, reasons: list[str]) -> int:
        """Bias definition questions toward library source rather than examples/docs."""
        score = 0
        seeks_definition = any(action in term_set.action_terms for action in ("where", "defined"))
        targets_serialization = any(
            term in term_set.concept_terms or term in term_set.broad_terms or term in term_set.compound_identifiers
            for term in ("serialization", "serializer", "serialize", "to_json", "from_json", "adl_serializer", "dump")
        )
        prefer_core = seeks_definition or term_set.asks_for_files or targets_serialization
        if prefer_core:
            if lowered_path.startswith("include/") or lowered_path.startswith("src/"):
                score += 25
                if reasons is not None:
                    reasons.append("library source preference")
            if lowered_path.startswith("docs/") or "/examples/" in lowered_path:
                score -= 20
                if reasons is not None:
                    reasons.append("example/docs penalty")
            if lowered_path.startswith("tests/"):
                score -= 10
                if reasons is not None:
                    reasons.append("test path penalty")
        if targets_serialization:
            if "detail/output/serializer" in lowered_path:
                score += 80
                if reasons is not None:
                    reasons.append("serializer implementation")
            if "detail/conversions/to_json" in lowered_path or "detail/conversions/from_json" in lowered_path:
                score += 60
                if reasons is not None:
                    reasons.append("conversion implementation")
            if lowered_path.endswith("adl_serializer.hpp"):
                score += 50
                if reasons is not None:
                    reasons.append("adl serializer customization")
            if lowered_path.endswith("json.hpp"):
                score += 45
                if reasons is not None:
                    reasons.append("public serialization entrypoints")
            if lowered_path.startswith("docs/") or "/examples/" in lowered_path:
                score -= 35
                if reasons is not None:
                    reasons.append("non-core example penalty")
            if lowered_path.startswith("tests/"):
                score -= 20
                if reasons is not None:
                    reasons.append("non-core test penalty")
        return score

    def _score_chunk(self, chunk, indexed_file, term_set: QueryTermSet) -> int:
        """Score a chunk by content overlap and symbol presence."""
        lowered_content = chunk.content.lower()
        lowered_path = chunk.file_path.lower()
        file_name = Path(chunk.file_path).name.lower()
        symbol_set = set(symbol_name.lower() for symbol_name in indexed_file.symbols)
        score = 0

        for identifier in term_set.exact_identifiers:
            if identifier == Path(file_name).stem:
                score += 120
            score += self._score_identifier_path_match(identifier, lowered_path, file_name)
            if identifier in symbol_set:
                score += 95
            if identifier in lowered_content:
                score += 80
            split_terms = self._split_identifier(identifier)
            if split_terms and self._all_subtokens_in_text(split_terms, lowered_path):
                score += 25

        for identifier in term_set.compound_identifiers:
            if identifier in lowered_path:
                score += 15
            if identifier in symbol_set:
                score += 35
            if identifier in lowered_content:
                score += 8

        if term_set.exact_identifiers:
            score += self._score_broad_terms(term_set.penalized_broad_terms, lowered_path, lowered_content, 2, [], "fallback")
            return score

        score += self._score_broad_terms(term_set.broad_terms, lowered_path, lowered_content, 10, [], "broad")
        return score

    def _split_identifier(self, identifier: str) -> list[str]:
        """Split one identifier into weaker support subtokens."""
        split_terms = []
        for namespace_part in identifier.split("::"):
            for token in namespace_part.split("_"):
                lowered_token = token.lower()
                if lowered_token and lowered_token not in self.STOP_TERMS:
                    split_terms.append(lowered_token)
        return self._unique_preserve_order(split_terms)

    def _is_parent_term_of_identifier(self, broad_term: str, exact_identifiers: list[str]) -> bool:
        """Return True when a broad term is a component of a more specific identifier."""
        for identifier in exact_identifiers:
            if broad_term in self._split_identifier(identifier):
                return True
        return False

    def _all_subtokens_in_text(self, subtokens: list[str], text: str) -> bool:
        """Return True when every subtoken appears in the target text."""
        if not subtokens:
            return False
        for subtoken in subtokens:
            if subtoken not in text:
                return False
        return True

    def _score_broad_terms(
        self,
        broad_terms: list[str],
        path_text: str,
        content_text: str,
        weight: int,
        reasons: list[str],
        label: str,
    ) -> int:
        """Score broad terms weakly, especially when identifiers already exist."""
        score = 0
        for term in broad_terms:
            if term in path_text:
                score += weight
                if reasons is not None:
                    reasons.append(f"{label} path match")
            if term in content_text:
                score += weight
                if reasons is not None:
                    reasons.append(f"{label} content match")
        return score

    def _find_chunk_for_line(self, indexed_file, line_number: int | None):
        """Find the chunk that covers the given line."""
        if line_number is None:
            return indexed_file.chunks[0]
        for chunk in indexed_file.chunks:
            if chunk.start_line <= line_number <= chunk.end_line:
                return chunk
        return indexed_file.chunks[0]

    def _find_symbol_chunk(self, indexed_file, symbol_name: str):
        """Find the chunk that contains the requested symbol."""
        for chunk in indexed_file.chunks:
            for chunk_symbol in chunk.symbols:
                if chunk_symbol.lower() == symbol_name.lower():
                    return chunk
        return indexed_file.chunks[0]

    def _maybe_add_related_implementation(
        self,
        file_path: str,
        candidates: list[RetrievalCandidate],
        dropped_candidates: list[RetrievalCandidate],
        limit: int,
        retrieval_plan: RetrievalPlan,
    ) -> None:
        """Add the matching source file when the current file is a header."""
        extension = Path(file_path).suffix.lower()
        if extension not in (".h", ".hpp", ".hh", ".hxx"):
            return
        base_name = Path(file_path).stem
        for indexed_file in self.repository_index.files.values():
            if Path(indexed_file.file_path).stem != base_name:
                continue
            if Path(indexed_file.file_path).suffix.lower() not in (".c", ".cc", ".cpp", ".cxx"):
                continue
            if not self._path_is_in_scope(indexed_file.file_path, retrieval_plan.scope_prefixes):
                continue
            if not indexed_file.chunks:
                continue
            candidate = RetrievalCandidate(
                indexed_file.file_path,
                indexed_file.chunks[0].get_location_text(),
                14,
                "related implementation file",
                indexed_file.chunks[0],
            )
            if len(candidates) < limit:
                candidates.append(candidate)
            elif dropped_candidates is not None:
                dropped_candidates.append(candidate)
            return

    def _unique_preserve_order(self, values: list[str]) -> list[str]:
        """Return values without duplicates while preserving order."""
        ordered_values = []
        seen_values = set()
        for value in values:
            if value in seen_values:
                continue
            seen_values.add(value)
            ordered_values.append(value)
        return ordered_values

    def _iter_indexed_files(self, retrieval_plan: RetrievalPlan):
        """Yield indexed files filtered by requested scope prefixes."""
        for indexed_file in self.repository_index.files.values():
            if self._path_is_in_scope(indexed_file.file_path, retrieval_plan.scope_prefixes):
                yield indexed_file

    def _path_is_in_scope(self, file_path: str, scope_prefixes: list[str]) -> bool:
        """Return True when the file path matches any requested scope prefix."""
        if not scope_prefixes:
            return True
        lowered_path = file_path.lower()
        for scope_prefix in scope_prefixes:
            if lowered_path.startswith(scope_prefix.lower()):
                return True
        return False

    def _merge_candidates(
        self,
        existing_candidates: list[RetrievalCandidate],
        new_candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        """Merge candidates across retrieval steps while preserving the strongest evidence."""
        merged = {}
        for candidate in existing_candidates + new_candidates:
            chunk_key = candidate.chunk.get_location_text()
            current_candidate = merged.get(chunk_key)
            if current_candidate is None or candidate.relevance_score > current_candidate.relevance_score:
                merged[chunk_key] = candidate
        ordered_candidates = sorted(merged.values(), key=self._candidate_sort_key, reverse=True)
        return ordered_candidates[: max(limit * 2, limit)]

    def _meets_evidence_target(self, candidates: list[RetrievalCandidate], retrieval_plan: RetrievalPlan) -> bool:
        """Return True when the current candidate set satisfies the evidence target."""
        if not candidates:
            return False
        unique_files = {candidate.file_path for candidate in candidates}
        requirements = retrieval_plan.evidence_requirements
        if len(unique_files) < requirements.minimum_files:
            return False
        minimum_chunks = requirements.minimum_files * max(1, requirements.supporting_chunks_per_file)
        return len(candidates) >= minimum_chunks

    def _expand_neighbors(self, candidates: list[RetrievalCandidate], retrieval_plan: RetrievalPlan) -> list[RetrievalCandidate]:
        """Add neighboring chunks when the evidence plan asks for nearby context."""
        neighbor_limit = min(retrieval_plan.evidence_requirements.neighbor_chunks, retrieval_plan.budget.neighbor_limit)
        if neighbor_limit <= 0:
            return candidates

        expanded_candidates = list(candidates)
        for candidate in list(candidates):
            indexed_file = self.repository_index.files.get(candidate.file_path)
            if indexed_file is None:
                continue
            chunk_index = self._find_chunk_index(indexed_file, candidate.chunk)
            if chunk_index == -1:
                continue
            for offset in range(1, neighbor_limit + 1):
                for neighbor_index in (chunk_index - offset, chunk_index + offset):
                    if neighbor_index < 0 or neighbor_index >= len(indexed_file.chunks):
                        continue
                    neighbor_chunk = indexed_file.chunks[neighbor_index]
                    expanded_candidates.append(
                        RetrievalCandidate(
                            indexed_file.file_path,
                            neighbor_chunk.get_location_text(),
                            max(candidate.relevance_score - 5 - offset, 1),
                            f"{candidate.reason}, neighbor expansion",
                            neighbor_chunk,
                        )
                    )
        return expanded_candidates

    def _find_chunk_index(self, indexed_file, target_chunk) -> int:
        """Return the index of a chunk inside one indexed file."""
        for chunk_index, chunk in enumerate(indexed_file.chunks):
            if chunk.start_line == target_chunk.start_line and chunk.end_line == target_chunk.end_line:
                return chunk_index
        return -1

    def _finalize_batch(self, candidates: list[RetrievalCandidate], search_type: str, retrieval_plan: RetrievalPlan) -> RetrievalBatch:
        """Sort, limit, and record dropped candidates."""
        ordered_candidates = sorted(candidates, key=self._candidate_sort_key, reverse=True)
        kept_candidates = ordered_candidates[:retrieval_plan.limit]
        dropped_candidates = ordered_candidates[retrieval_plan.limit:]
        return RetrievalBatch(
            kept_candidates,
            search_type,
            dropped_candidates,
            literal_text=retrieval_plan.literal_text,
            scope_prefixes=retrieval_plan.scope_prefixes,
        )

    def _candidate_sort_key(self, candidate: RetrievalCandidate) -> tuple[int, int]:
        """Return a deterministic ordering key."""
        return (candidate.relevance_score, -candidate.chunk.start_line)
