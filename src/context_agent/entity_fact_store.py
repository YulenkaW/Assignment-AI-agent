"""Entity fact storage for grounded codebase knowledge.

This file exists separately so grounded entity behavior is easy to explain and audit.
It stores multiple facts per entity instead of collapsing everything into one record.
"""

from __future__ import annotations

from typing import Iterable


class GroundedEntityFact:
    """Stores one grounded fact about an entity."""

    def __init__(self, entity_name: str, fact_text: str, source_type: str, path_text: str = "") -> None:
        self.entity_name = entity_name
        self.fact_text = fact_text
        self.source_type = source_type
        self.path_text = path_text


class EntityFactStore:
    """Stores multiple grounded facts per entity with source tags."""

    ALLOWED_SOURCES = {"code", "build", "test", "user"}

    def __init__(self) -> None:
        self.entity_facts = {}

    def remember_grounded_entity(self, entity_name: str, fact_text: str, source_type: str, path_text: str = "") -> None:
        """Persist a grounded fact for later recall."""
        if not entity_name:
            return
        if source_type not in self.ALLOWED_SOURCES:
            raise ValueError(f"Unsupported entity source type: {source_type}")

        lowered_name = entity_name.lower()
        fact_record = GroundedEntityFact(entity_name, fact_text, source_type, path_text)
        if lowered_name not in self.entity_facts:
            self.entity_facts[lowered_name] = []

        for existing_fact in self.entity_facts[lowered_name]:
            if self._is_duplicate_fact(existing_fact, fact_record):
                return

        self.entity_facts[lowered_name].append(fact_record)

    def recall_entities(self, names: Iterable[str], preferred_path: str = "") -> list[tuple[str, str]]:
        """Return the best facts for the requested entity names."""
        results = []
        for name in names:
            lowered_name = name.lower()
            if lowered_name not in self.entity_facts:
                continue
            best_fact = self._select_best_fact(self.entity_facts[lowered_name], preferred_path)
            if best_fact is not None:
                results.append((best_fact.entity_name, best_fact.fact_text))
        return results

    def recall_entities_with_sources(self, names: Iterable[str], preferred_path: str = "") -> list[GroundedEntityFact]:
        """Return the best full fact records including their source tags."""
        results = []
        for name in names:
            lowered_name = name.lower()
            if lowered_name not in self.entity_facts:
                continue
            best_fact = self._select_best_fact(self.entity_facts[lowered_name], preferred_path)
            if best_fact is not None:
                results.append(best_fact)
        return results

    def get_all_facts_for_entity(self, entity_name: str) -> list[GroundedEntityFact]:
        """Return every stored fact for one entity name."""
        return list(self.entity_facts.get(entity_name.lower(), []))

    def _select_best_fact(self, facts: list[GroundedEntityFact], preferred_path: str) -> GroundedEntityFact | None:
        """Choose the best fact using source trust and optional path preference."""
        if not facts:
            return None
        ordered_facts = sorted(facts, key=self._make_fact_sort_key(preferred_path), reverse=True)
        return ordered_facts[0]

    def _make_fact_sort_key(self, preferred_path: str):
        """Build the score function used to rank candidate facts."""
        def sort_key(fact: GroundedEntityFact) -> int:
            return self._score_fact(fact, preferred_path)

        return sort_key

    def _score_fact(self, fact: GroundedEntityFact, preferred_path: str) -> int:
        """Score a fact for recall quality."""
        score = 0
        source_scores = {
            "code": 10,
            "build": 8,
            "test": 7,
            "user": 6,
        }
        score += source_scores.get(fact.source_type, 0)
        if preferred_path and fact.path_text and preferred_path in fact.path_text:
            score += 5
        return score

    def _is_duplicate_fact(self, left_fact: GroundedEntityFact, right_fact: GroundedEntityFact) -> bool:
        """Return True when two fact records are effectively the same."""
        return (
            left_fact.entity_name.lower() == right_fact.entity_name.lower()
            and left_fact.fact_text == right_fact.fact_text
            and left_fact.source_type == right_fact.source_type
            and left_fact.path_text == right_fact.path_text
        )
