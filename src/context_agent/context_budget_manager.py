"""Token counting and prompt trimming utilities.

This module owns the hard 5,000-token policy. Every model call must pass through
this manager so the limit is enforced programmatically rather than by convention.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

from .agent_models import PromptSection
from .prompt_observability import PromptAssemblyReport


class TokenBudgetManager:
    """Tracks model token limits and trims prompt sections to fit."""

    def __init__(self, model_name: str, max_total_tokens: int = 5000, reserved_output_tokens: int = 900) -> None:
        self.model_name = model_name
        self.max_total_tokens = max_total_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.encoding = self._load_encoding(model_name)

    def _load_encoding(self, model_name: str):
        """Load the best available tokenizer for the configured model."""
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def get_max_input_tokens(self) -> int:
        """Return the request budget after reserving space for model output."""
        return self.max_total_tokens - self.reserved_output_tokens

    def count_text_tokens(self, text: str) -> int:
        """Count tokens for one text block."""
        if self.encoding is None:
            return max(1, len(text.split()))
        return len(self.encoding.encode(text))

    def trim_sections_with_caps(self, sections: Iterable[PromptSection], per_label_caps: dict[str, int]) -> tuple[list[PromptSection], PromptAssemblyReport]:
        """Trim sections with both per-label caps and a global cap."""
        ordered_sections = sorted(sections, key=self._sort_by_priority, reverse=True)
        label_token_usage = defaultdict(int)
        selected_sections = []
        total_used_tokens = 0
        report = PromptAssemblyReport()

        for section in ordered_sections:
            original_token_count = self.count_text_tokens(section.text)
            section_cap = per_label_caps.get(section.label)
            candidate_section = section
            section_status = "kept"
            detail_text = ""

            if section_cap is not None:
                remaining_label_tokens = section_cap - label_token_usage[section.label]
                if remaining_label_tokens <= 0:
                    report.add_decision(section.label, "dropped", original_token_count, section.priority, "label token cap reached")
                    continue
                truncated_section = self._truncate_section_to_token_limit(section, remaining_label_tokens)
                if truncated_section is None:
                    report.add_decision(section.label, "dropped", original_token_count, section.priority, "nothing fit under label cap")
                    continue
                candidate_section = truncated_section
                if candidate_section.text != section.text:
                    section_status = "truncated"
                    detail_text = "trimmed by label cap"

            candidate_tokens = self.count_text_tokens(candidate_section.text)
            if total_used_tokens + candidate_tokens > self.get_max_input_tokens():
                report.add_decision(section.label, "dropped", candidate_tokens, section.priority, "global token cap reached")
                continue

            selected_sections.append(candidate_section)
            total_used_tokens += candidate_tokens
            label_token_usage[candidate_section.label] += candidate_tokens
            report.add_decision(candidate_section.label, section_status, candidate_tokens, candidate_section.priority, detail_text)

        report.set_total_input_tokens(total_used_tokens)
        return selected_sections, report

    def _truncate_section_to_token_limit(self, section: PromptSection, token_limit: int) -> PromptSection | None:
        """Return a shortened copy of a section that fits the requested token limit."""
        if token_limit <= 0:
            return None
        if self.count_text_tokens(section.text) <= token_limit:
            return section

        lines = section.text.splitlines()
        truncated_lines = []
        for line in lines:
            truncated_lines.append(line)
            candidate_text = "\n".join(truncated_lines)
            if self.count_text_tokens(candidate_text) > token_limit:
                truncated_lines.pop()
                break

        if not truncated_lines:
            words = section.text.split()
            shortened_words = []
            for word in words:
                shortened_words.append(word)
                candidate_text = " ".join(shortened_words)
                if self.count_text_tokens(candidate_text) > token_limit:
                    shortened_words.pop()
                    break
            if not shortened_words:
                return None
            truncated_text = " ".join(shortened_words)
        else:
            truncated_text = "\n".join(truncated_lines)

        ellipsis = "\n[truncated to fit token cap]"
        if self.count_text_tokens(truncated_text + ellipsis) <= token_limit:
            truncated_text += ellipsis
        return PromptSection(section.label, truncated_text, section.priority)

    def _sort_by_priority(self, section: PromptSection) -> int:
        """Return the numeric priority used for deterministic prompt trimming."""
        return section.priority
