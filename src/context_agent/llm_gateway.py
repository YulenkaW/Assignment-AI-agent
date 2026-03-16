"""LLM integration and deterministic fallback behavior.

The primary path uses LangChain prompt templates and ChatOpenAI when credentials are
available. The fallback path keeps the demo runnable even without API access.
"""

from __future__ import annotations

import os

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None
    ChatOpenAI = None


class LlmGateway:
    """Wraps model calls so the rest of the agent stays provider-agnostic."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.chat_model = self._build_chat_model()

    def _build_chat_model(self):
        """Create the LangChain chat model when configuration is available."""
        if ChatOpenAI is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.model_name, temperature=0)

    def can_call_model(self) -> bool:
        """Return True when an external LLM call is available."""
        return self.chat_model is not None and ChatPromptTemplate is not None

    def generate_answer(self, system_text: str, user_text: str) -> str:
        """Generate a grounded answer using LangChain when possible."""
        if not self.can_call_model():
            raise RuntimeError("LLM access is not configured")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_text}"),
                ("human", "{user_text}"),
            ]
        )
        chain = prompt | self.chat_model
        response = chain.invoke(
            {
                "system_text": system_text,
                "user_text": user_text,
            }
        )
        return response.content


class DeterministicDemoResponder:
    """Builds a local fallback answer when no LLM credentials are configured."""

    def build_understanding_answer(self, query_text: str, retrieved_matches) -> str:
        """Return a short grounded answer for understanding queries."""
        if not retrieved_matches:
            return f"I could not ground an answer for: {query_text}"
        top_match = retrieved_matches[0]
        symbol_text = ", ".join(top_match.chunk.symbols) if top_match.chunk.symbols else "no obvious symbols"
        return (
            f"Best match: {top_match.chunk.path}:{top_match.chunk.start_line}-{top_match.chunk.end_line}. "
            f"Relevant symbols: {symbol_text}. "
            f"This is the deterministic fallback because no LLM API key is configured."
        )

    def build_execution_answer(self, command_results, error_chunk, failure_analysis=None, execution_notes=None) -> str:
        """Return a readable local answer for build and test execution queries."""
        report_lines = []
        if failure_analysis is not None and failure_analysis.failure_kind:
            report_lines.append(f"Failure classification: {failure_analysis.failure_kind}")
        if execution_notes:
            for note in execution_notes:
                report_lines.append(note)
        for result in command_results:
            report_lines.append(f"{result.get_command_text()} -> exit {result.return_code}")
            combined_output = result.get_combined_output().strip()
            if combined_output:
                report_lines.append(self._summarize_output(combined_output, 300))
        if error_chunk is not None:
            report_lines.append(
                f"Likely relevant code: {error_chunk.path}:{error_chunk.start_line}-{error_chunk.end_line}"
            )
        if failure_analysis is not None and failure_analysis.stop_reason:
            report_lines.append(f"Stop reason: {failure_analysis.stop_reason}")
        if not report_lines:
            report_lines.append("No commands were executed.")
        return "\n".join(report_lines)

    def _summarize_output(self, output_text: str, limit: int) -> str:
        """Shorten long command output for deterministic local reports."""
        if len(output_text) <= limit:
            return output_text
        first_half = output_text[: limit // 2]
        second_half = output_text[-limit // 2 :]
        return f"{first_half}\n...\n{second_half}"
