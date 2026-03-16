"""Model-assisted action orchestration with deterministic validation downstream."""

from __future__ import annotations

import logging
import os

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None
    ChatOpenAI = None

from .contracts import ActionProposal, AnalysisReport, QueryDiagnostics, RouteDecision


class ToolOrchestrator:
    """Ask the model to propose the next action from an allowed set.

    The orchestrator never executes tools directly. The controller remains the final
    authority and validates every proposed action against constraints.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.chat_model = self._build_chat_model()

    def propose_action(
        self,
        query_text: str,
        route_decision: RouteDecision,
        allowed_actions: list[str],
        analysis_report: AnalysisReport | None,
        diagnostics: QueryDiagnostics | None = None,
    ) -> ActionProposal:
        """Return the model-proposed next action or a deterministic fallback."""
        if self._can_call_model() and allowed_actions:
            if diagnostics is not None:
                diagnostics.orchestrator_attempts += 1
            try:
                action_name = self._propose_with_model(query_text, route_decision, allowed_actions, analysis_report)
                if action_name in allowed_actions:
                    if diagnostics is not None:
                        diagnostics.orchestrator_successes += 1
                    return ActionProposal(action_name, "model-selected allowed action", "model")
            except Exception as error:
                self.logger.warning("Falling back after model action proposal failed: %s", error)
                self.chat_model = None
                if diagnostics is not None:
                    diagnostics.orchestrator_fallbacks += 1
                    diagnostics.add_fallback("tool_orchestrator", str(error))
        if allowed_actions:
            return ActionProposal(allowed_actions[0], "deterministic fallback action", "controller")
        return ActionProposal("stop", "no valid remaining actions", "controller")

    def _build_chat_model(self):
        """Create the OpenAI-backed chat model when configuration exists."""
        if ChatOpenAI is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.model_name, temperature=0)

    def _can_call_model(self) -> bool:
        """Return True when model calls are available."""
        return self.chat_model is not None and ChatPromptTemplate is not None

    def _propose_with_model(
        self,
        query_text: str,
        route_decision: RouteDecision,
        allowed_actions: list[str],
        analysis_report: AnalysisReport | None,
    ) -> str:
        """Ask the model to choose one allowed action."""
        analysis_text = ""
        if analysis_report is not None:
            analysis_text = analysis_report.recommended_next_action
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Choose exactly one action from the allowed list. Respond with only the action name.",
                ),
                (
                    "human",
                    "Query: {query}\nTask Type: {task_type}\nAllowed Actions: {actions}\nAnalysis Hint: {analysis_hint}",
                ),
            ]
        )
        chain = prompt | self.chat_model
        response = chain.invoke(
            {
                "query": query_text,
                "task_type": route_decision.task_type,
                "actions": ", ".join(allowed_actions),
                "analysis_hint": analysis_text,
            }
        )
        return response.content.strip()
