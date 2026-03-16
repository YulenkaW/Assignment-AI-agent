"""Planning layer for retrieval, execution, and stop policy decisions."""

from __future__ import annotations

from .contracts import AgentPlan, PlanStep, QueryRequest, RouteDecision


class PlanningLayer:
    """Convert routing output into an explicit high-level plan."""

    def create_plan(self, request: QueryRequest, route_decision: RouteDecision) -> AgentPlan:
        """Create the ordered plan for the query."""
        steps = self._build_steps(route_decision)
        stop_conditions = [
            "enough evidence found",
            "build or test result explained",
            "token budget getting tight",
            "max iterations reached",
        ]
        return AgentPlan(route_decision, request.query_text, stop_conditions, steps)

    def _build_steps(self, route_decision: RouteDecision) -> list[PlanStep]:
        """Create the visible planner steps."""
        steps = []
        if route_decision.preferred_flow == "retrieval_first":
            steps.extend(self._build_retrieval_steps("plan retrieval before loading repository context", "retrieve repository evidence first"))
        if route_decision.needs_execution:
            steps.extend(self._build_execution_steps())
        if route_decision.preferred_flow == "execution_first" and route_decision.needs_retrieval:
            steps.extend(self._build_retrieval_steps("plan targeted retrieval from execution evidence", "retrieve file regions referenced by execution evidence"))
        steps.append(PlanStep("reasoning", "reason", "synthesize the final grounded explanation"))
        steps.append(PlanStep("response", "respond", "format the final answer for the user"))
        return steps

    def _build_retrieval_steps(self, planning_text: str, retrieval_text: str) -> list[PlanStep]:
        """Build the paired retrieval-planning and retrieval steps."""
        return [
            PlanStep("retrieval_planning", "plan", planning_text),
            PlanStep("retrieval", "retrieve", retrieval_text),
        ]

    def _build_execution_steps(self) -> list[PlanStep]:
        """Build the shared execution-related steps."""
        return [
            PlanStep("build", "execute", "run the build path when required"),
            PlanStep("test", "execute", "run the test path when required"),
            PlanStep("output_parse", "parse", "parse command outputs into deterministic evidence"),
            PlanStep("analysis", "analyze", "analyze parsed output and locate likely root cause"),
        ]
