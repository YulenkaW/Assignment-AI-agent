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
            steps.append(PlanStep("retrieval_planning", "plan", "plan retrieval before loading repository context"))
            steps.append(PlanStep("retrieval", "retrieve", "retrieve repository evidence first"))
        if route_decision.needs_execution:
            steps.append(PlanStep("build", "execute", "run the build path when required"))
            steps.append(PlanStep("test", "execute", "run the test path when required"))
            steps.append(PlanStep("output_parse", "parse", "parse command outputs into deterministic evidence"))
            steps.append(PlanStep("analysis", "analyze", "analyze parsed output and locate likely root cause"))
        if route_decision.preferred_flow == "execution_first" and route_decision.needs_retrieval:
            steps.append(PlanStep("retrieval_planning", "plan", "plan targeted retrieval from execution evidence"))
            steps.append(PlanStep("retrieval", "retrieve", "retrieve file regions referenced by execution evidence"))
        steps.append(PlanStep("reasoning", "reason", "synthesize the final grounded explanation"))
        steps.append(PlanStep("response", "respond", "format the final answer for the user"))
        return steps
