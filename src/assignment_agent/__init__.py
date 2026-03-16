"""Separate assignment-ready agent package.

This package is intentionally independent from the existing `context_agent`
implementation so the original code path remains unchanged.
"""

from .agent_controller import AssignmentAgentController

__all__ = ["AssignmentAgentController"]
