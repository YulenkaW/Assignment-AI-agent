"""Command-line entry point for the demo agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .main_agent import ContextAwareCodebaseAgent


class CommandLineApplication:
    """Parses command-line arguments and runs the agent."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Create the CLI parser for local demo use."""
        parser = argparse.ArgumentParser(description="Context-constrained local C++ codebase agent")
        parser.add_argument("--repo", required=True, help="Path to a local C++ repository")
        parser.add_argument("--query", required=True, help="User query to answer or execute")
        parser.add_argument("--model", default=None, help="Optional model name")
        parser.add_argument("--max-total-tokens", type=int, default=5000, help="Hard per-call context budget")
        parser.add_argument("--show-prompt-report", action="store_true", help="Print the prompt assembly report after the answer")
        return parser

    def run(self) -> None:
        """Run the command-line application."""
        parser = self.build_parser()
        arguments = parser.parse_args()
        agent = ContextAwareCodebaseAgent(
            repository_path=Path(arguments.repo),
            model_name=arguments.model,
            max_total_tokens=arguments.max_total_tokens,
        )
        answer = agent.answer_query(arguments.query)
        print(answer.answer_text)

        if arguments.show_prompt_report and answer.prompt_report is not None:
            print("\n--- Prompt Report ---")
            print(answer.prompt_report.render_text())


def main() -> None:
    """Run the packaged CLI entry point."""
    CommandLineApplication().run()


if __name__ == "__main__":
    main()
