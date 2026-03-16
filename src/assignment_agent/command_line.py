"""CLI entry point for the separate assignment agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent_controller import AssignmentAgentController


class AssignmentAgentCli:
    """Parse CLI arguments and run the separate agent implementation."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the command-line parser."""
        parser = argparse.ArgumentParser(description="Assignment-ready local repository agent")
        parser.add_argument("--repo", required=True, help="Path to the target repository")
        parser.add_argument("--query", required=True, help="Question or build/test request")
        parser.add_argument("--model", default="gpt-4.1-mini", help="Optional OpenAI model name")
        parser.add_argument("--max-total-tokens", type=int, default=5000, help="Hard context limit")
        parser.add_argument("--show-prompt-report", action="store_true", help="Print the prompt assembly report")
        return parser

    def run(self) -> None:
        """Run the command-line workflow."""
        parser = self.build_parser()
        arguments = parser.parse_args()
        controller = AssignmentAgentController(
            repository_path=Path(arguments.repo),
            model_name=arguments.model,
            max_total_tokens=arguments.max_total_tokens,
        )
        response = controller.answer_query(arguments.query)
        print(response.answer_text)
        if arguments.show_prompt_report and response.prompt_report is not None:
            print("\n--- Prompt Report ---")
            print(response.prompt_report.render_text())


def main() -> None:
    """Run the packaged CLI entry point."""
    AssignmentAgentCli().run()


if __name__ == "__main__":
    main()
