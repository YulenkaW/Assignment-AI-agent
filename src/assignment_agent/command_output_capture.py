"""Command output capture for build and test execution."""

from __future__ import annotations

from .contracts import CapturedCommandOutput, ExecutionBatch


class CommandOutputCapture:
    """Normalize execution results before parsing."""

    def capture_batch(self, execution_batch: ExecutionBatch) -> list[CapturedCommandOutput]:
        """Capture the execution batch into parser-friendly records."""
        captured_outputs = []
        for result in execution_batch.results:
            captured_outputs.append(
                CapturedCommandOutput(
                    result.get_command_text(),
                    result.exit_code,
                    result.stdout_text,
                    result.stderr_text,
                    result.get_combined_output(),
                )
            )
        return captured_outputs
