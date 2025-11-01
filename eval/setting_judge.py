#!/usr/bin/env python3
"""Orchestrator that coordinates InitialImpressionJudge and ConsistencyVerificationJudge."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

try:  # support running as a script or module
    from .initial_impression_judge import InitialImpressionJudge
    from .consistency_verification_judge import ConsistencyVerificationJudge
    from .text_llm_judge import ModelRunner
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.initial_impression_judge import InitialImpressionJudge  # type: ignore
    from eval.consistency_verification_judge import ConsistencyVerificationJudge  # type: ignore
    from eval.text_llm_judge import ModelRunner  # type: ignore

from util.openai_runner import OpenAIConfigError, create_openai_runner


class SettingJudge:
    """Orchestrator that evaluates setting consistency using two-stage analysis.

    This class coordinates InitialImpressionJudge (extracts who/where/when from
    opening) and ConsistencyVerificationJudge (verifies consistency with full text).
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SettingJudge orchestrator.

        Args:
            model_runner: Callable that takes a prompt and returns LLM response
            model_kwargs: Optional dict of additional kwargs to pass to model_runner
        """
        self.model_runner = model_runner
        self.model_kwargs = model_kwargs or {}

        # Create both judge instances
        self.initial_judge = InitialImpressionJudge(
            model_runner=model_runner,
            model_kwargs=self.model_kwargs,
        )
        self.consistency_judge = ConsistencyVerificationJudge(
            model_runner=model_runner,
            model_kwargs=self.model_kwargs,
        )

    def _extract_first_50_words(self, text: str) -> str:
        """Extract the first 50 words from text.

        Args:
            text: The full text content

        Returns:
            String containing the first 50 words (or fewer if text is shorter)
        """
        words = text.split()[:50]
        return " ".join(words)

    def evaluate(self, text_source: Path | str) -> dict[str, Any]:
        """Run the two-stage evaluation on the given text.

        Args:
            text_source: Either a Path to a text file or a string containing the text

        Returns:
            Dict with source, first_50_words, initial_impression,
            consistency_verification, and result fields

        Raises:
            FileNotFoundError: If text_source is a Path that doesn't exist
            IsADirectoryError: If text_source is a Path to a directory
        """
        # Handle file path vs string input
        if isinstance(text_source, Path):
            if not text_source.exists():
                raise FileNotFoundError(f"Input file {text_source} does not exist")
            if text_source.is_dir():
                raise IsADirectoryError(
                    f"Expected a file path, got directory {text_source}"
                )
            text = text_source.read_text(encoding="utf-8")
            source = str(text_source)
            text_path = text_source
        else:
            # text_source is a string
            text = text_source
            source = "<string>"
            # For ConsistencyVerificationJudge, we need a file path
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            temp_file.write(text)
            temp_file.close()
            text_path = Path(temp_file.name)

        try:
            # Step 1: Extract first 50 words
            first_50_words = self._extract_first_50_words(text)

            # Step 2: Call InitialImpressionJudge with first 50 words
            # We need to create a temporary file for InitialImpressionJudge as well
            temp_initial_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            temp_initial_file.write(first_50_words)
            temp_initial_file.close()
            temp_initial_path = Path(temp_initial_file.name)

            try:
                initial_result = self.initial_judge.evaluate(temp_initial_path)
            finally:
                # Clean up temporary file
                temp_initial_path.unlink()

            # Step 3: Call ConsistencyVerificationJudge with full text and initial answers
            consistency_result = self.consistency_judge.evaluate(
                text_path=text_path,
                initial_answers=initial_result["result"],
            )

            # Step 4: Aggregate results
            return {
                "source": source,
                "first_50_words": first_50_words,
                "initial_impression": {
                    "prompt": initial_result["prompt"],
                    "raw_response": initial_result["raw_response"],
                    "result": initial_result["result"],
                },
                "consistency_verification": {
                    "prompt": consistency_result["prompt"],
                    "raw_response": consistency_result["raw_response"],
                    "result": consistency_result["result"],
                },
                "result": consistency_result["result"]["overall_rating"],
            }
        finally:
            # Clean up temporary file if we created one for string input
            if source == "<string>":
                text_path.unlink()


def main() -> None:
    """CLI entry point for running the setting judge orchestrator."""
    parser = argparse.ArgumentParser(
        description="Evaluate setting consistency using two-stage analysis.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Path to the input .txt file")
    group.add_argument("--text", type=str, help="Text string to evaluate")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature to forward to the OpenAI API.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the raw LLM responses.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON.",
    )
    args = parser.parse_args()

    default_kwargs: dict[str, Any] = {}
    if args.temperature is not None:
        default_kwargs["temperature"] = args.temperature

    try:
        runner = create_openai_runner(
            model=args.model,
            default_kwargs=default_kwargs,
        )
    except OpenAIConfigError as exc:
        raise SystemExit(str(exc)) from exc

    judge = SettingJudge(model_runner=runner, model_kwargs=default_kwargs)

    # Determine input source
    text_source: Path | str
    if args.file:
        text_source = args.file
    else:
        text_source = args.text

    result = judge.evaluate(text_source)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Print human-readable output
        print("Setting Judge Evaluation Results:")
        print(f"Source: {result['source']}")
        print(f"\nFirst 50 words: {result['first_50_words'][:100]}...")

        print("\nInitial Impression:")
        initial = result["initial_impression"]["result"]
        print(f"  Who: {initial['who']}")
        print(f"  Where: {initial['where']}")
        print(f"  When: {initial['when']}")

        print("\nConsistency Verification:")
        consistency = result["consistency_verification"]["result"]
        print(f"  Who match: {consistency['who_match']}")
        print(f"  Where match: {consistency['where_match']}")
        print(f"  When match: {consistency['when_match']}")
        print(f"  Overall rating: {consistency['overall_rating']}")
        print(f"  Explanation: {consistency['explanation']}")

        print(f"\nFinal Rating: {result['result']}")

        if args.show_raw:
            print("\n--- Raw Initial Impression Response ---")
            print(result["initial_impression"]["raw_response"])
            print("\n--- Raw Consistency Verification Response ---")
            print(result["consistency_verification"]["raw_response"])


if __name__ == "__main__":
    main()
