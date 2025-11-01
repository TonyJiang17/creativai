#!/usr/bin/env python3
"""LLM judge that extracts who/where/when from opening text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # support running as a script or module
    from .text_llm_judge import ModelRunner, TextLLMJudge
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.text_llm_judge import ModelRunner, TextLLMJudge  # type: ignore

from util.openai_runner import OpenAIConfigError, create_openai_runner

PROMPT_TEMPLATE = '''You are evaluating the opening of a creative text. Read ONLY the first 50 words below and answer three questions based on what these opening words suggest.

FIRST 50 WORDS:
{text}

Answer these questions based ONLY on what you just read:
1. Who is this story about? (Describe the main character(s) or subject)
2. Where is this story set? (Describe the location)
3. When did this story happen relative to now? (e.g., "past", "present", "future", "historical period", etc.)

Respond in JSON format:
{{
  "who": "your answer",
  "where": "your answer",
  "when": "your answer"
}}
'''


class InitialImpressionJudge(TextLLMJudge):
    """Judge that extracts who/where/when from the first 50 words of text."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        prompt_template: str = PROMPT_TEMPLATE,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the InitialImpressionJudge.

        Args:
            model_runner: Callable that takes a prompt and returns LLM response
            prompt_template: Template string with {text} placeholder
            model_kwargs: Optional dict of additional kwargs to pass to model_runner
        """
        super().__init__(
            model_runner=model_runner,
            prompt_template=prompt_template,
            model_kwargs=model_kwargs or {},
            metadata={"metric": "initial_impression", "output": "dict"},
        )

    def build_prompt(self, *, text: str, source: Path) -> str:
        """Format the prompt with first 50 words of text.

        Args:
            text: Full text content
            source: Path to the source file (unused but required by interface)

        Returns:
            Formatted prompt string with first 50 words
        """
        words = text.split()[:50]
        first_50_words = " ".join(words)
        return self.prompt_template.format(text=first_50_words)

    def parse_response(self, response: str) -> dict[str, str]:
        """Parse the LLM response into structured who/where/when dict.

        Args:
            response: Raw response string from the LLM

        Returns:
            Dict with 'who', 'where', and 'when' keys

        Raises:
            ValueError: If response is not valid JSON or missing required fields
        """
        cleaned = response.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {response!r}") from exc

        # Validate required fields
        required_fields = ["who", "where", "when"]
        for field in required_fields:
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' in LLM response. "
                    f"Got response: {response!r}"
                )

        return {
            "who": str(data["who"]),
            "where": str(data["where"]),
            "when": str(data["when"]),
        }


def main() -> None:
    """CLI entry point for running the initial impression judge."""
    parser = argparse.ArgumentParser(
        description="Extract who/where/when from the opening of a text file.",
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
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
        help="Print the raw LLM response.",
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

    judge = InitialImpressionJudge(model_runner=runner)
    report = judge.evaluate(args.input)

    result = report["result"]
    print(f"Who: {result['who']}")
    print(f"Where: {result['where']}")
    print(f"When: {result['when']}")

    if args.show_raw:
        print("\n--- Raw Response ---\n" + report["raw_response"])


if __name__ == "__main__":
    main()
