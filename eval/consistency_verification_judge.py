#!/usr/bin/env python3
"""LLM judge that verifies initial impressions match the full text."""

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

PROMPT_TEMPLATE = '''You are evaluating whether the opening of a story accurately represents the complete story.

INITIAL IMPRESSIONS FROM FIRST 50 WORDS:
- Who: {who_answer}
- Where: {where_answer}
- When: {when_answer}

COMPLETE STORY:
{full_text}

Task: Determine if the initial impressions match the complete story.

For each element (who, where, when), assess:
- "yes" if the initial impression matches the full story
- "no" if contradicted or misaligned
- "unclear" if the opening didn't establish it clearly

Then provide an overall rating:
- "complete": All three elements match
- "partial": Mixed results (some match, some don't)
- "none": Initial impressions are largely contradicted

Respond in strict JSON format:
{{
  "who_match": "yes/no/unclear",
  "where_match": "yes/no/unclear",
  "when_match": "yes/no/unclear",
  "overall_rating": "complete/partial/none",
  "explanation": "Brief explanation of your reasoning"
}}
Your reply must be only that JSON object. Do not include markdown, code fences, backticks, or any surrounding text.
'''


class ConsistencyVerificationJudge(TextLLMJudge):
    """Judge that verifies initial impressions against the full text.

    This judge evaluates whether initial answers about who/where/when from
    the opening of a story are consistent with the complete text.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        prompt_template: str = PROMPT_TEMPLATE,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the consistency verification judge.

        Args:
            model_runner: Callable that takes a prompt and returns LLM response
            prompt_template: Template string for the evaluation prompt
            model_kwargs: Optional kwargs to pass to model_runner
        """
        super().__init__(
            model_runner=model_runner,
            prompt_template=prompt_template,
            model_kwargs=model_kwargs or {},
            metadata={
                "metric": "consistency_verification",
                "output": "dict[str, str]"
            },
        )

    def evaluate(
        self,
        text_path: Path,
        initial_answers: dict[str, str],
    ) -> dict[str, Any]:
        """Run the judge on the given text file with initial answers.

        Args:
            text_path: Path to the text file to evaluate
            initial_answers: Dict with "who", "where", "when" keys

        Returns:
            Dict containing source, prompt, raw_response, result, and metadata

        Raises:
            FileNotFoundError: If text_path does not exist
            IsADirectoryError: If text_path is a directory
        """
        if not text_path.exists():
            raise FileNotFoundError(f"Input file {text_path} does not exist")
        if text_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got directory {text_path}")

        text = text_path.read_text(encoding="utf-8")
        prompt = self.build_prompt(
            text=text,
            source=text_path,
            initial_answers=initial_answers
        )
        raw_response = self.model_runner(prompt, **self.model_kwargs)
        parsed_result = self.parse_response(raw_response)

        return {
            "source": str(text_path),
            "prompt": prompt,
            "raw_response": raw_response,
            "result": parsed_result,
            "metadata": dict(self.metadata),
        }

    def build_prompt(
        self,
        *,
        text: str,
        source: Path,
        initial_answers: dict[str, str],
    ) -> str:
        """Format the prompt with text and initial answers.

        Args:
            text: The full text content to evaluate
            source: Path to the source file (for reference)
            initial_answers: Dict with "who", "where", "when" keys

        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            full_text=text,
            who_answer=initial_answers.get("who", "Not specified"),
            where_answer=initial_answers.get("where", "Not specified"),
            when_answer=initial_answers.get("when", "Not specified"),
        )

    def parse_response(self, response: str) -> dict[str, str]:
        """Parse the LLM response into structured consistency data.

        Args:
            response: Raw LLM response string

        Returns:
            Dict with who_match, where_match, when_match, overall_rating, explanation

        Raises:
            ValueError: If JSON is invalid or fields are missing/invalid
        """
        cleaned = response.strip()

        # Remove markdown code fences if present
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

        # Check required fields
        required_fields = ["who_match", "where_match", "when_match", "overall_rating", "explanation"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in response: {response!r}")

        # Validate match values
        valid_matches = {"yes", "no", "unclear"}
        for field in ["who_match", "where_match", "when_match"]:
            if data[field] not in valid_matches:
                raise ValueError(
                    f"{field} must be one of {valid_matches}, got {data[field]!r}"
                )

        # Validate overall rating
        valid_ratings = {"complete", "partial", "none"}
        if data["overall_rating"] not in valid_ratings:
            raise ValueError(
                f"overall_rating must be one of {valid_ratings}, got {data['overall_rating']!r}"
            )

        return {
            "who_match": data["who_match"],
            "where_match": data["where_match"],
            "when_match": data["when_match"],
            "overall_rating": data["overall_rating"],
            "explanation": data["explanation"],
        }


def main() -> None:
    """CLI entry point for consistency verification judge."""
    parser = argparse.ArgumentParser(
        description="Verify initial impressions against full story text.",
    )
    parser.add_argument("--file", type=Path, required=True, help="Path to the story text file")
    parser.add_argument("--who", required=True, help="Initial impression of 'who'")
    parser.add_argument("--where", required=True, help="Initial impression of 'where'")
    parser.add_argument("--when", required=True, help="Initial impression of 'when'")
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

    judge = ConsistencyVerificationJudge(model_runner=runner)

    initial_answers = {
        "who": args.who,
        "where": args.where,
        "when": args.when,
    }

    report = judge.evaluate(text_path=args.file, initial_answers=initial_answers)

    result = report["result"]
    print("Consistency Verification Results:")
    print(f"  Who match: {result['who_match']}")
    print(f"  Where match: {result['where_match']}")
    print(f"  When match: {result['when_match']}")
    print(f"  Overall rating: {result['overall_rating']}")
    print(f"  Explanation: {result['explanation']}")

    if args.show_raw:
        print("\n--- Raw Response ---\n" + report["raw_response"])


if __name__ == "__main__":
    main()
