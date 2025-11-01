#!/usr/bin/env python3
"""LLM judge that checks conflict clarity and stakes escalation in a text sample."""

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

PROMPT_TEMPLATE = '''You are an expert fiction editor assessing whether a passage delivers clear conflict and meaningful stakes escalation.

Reference principles:
- Establish one dominant stake early (or a tightly braided pair) and keep pressure coherent under that lens.
- Show at least one meaningful tightening, reversal, or deepening that narrows options or shifts knowledge.
- Agency may be decision, endurance, renunciation, or focused attention, but it must exact an on-page cost.
- Coincidence may complicate the conflict but must not solve it; resolutions should feel earned by the protagonist's choices.
- Stakes must feel concrete and embodied, not abstract labels; we should sense what might be lost.

After reading the text, output strict JSON with this shape (no extra commentary):
{{
  "conflict_and_stakes": true | false,
  "rationale": "one sentence explaining the decision"
}}
Your reply must be only that JSON object. Do not include markdown, code fences, backticks, or any surrounding text.

Evaluate the following passage:
"""{text}"""
'''


class ConflictAndStakesJudge(TextLLMJudge):
    """Judge that returns True when the text shows clear conflict and high stakes"""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        prompt_template: str = PROMPT_TEMPLATE,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            prompt_template=prompt_template,
            model_kwargs=model_kwargs or {},
            metadata={"metric": "conflict_and_stakes", "output": "boolean"},
        )

    def parse_response(self, response: str) -> bool:
        cleaned = response.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {response!r}") from exc

        value = data.get("conflict_and_stakes")
        if isinstance(value, bool):
            return value

        raise ValueError(
            "Unable to parse boolean response from LLM output. "
            f"Got response: {response!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the conflict and stakes judge on a text file.",
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name to use (default: gpt-5-mini)",
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

    judge = ConflictAndStakesJudge(model_runner=runner)
    report = judge.evaluate(args.input)

    print(report["result"])

    if args.show_raw:
        print("\n--- Raw Response ---\n" + report["raw_response"])


if __name__ == "__main__":
    main()
