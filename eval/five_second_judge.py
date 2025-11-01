#!/usr/bin/env python3
"""LLM judge that evaluates whether story openings contain compelling emotional hooks."""

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

PROMPT_TEMPLATE = '''You are an expert fiction editor evaluating whether a story opening contains a compelling emotional hook within the first ~5 seconds of reading.

A compelling hook should demonstrate several of these qualities:
- Emotional intensity: Creates strong feelings (anger, sympathy, shock, curiosity, fear)
- Gossip-worthiness: Presents a naturally discussable situation that people would want to talk about
- Clear conflict: Shows an obvious problem, tension, or injustice
- Relatable themes: Involves universal experiences like relationships, justice/injustice, embarrassment, betrayal, family dynamics, or workplace situations
- Character positioning: Establishes the character as either wronged/righteous OR in a ridiculous/embarrassing situation
- Immediate engagement: Hooks the reader within the first 5 seconds (opening sentences)
- Cliffhanger: Ends on a suspenseful note that makes readers want to continue

Evaluate whether this opening has a compelling hook. Output strict JSON with this exact shape (no extra commentary):
{{
  "has_compelling_hook": true | false,
  "rationale": "one sentence explaining the decision"
}}

Your reply must be only that JSON object. Do not include markdown, code fences, backticks, or any surrounding text.

Evaluate the following passage:
"""{text}"""
'''


class FiveSecondJudge(TextLLMJudge):
    """Judge that returns True when the text contains a compelling emotional hook in the opening."""

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
            metadata={"metric": "five_second_hook", "output": "boolean"},
        )

    def parse_response(self, response: str) -> bool:
        cleaned = response.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {response!r}") from exc

        value = data.get("has_compelling_hook")
        if isinstance(value, bool):
            return value

        raise ValueError(
            "Unable to parse boolean response from LLM output. "
            f"Got response: {response!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five-second hook judge on a text file.",
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

    judge = FiveSecondJudge(model_runner=runner)
    report = judge.evaluate(args.input)

    print(report["result"])

    if args.show_raw:
        print("\n--- Raw Response ---\n" + report["raw_response"])


if __name__ == "__main__":
    main()