#!/usr/bin/env python3
"""CLI that runs all available text metrics and judges on a file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeVar

try:  # support running as a module or script
    from .conflict_and_stakes_judge import ConflictAndStakesJudge
    from .flesch_reading_ease import flesch_reading_ease
    from .motivation_consistency_judge import MotivationConsistencyJudge
    from .text_llm_judge import ModelRunner, TextLLMJudge
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.conflict_and_stakes_judge import ConflictAndStakesJudge  # type: ignore
    from eval.flesch_reading_ease import flesch_reading_ease  # type: ignore
    from eval.motivation_consistency_judge import MotivationConsistencyJudge  # type: ignore
    from eval.text_llm_judge import ModelRunner, TextLLMJudge  # type: ignore

from util.openai_runner import OpenAIConfigError, create_openai_runner

MetricResult = dict[str, Any]
JudgeType = TypeVar("JudgeType", bound=TextLLMJudge)


def run_llm_judge(
    judge_cls: type[JudgeType],
    *,
    text_path: Path,
    runner: ModelRunner,
) -> MetricResult:
    judge = judge_cls(model_runner=runner)
    report = judge.evaluate(text_path)
    return {
        "result": report["result"],
        "raw_response": report["raw_response"],
        "metadata": report["metadata"],
    }


def compute_aggregate_score(results: dict[str, MetricResult]) -> Any:
    """Placeholder for later aggregate scoring logic."""

    return None


def run_all_metrics(text_path: Path) -> dict[str, Any]:
    if not text_path.exists():
        raise FileNotFoundError(f"Input file '{text_path}' does not exist.")
    if text_path.is_dir():
        raise IsADirectoryError("Please supply a file path, not a directory.")

    text = text_path.read_text(encoding="utf-8")

    runner = create_openai_runner()

    results: dict[str, MetricResult] = {}
    results["flesch_reading_ease"] = flesch_reading_ease(text)
    results["motivation_consistency"] = run_llm_judge(
        MotivationConsistencyJudge,
        text_path=text_path,
        runner=runner,
    )
    results["conflict_and_stakes"] = run_llm_judge(
        ConflictAndStakesJudge,
        text_path=text_path,
        runner=runner,
    )

    aggregate = compute_aggregate_score(results)

    return {
        "source": str(text_path),
        "metrics": results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configured metrics against a text file.")
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    parser.add_argument("--output", type=Path, help="Optional path for JSON output")
    args = parser.parse_args()

    try:
        report = run_all_metrics(args.input)
    except OpenAIConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
