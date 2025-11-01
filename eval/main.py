#!/usr/bin/env python3
"""CLI that runs all available text metrics and judges on a file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

try:  # support running as a module or script
    from .conflict_and_stakes_judge import ConflictAndStakesJudge
    from .flesch_reading_ease import flesch_reading_ease
    from .motivation_consistency_judge import MotivationConsistencyJudge
    from .text_llm_judge import ModelRunner, TextLLMJudge
    from .surprise import surprise_score
    from .five_second_judge import FiveSecondJudge
    from .setting_judge import SettingJudge
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.conflict_and_stakes_judge import ConflictAndStakesJudge  # type: ignore
    from eval.flesch_reading_ease import flesch_reading_ease  # type: ignore
    from eval.motivation_consistency_judge import MotivationConsistencyJudge  # type: ignore
    from eval.text_llm_judge import ModelRunner, TextLLMJudge  # type: ignore
    from eval.surprise import surprise_score  # type: ignore
    from eval.five_second_judge import FiveSecondJudge
    from eval.setting_judge import SettingJudge

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


def run_all_metrics(text_path: Path, *, runner: ModelRunner | None = None) -> dict[str, Any]:
    if not text_path.exists():
        raise FileNotFoundError(f"Input file '{text_path}' does not exist.")
    if text_path.is_dir():
        raise IsADirectoryError("Please supply a file path, not a directory.")

    text = text_path.read_text(encoding="utf-8")

    llm_runner = runner or create_openai_runner()

    results: dict[str, MetricResult] = {}
    results["flesch_reading_ease"] = flesch_reading_ease(text)
    results["motivation_consistency"] = run_llm_judge(
        MotivationConsistencyJudge,
        text_path=text_path,
        runner=llm_runner,
    )
    results["conflict_and_stakes"] = run_llm_judge(
        ConflictAndStakesJudge,
        text_path=text_path,
        runner=llm_runner,
    )
    results["surprise"] = surprise_score(text)
    results["five_second"] = run_llm_judge(
        FiveSecondJudge,
        text_path=text_path,
        runner=llm_runner
    )
    results["setting"] = SettingJudge(model_runner=llm_runner).evaluate(text)

    aggregate = compute_aggregate_score(results)

    return {
        "source": str(text_path),
        "metrics": results,
        "aggregate": aggregate,
    }


def _iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.txt")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured metrics against text files.")
    parser.add_argument("input", type=Path, help="Path to the input .txt file or directory of .txt files")
    parser.add_argument(
        "--batch-output-file-name",
        type=str,
        help="When input is a directory, required name (without extension) for the JSON report saved in eval/results/.",
    )
    args = parser.parse_args()

    target = args.input

    if target.is_dir():
        if not args.batch_output_file_name:
            raise SystemExit("--batch-output-file-name is required when input is a directory")

        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{args.batch_output_file_name}.json"

        runner = create_openai_runner()

        aggregate_report: dict[str, Any] = {
            "source_directory": str(target),
            "files": {},
        }

        for text_file in _iter_text_files(target):
            aggregate_report["files"][str(text_file)] = run_all_metrics(
                text_file,
                runner=runner,
            )

        output_path.write_text(json.dumps(aggregate_report, indent=2), encoding="utf-8")
        print(f"Wrote batch results to {output_path}")
        return

    try:
        report = run_all_metrics(target)
    except OpenAIConfigError as exc:
        raise SystemExit(str(exc)) from exc
    breakpoint()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


""" 
Usage examples:

  - Single file: python eval/main.py texts/sample_aitah/s1.txt
  - Directory: python eval/main.py texts/sample_aitah --batch-output-file-name sample_run (saves to eval/results/sample_run.json)
"""
