#!/usr/bin/env python3
"""CLI that runs JSON-driven LLM judges on text files.

This module loads judge configurations from llm_judge_prompts.json, creates
TextLLMJudge instances for each question, and evaluates text files against
all configured judges. Results are output as structured JSON.

The judge configurations define questions and optional example texts that
guide the LLM evaluation. Each judge returns a binary answer (true/false)
with reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:  # support running as a module or script
    from .text_llm_judge import ModelRunner, TextLLMJudge
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.text_llm_judge import ModelRunner, TextLLMJudge  # type: ignore

import os
from dotenv import load_dotenv
load_dotenv()

from util.openai_runner import OpenAIConfigError, create_openai_runner


def load_judge_configs(config_path: Path) -> list[dict[str, Any]]:
    """
    Load judge configurations from JSON file and resolve example file contents.

    Reads the judge configuration JSON, validates structure, and loads any
    example files referenced in the configuration. Example file paths are
    converted to string content in the returned configs.

    Args:
        config_path: Path to the llm_judge_prompts.json configuration file

    Returns:
        List of config dicts with example file contents loaded. Each dict contains:
            - question: The evaluation question (required)
            - true_example: Content of true_example_file if specified (optional)
            - false_example: Content of false_example_file if specified (optional)

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If config structure is invalid (not a list or missing question)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        config_text = config_path.read_text(encoding="utf-8")
        configs = json.loads(config_text)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in config file {config_path}: {e.msg}",
            e.doc,
            e.pos
        )

    if not isinstance(configs, list):
        raise ValueError(f"Config file must contain a JSON array, got {type(configs).__name__}")

    # Load example files and convert paths to content
    base_dir = config_path.parent.parent  # Go up to project root
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError(f"Each config entry must be a dict, got {type(config).__name__}")

        if "question" not in config:
            raise ValueError(f"Config entry missing required 'question' field: {config}")

        # Load true example if specified
        if "true_example_file" in config:
            true_path = base_dir / config["true_example_file"]
            if true_path.exists():
                config["true_example"] = true_path.read_text(encoding="utf-8")
            else:
                # Log warning but continue - example is optional
                print(f"Warning: true_example_file not found: {true_path}")
                config["true_example"] = None
            # Remove file path key
            del config["true_example_file"]

        # Load false example if specified
        if "false_example_file" in config:
            false_path = base_dir / config["false_example_file"]
            if false_path.exists():
                config["false_example"] = false_path.read_text(encoding="utf-8")
            else:
                # Log warning but continue - example is optional
                print(f"Warning: false_example_file not found: {false_path}")
                config["false_example"] = None
            # Remove file path key
            del config["false_example_file"]

    return configs


def create_judges(configs: list[dict[str, Any]], runner: ModelRunner) -> list[TextLLMJudge]:
    """
    Create TextLLMJudge instances from configuration list.

    Args:
        configs: List of judge config dicts with question and optional examples
        runner: Model runner callable to pass to each judge

    Returns:
        List of initialized TextLLMJudge instances, one per config
    """
    judges: list[TextLLMJudge] = []
    for config in configs:
        judge = TextLLMJudge(
            model_runner=runner,
            question=config["question"],
            true_example=config.get("true_example"),
            false_example=config.get("false_example"),
        )
        judges.append(judge)
    return judges


def run_all_judges(text_path: Path, judges: list[TextLLMJudge]) -> dict[str, Any]:
    """
    Run all judges on a single text file and return structured results.

    Args:
        text_path: Path to the text file to evaluate
        judges: List of TextLLMJudge instances to run

    Returns:
        Dict with structure:
            {
                "source": str(text_path),
                "score": float,
                "judges": {
                    "judge_0": {
                        "question": str,
                        "answer": bool,
                        "reason": str,
                        "raw_response": str
                    },
                    ...
                }
            }
    """
    result: dict[str, Any] = {
        "source": str(text_path),
        "judges": {}
    }

    true_count = 0
    for index, judge in enumerate(judges):
        evaluation = judge.evaluate(text_path)
        answer = evaluation["result"]["answer"]
        result["judges"][f"judge_{index}"] = {
            "question": judge.question,
            "answer": answer,
            "reason": evaluation["result"]["reason"],
            "raw_response": evaluation["raw_response"]
        }
        if answer:
            true_count += 1

    # Calculate score as (number of true answers) / (total judges)
    total_judges = len(judges)
    result["score"] = true_count / total_judges if total_judges > 0 else 0.0

    return result


def _iter_text_files(directory: Path) -> Iterable[Path]:
    """Iterate over .txt files in a directory in sorted order."""
    for path in sorted(directory.glob("*.txt")):
        if path.is_file():
            yield path


def main() -> None:
    """CLI entry point for running judges on text files."""
    parser = argparse.ArgumentParser(description="Run configured LLM judges against text files.")
    parser.add_argument("input", type=Path, help="Path to the input .txt file or directory of .txt files")
    parser.add_argument(
        "--batch-output-file-name",
        type=str,
        help="When input is a directory, required name (without extension) for the JSON report saved in eval/results/.",
    )
    args = parser.parse_args()

    target = args.input

    # Load judge configurations
    config_path = Path(__file__).resolve().parent / "llm_judge_prompts.json"
    configs = load_judge_configs(config_path)

    # Create model runner and judges
    runner = create_openai_runner()
    judges = create_judges(configs, runner)

    # Handle directory batch processing
    if target.is_dir():
        if not args.batch_output_file_name:
            raise SystemExit("--batch-output-file-name is required when input is a directory")

        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{args.batch_output_file_name}.json"

        aggregate_report: dict[str, Any] = {
            "source_directory": str(target),
            "files": {},
        }

        files = list(_iter_text_files(target))
        total = len(files)

        for index, text_file in enumerate(files, start=1):
            print(f"Processing {index}/{total}: {text_file}")
            aggregate_report["files"][str(text_file)] = run_all_judges(text_file, judges)

        output_path.write_text(json.dumps(aggregate_report, indent=2), encoding="utf-8")
        print(f"Wrote batch results to {output_path}")
        return

    # Handle single file processing
    try:
        report = run_all_judges(target, judges)
    except OpenAIConfigError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


"""
Usage examples:

  - Single file: python eval/main.py texts/weak_hook.txt
  - Directory: python eval/main.py texts/sample_aitah --batch-output-file-name sample_run
    (saves to eval/results/sample_run.json)
"""
