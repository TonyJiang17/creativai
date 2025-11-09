#!/usr/bin/env python3
"""Programmatic evaluation of text strings using all configured judges.

This module provides a function for evaluating text strings (not file paths)
using all judges defined in llm_judge_prompts.json plus the LengthJudge.
Designed for use in generator scripts to evaluate generated text programmatically.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:  # support running as a module or script
    from .text_llm_judge import ModelRunner, TextLLMJudge
    from .length_judge import LengthJudge
    from .main import load_judge_configs, create_judges
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from eval.text_llm_judge import ModelRunner, TextLLMJudge  # type: ignore
    from eval.length_judge import LengthJudge  # type: ignore
    from eval.main import load_judge_configs, create_judges  # type: ignore

from util.openai_runner import create_openai_runner


def _evaluate_single_judge(index: int, judge: TextLLMJudge, text: str) -> tuple[int, dict[str, Any]]:
    """
    Evaluate a single judge on the given text.

    Args:
        index: Judge index for result key
        judge: TextLLMJudge instance to run
        text: Text string to evaluate

    Returns:
        Tuple of (index, result_dict) where result_dict contains:
            - question: str
            - answer: bool
            - reason: str
            - raw_response: str
    """
    import time
    start_time = time.time()
    print(f"[{start_time:.3f}] Starting judge {index}")
    prompt = judge.build_prompt(text)
    raw_response = judge.model_runner(prompt, **judge.model_kwargs)
    parsed = judge.parse_response(raw_response)
    elapsed = time.time() - start_time
    print(f"[{time.time():.3f}] Completed judge {index} in {elapsed:.2f}s")

    return (index, {
        "question": judge.question,
        "answer": parsed["answer"],
        "reason": parsed["reason"],
        "raw_response": raw_response,
    })


def evaluate_text(text: str) -> dict[str, Any]:
    """
    Evaluate text string using all configured judges.

    This function runs all judges defined in llm_judge_prompts.json plus the
    LengthJudge on the provided text string. It returns the same structured
    result format as run_all_judges() in main.py but adapted for text strings
    instead of file paths.

    The model runner is created automatically using create_openai_runner() from util.

    Args:
        text: The text string to evaluate

    Returns:
        Dict with structure:
            {
                "score": float,  # (true answers) / (total judges)
                "judges": {
                    "judge_0": {
                        "question": str,
                        "answer": bool,
                        "reason": str,
                        "raw_response": str
                    },
                    "judge_1": {...},
                    ...
                    "length_judge": {...}
                }
            }

    Raises:
        FileNotFoundError: If judge config file doesn't exist
        ValueError: If judge config is invalid or LLM response cannot be parsed
        json.JSONDecodeError: If judge config is not valid JSON

    Example:
        >>> text = "AITA for refusing to attend my sister's wedding?..."
        >>> result = evaluate_text(text)
        >>> print(f"Score: {result['score']}")
        >>> for judge_name, judge_result in result['judges'].items():
        ...     print(f"{judge_name}: {judge_result['answer']}")
    """
    # Create model runner
    model_runner = create_openai_runner()

    # Load judge configurations
    config_path = Path(__file__).resolve().parent / "llm_judge_prompts.json"
    configs = load_judge_configs(config_path)

    # Create judges
    judges = create_judges(configs, model_runner)

    # Initialize result structure
    result: dict[str, Any] = {"judges": {}}

    # Run all LLM judges in parallel
    true_count = 0
    with ThreadPoolExecutor() as executor:
        # Submit all judge evaluations
        futures = {
            executor.submit(_evaluate_single_judge, index, judge, text): index
            for index, judge in enumerate(judges)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            index, judge_result = future.result()
            answer = judge_result["answer"]
            result["judges"][f"judge_{index}"] = judge_result
            if answer:
                true_count += 1

    # Run length judge (adapted for text string instead of file)
    length_judge = LengthJudge()
    word_count = len(text.split())
    length_answer = word_count > length_judge.words

    result["judges"]["length_judge"] = {
        "question": f"Is length of text greater than {length_judge.words} words?",
        "answer": length_answer,
        "reason": "",
        "raw_response": str(length_answer),
    }
    if length_answer:
        true_count += 1

    # Calculate score as (number of true answers) / (total judges)
    total_judges = len(judges) + 1
    result["score"] = true_count / total_judges if total_judges > 0 else 0.0

    return result
