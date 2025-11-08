#!/usr/bin/env python3
"""Length-based text evaluator.

Evaluates whether a text file exceeds a specified word count threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class LengthJudge:
    def __init__(self, words=200):
        self.words = words

    def evaluate(self, text_path):
        if not text_path.exists():
            raise FileNotFoundError(f"Input file {text_path} does not exist")
        if text_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got directory {text_path}")

        text = text_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        result = word_count > self.words

        return {
            "source": str(text_path),
            "prompt": f"Is length of text greater than {self.words} words?",
            "raw_response": result,
            "result": result,
            "word_count": word_count,
            "metadata": {
                "question": f"Is length of text greater than {self.words} words?",
                "word_count": word_count,
                "threshold": self.words,
            },
        }


def main() -> None:
    """CLI entry point for length evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate whether a text file exceeds a word count threshold."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input .txt file to evaluate"
    )
    parser.add_argument(
        "--words",
        type=int,
        default=200,
        help="Word count threshold (default: 200)"
    )
    args = parser.parse_args()

    judge = LengthJudge(words=args.words)
    result = judge.evaluate(args.input)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
