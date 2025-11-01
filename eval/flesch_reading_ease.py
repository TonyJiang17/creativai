#!/usr/bin/env python3
'''Compute the Flesch Reading Ease Score for a text file.'''

import argparse
import re
from pathlib import Path

SENTENCE_END_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[A-Za-z']+")


def count_sentences(text: str) -> int:
    sentences = [segment.strip() for segment in SENTENCE_END_RE.split(text) if segment.strip()]
    return len(sentences)


def extract_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0

    vowels = "aeiouy"
    syllables = 0
    prev_was_vowel = False

    for char in cleaned:
        if char in vowels:
            if not prev_was_vowel:
                syllables += 1
            prev_was_vowel = True
        else:
            prev_was_vowel = False

    if cleaned.endswith("e") and not cleaned.endswith(("le", "ue")) and syllables > 1:
        syllables -= 1

    return syllables if syllables > 0 else 1


def flesch_reading_ease(text: str) -> dict[str, float]:
    words = extract_words(text)
    sentence_count = count_sentences(text)
    total_words = len(words)
    total_sentences = sentence_count if sentence_count > 0 else 1
    total_syllables = sum(count_syllables(word) for word in words)

    if total_words == 0:
        raise ValueError("Unable to compute score: no words were found in the input.")

    score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

    return {
        "score": score,
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_syllables": total_syllables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the Flesch Reading Ease Score for a text file.")
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file '{args.input}' does not exist.")
    if args.input.is_dir():
        raise SystemExit("Please supply a file path, not a directory.")

    text = args.input.read_text(encoding="utf-8")

    try:
        metrics = flesch_reading_ease(text)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Flesch Reading Ease Score: {metrics['score']:.2f}")
    print(f"Total sentences: {metrics['total_sentences']}")
    print(f"Total words: {metrics['total_words']}")
    print(f"Total syllables: {metrics['total_syllables']}")


if __name__ == "__main__":
    main()
