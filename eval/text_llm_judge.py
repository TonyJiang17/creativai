#!/usr/bin/env python3
"""Abstractions for text-based LLM-as-judge evaluations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["ModelRunner", "TextLLMJudge"]

ModelRunner = Callable[..., str]


@dataclass
class TextLLMJudge(ABC):
    """Reusable base class for LLM-as-judge evaluations over text input."""

    model_runner: ModelRunner
    prompt_template: str = ""
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, text_path: Path) -> dict[str, Any]:
        """Run the judge on the given text file and return structured metrics."""

        if not text_path.exists():
            raise FileNotFoundError(f"Input file {text_path} does not exist")
        if text_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got directory {text_path}")

        text = text_path.read_text(encoding="utf-8")
        prompt = self.build_prompt(text=text, source=text_path)
        raw_response = self.model_runner(prompt, **self.model_kwargs)
        parsed_result = self.parse_response(raw_response)

        return {
            "source": str(text_path),
            "prompt": prompt,
            "raw_response": raw_response,
            "result": parsed_result,
            "metadata": dict(self.metadata),
        }

    def build_prompt(self, *, text: str, source: Path) -> str:
        """Format the prompt supplied to the LLM."""

        return self.prompt_template.format(text=text, source=source)

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Convert the raw LLM response into a structured result."""
