#!/usr/bin/env python3
"""Concrete JSON-driven LLM-as-judge evaluation for text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

__all__ = ["ModelRunner", "TextLLMJudge"]

ModelRunner = Callable[..., str]


class TextLLMJudge:
    """
    Concrete JSON-driven LLM judge that evaluates text based on a question.

    This class provides a flexible framework for binary (true/false) evaluations
    of text using an LLM. It accepts a question and optional examples to guide
    the evaluation, then returns a structured result with the answer and reasoning.

    The judge constructs prompts with the question, optional examples showing
    what true and false answers look like, and the text to evaluate. The LLM
    responds with JSON containing a boolean answer and explanation.

    Example usage:
        >>> from util.openai_runner import create_openai_runner
        >>> runner = create_openai_runner(model="gpt-4")
        >>> judge = TextLLMJudge(
        ...     model_runner=runner,
        ...     question="Does this text contain creative metaphors?",
        ...     true_example="The moon whispered secrets to the restless ocean.",
        ...     false_example="The car was parked in the driveway."
        ... )
        >>> result = judge.evaluate(Path("sample.txt"))
        >>> print(result["result"]["answer"])  # True or False
        >>> print(result["result"]["reason"])  # Explanation

    Args:
        model_runner: Callable that takes a prompt string and returns LLM response.
                     Should accept **kwargs for model configuration.
        question: The yes/no question to ask about the text being evaluated.
        true_example: Optional example text where the answer should be TRUE.
        false_example: Optional example text where the answer should be FALSE.
        model_kwargs: Optional dict of kwargs to pass to model_runner (e.g., temperature).
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        question: str,
        true_example: str | None = None,
        false_example: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the TextLLMJudge.

        Args:
            model_runner: Callable that takes a prompt and returns LLM response
            question: The question to ask about the text
            true_example: Optional example where answer is TRUE
            false_example: Optional example where answer is FALSE
            model_kwargs: Optional kwargs to pass to model_runner
        """
        self.model_runner = model_runner
        self.question = question
        self.true_example = true_example
        self.false_example = false_example
        self.model_kwargs = model_kwargs or {}

    def build_prompt(self, text: str) -> str:
        """
        Construct the evaluation prompt with question, examples, and text.

        Args:
            text: The text content to evaluate

        Returns:
            Formatted prompt string to send to the LLM
        """
        sections = [f"Question: {self.question}", ""]

        if self.true_example is not None:
            sections.append("Example where the answer is TRUE:")
            sections.append(self.true_example)
            sections.append("")

        if self.false_example is not None:
            sections.append("Example where the answer is FALSE:")
            sections.append(self.false_example)
            sections.append("")

        sections.append("Text to evaluate:")
        sections.append(text)
        sections.append("")
        sections.append("Please respond with JSON containing:")
        sections.append('- "answer": true or false')
        sections.append('- "reason": brief explanation')

        return "\n".join(sections)

    def parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse the JSON response from the LLM.

        Args:
            response: Raw string response from the LLM

        Returns:
            Dict with 'answer' (bool) and 'reason' (str)

        Raises:
            ValueError: If JSON is malformed or missing required fields
            TypeError: If answer is not bool or reason is not str
        """
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response was: {response[:200]}")

        if "answer" not in parsed:
            raise ValueError(f"Response missing 'answer' field. Got: {parsed}")

        if "reason" not in parsed:
            raise ValueError(f"Response missing 'reason' field. Got: {parsed}")

        answer = parsed["answer"]
        reason = parsed["reason"]

        if not isinstance(answer, bool):
            raise TypeError(f"Expected 'answer' to be bool, got {type(answer).__name__}: {answer}")

        if not isinstance(reason, str):
            raise TypeError(f"Expected 'reason' to be str, got {type(reason).__name__}: {reason}")

        return {
            "answer": answer,
            "reason": reason,
        }

    def evaluate(self, text_path: Path) -> dict[str, Any]:
        """
        Run the judge on the given text file and return structured results.

        Args:
            text_path: Path to the text file to evaluate

        Returns:
            Dict containing:
                - source: Path to the evaluated file
                - prompt: Full prompt sent to the LLM
                - raw_response: Raw response from the LLM
                - result: Parsed result dict with 'answer' and 'reason'
                - metadata: Additional info including the question

        Raises:
            FileNotFoundError: If text_path doesn't exist
            IsADirectoryError: If text_path is a directory
            ValueError: If LLM response cannot be parsed
        """
        if not text_path.exists():
            raise FileNotFoundError(f"Input file {text_path} does not exist")
        if text_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got directory {text_path}")

        text = text_path.read_text(encoding="utf-8")
        prompt = self.build_prompt(text)
        raw_response = self.model_runner(prompt, **self.model_kwargs)
        result = self.parse_response(raw_response)

        return {
            "source": str(text_path),
            "prompt": prompt,
            "raw_response": raw_response,
            "result": result,
            "metadata": {
                "question": self.question,
            },
        }
