#!/usr/bin/env python3
"""Tests for SettingJudge orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.setting_judge import SettingJudge


def mock_runner_initial(prompt: str, **kwargs: Any) -> str:
    """Mock model runner for InitialImpressionJudge."""
    return json.dumps({
        "who": "A young detective",
        "where": "A foggy London street",
        "when": "Victorian era"
    })


def mock_runner_consistency(prompt: str, **kwargs: Any) -> str:
    """Mock model runner for ConsistencyVerificationJudge."""
    return json.dumps({
        "who_match": "yes",
        "where_match": "yes",
        "when_match": "yes",
        "overall_rating": "complete",
        "explanation": "All initial impressions match the full text."
    })


class MockRunnerOrchestrated:
    """Mock runner that switches behavior based on prompt content."""

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Return appropriate response based on prompt."""
        # Check for consistency prompt first (more specific)
        if "INITIAL IMPRESSIONS FROM FIRST 50 WORDS:" in prompt:
            return mock_runner_consistency(prompt, **kwargs)
        elif "FIRST 50 WORDS:" in prompt:
            return mock_runner_initial(prompt, **kwargs)
        else:
            raise ValueError(f"Unexpected prompt: {prompt[:100]}")


class TestSettingJudge:
    """Test suite for SettingJudge orchestrator."""

    def test_init_creates_judge_instances(self) -> None:
        """Test that __init__ creates both judge instances."""
        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)

        assert judge.initial_judge is not None
        assert judge.consistency_judge is not None
        assert judge.model_runner is runner

    def test_extract_first_50_words(self) -> None:
        """Test that first 50 words are extracted correctly."""
        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)

        text = " ".join([f"word{i}" for i in range(100)])
        first_50 = judge._extract_first_50_words(text)

        words = first_50.split()
        assert len(words) == 50
        assert words[0] == "word0"
        assert words[49] == "word49"

    def test_extract_first_50_words_fewer_than_50(self) -> None:
        """Test extraction when text has fewer than 50 words."""
        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)

        text = " ".join([f"word{i}" for i in range(30)])
        first_50 = judge._extract_first_50_words(text)

        words = first_50.split()
        assert len(words) == 30
        assert first_50 == text

    def test_evaluate_with_file_path(self, tmp_path: Path) -> None:
        """Test evaluate() with a file path."""
        test_file = tmp_path / "test.txt"
        test_text = " ".join([f"word{i}" for i in range(100)])
        test_file.write_text(test_text)

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_file)

        assert "source" in result
        assert str(test_file) in result["source"]
        assert "first_50_words" in result
        assert "initial_impression" in result
        assert "consistency_verification" in result
        assert "result" in result

    def test_evaluate_with_text_string(self) -> None:
        """Test evaluate() with a text string."""
        test_text = " ".join([f"word{i}" for i in range(100)])

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_text)

        assert result["source"] == "<string>"
        assert "first_50_words" in result
        assert "initial_impression" in result
        assert "consistency_verification" in result
        assert "result" in result

    def test_evaluate_structure_matches_spec(self, tmp_path: Path) -> None:
        """Test that evaluate() returns dict matching the spec structure."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("The detective walked down the foggy London street.")

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_file)

        # Check top-level keys
        assert "source" in result
        assert "first_50_words" in result
        assert "initial_impression" in result
        assert "consistency_verification" in result
        assert "result" in result

        # Check initial_impression structure
        assert "prompt" in result["initial_impression"]
        assert "raw_response" in result["initial_impression"]
        assert "result" in result["initial_impression"]
        assert "who" in result["initial_impression"]["result"]
        assert "where" in result["initial_impression"]["result"]
        assert "when" in result["initial_impression"]["result"]

        # Check consistency_verification structure
        assert "prompt" in result["consistency_verification"]
        assert "raw_response" in result["consistency_verification"]
        assert "result" in result["consistency_verification"]
        assert "who_match" in result["consistency_verification"]["result"]
        assert "where_match" in result["consistency_verification"]["result"]
        assert "when_match" in result["consistency_verification"]["result"]
        assert "overall_rating" in result["consistency_verification"]["result"]
        assert "explanation" in result["consistency_verification"]["result"]

        # Check result field is the final rating
        assert result["result"] == "complete"

    def test_evaluate_result_is_overall_rating(self, tmp_path: Path) -> None:
        """Test that result field contains the overall_rating from consistency judge."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("The detective walked down the foggy London street.")

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_file)

        assert result["result"] == result["consistency_verification"]["result"]["overall_rating"]

    def test_evaluate_nonexistent_file_raises_error(self) -> None:
        """Test that evaluate() raises FileNotFoundError for nonexistent file."""
        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)

        with pytest.raises(FileNotFoundError):
            judge.evaluate(Path("/nonexistent/file.txt"))

    def test_first_50_words_extracted_correctly(self, tmp_path: Path) -> None:
        """Test that first 50 words are extracted and stored."""
        test_file = tmp_path / "test.txt"
        test_text = " ".join([f"word{i}" for i in range(100)])
        test_file.write_text(test_text)

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_file)

        first_50_words = result["first_50_words"]
        words = first_50_words.split()
        assert len(words) == 50
        assert words[0] == "word0"
        assert words[49] == "word49"

    def test_model_kwargs_passed_to_judges(self) -> None:
        """Test that model_kwargs are passed to both judges."""
        runner = MockRunnerOrchestrated()
        model_kwargs = {"temperature": 0.7, "max_tokens": 100}

        judge = SettingJudge(model_runner=runner, model_kwargs=model_kwargs)

        assert judge.initial_judge.model_kwargs == model_kwargs
        assert judge.consistency_judge.model_kwargs == model_kwargs

    def test_evaluate_passes_initial_answers_to_consistency(self, tmp_path: Path) -> None:
        """Test that initial answers are passed to consistency judge."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("The detective walked down the foggy London street.")

        runner = MockRunnerOrchestrated()
        judge = SettingJudge(model_runner=runner)
        result = judge.evaluate(test_file)

        # Verify that consistency verification received the initial answers
        consistency_prompt = result["consistency_verification"]["prompt"]
        assert "A young detective" in consistency_prompt
        assert "A foggy London street" in consistency_prompt
        assert "Victorian era" in consistency_prompt
