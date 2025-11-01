#!/usr/bin/env python3
"""Tests for ConsistencyVerificationJudge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.consistency_verification_judge import ConsistencyVerificationJudge


def mock_runner(prompt: str, **kwargs: Any) -> str:
    """Mock model runner that returns valid JSON response."""
    return json.dumps({
        "who_match": "yes",
        "where_match": "no",
        "when_match": "unclear",
        "overall_rating": "partial",
        "explanation": "The character matches but location changed."
    })


def test_consistency_verification_judge_initialization() -> None:
    """Test that judge can be initialized with model runner."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)
    assert judge.model_runner == mock_runner
    assert "who_match" in judge.prompt_template
    assert judge.metadata["metric"] == "consistency_verification"


def test_evaluate_returns_structured_dict(tmp_path: Path) -> None:
    """Test that evaluate returns complete structured response."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    # Create temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Sample story text for testing.")

    initial_answers = {
        "who": "A young detective",
        "where": "A dark alley",
        "when": "Late at night"
    }

    result = judge.evaluate(text_path=test_file, initial_answers=initial_answers)

    # Verify structure
    assert "source" in result
    assert "prompt" in result
    assert "raw_response" in result
    assert "result" in result
    assert "metadata" in result

    # Verify result structure
    parsed = result["result"]
    assert "who_match" in parsed
    assert "where_match" in parsed
    assert "when_match" in parsed
    assert "overall_rating" in parsed
    assert "explanation" in parsed


def test_evaluate_includes_initial_answers_in_prompt(tmp_path: Path) -> None:
    """Test that initial answers are injected into prompt."""
    captured_prompt: list[str] = []

    def capturing_runner(prompt: str, **kwargs: Any) -> str:
        captured_prompt.append(prompt)
        return mock_runner(prompt, **kwargs)

    judge = ConsistencyVerificationJudge(model_runner=capturing_runner)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Full story text.")

    initial_answers = {
        "who": "A young detective",
        "where": "A dark alley",
        "when": "Late at night"
    }

    judge.evaluate(text_path=test_file, initial_answers=initial_answers)

    assert len(captured_prompt) == 1
    prompt = captured_prompt[0]
    assert "A young detective" in prompt
    assert "A dark alley" in prompt
    assert "Late at night" in prompt
    assert "Full story text." in prompt


def test_parse_response_valid_json() -> None:
    """Test parsing valid JSON response."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    response = json.dumps({
        "who_match": "yes",
        "where_match": "no",
        "when_match": "unclear",
        "overall_rating": "partial",
        "explanation": "Some match, some don't."
    })

    parsed = judge.parse_response(response)

    assert parsed["who_match"] == "yes"
    assert parsed["where_match"] == "no"
    assert parsed["when_match"] == "unclear"
    assert parsed["overall_rating"] == "partial"
    assert parsed["explanation"] == "Some match, some don't."


def test_parse_response_invalid_json() -> None:
    """Test that invalid JSON raises ValueError."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    with pytest.raises(ValueError, match="Invalid JSON"):
        judge.parse_response("not valid json")


def test_parse_response_missing_field() -> None:
    """Test that missing required field raises ValueError."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    incomplete_response = json.dumps({
        "who_match": "yes",
        "where_match": "no",
        # Missing when_match, overall_rating, explanation
    })

    with pytest.raises(ValueError, match="Missing required field"):
        judge.parse_response(incomplete_response)


def test_parse_response_invalid_who_match_value() -> None:
    """Test that invalid who_match value raises ValueError."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    response = json.dumps({
        "who_match": "invalid",  # Should be yes/no/unclear
        "where_match": "no",
        "when_match": "unclear",
        "overall_rating": "partial",
        "explanation": "Test"
    })

    with pytest.raises(ValueError, match="who_match"):
        judge.parse_response(response)


def test_parse_response_invalid_overall_rating() -> None:
    """Test that invalid overall_rating raises ValueError."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    response = json.dumps({
        "who_match": "yes",
        "where_match": "no",
        "when_match": "unclear",
        "overall_rating": "invalid",  # Should be complete/partial/none
        "explanation": "Test"
    })

    with pytest.raises(ValueError, match="overall_rating"):
        judge.parse_response(response)


def test_evaluate_file_not_found() -> None:
    """Test that evaluate raises FileNotFoundError for missing file."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    with pytest.raises(FileNotFoundError):
        judge.evaluate(
            text_path=Path("/nonexistent/file.txt"),
            initial_answers={"who": "test", "where": "test", "when": "test"}
        )


def test_all_valid_match_values() -> None:
    """Test all valid match values are accepted."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    for match_value in ["yes", "no", "unclear"]:
        response = json.dumps({
            "who_match": match_value,
            "where_match": match_value,
            "when_match": match_value,
            "overall_rating": "complete",
            "explanation": "Test"
        })
        parsed = judge.parse_response(response)
        assert parsed["who_match"] == match_value


def test_all_valid_rating_values() -> None:
    """Test all valid overall_rating values are accepted."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    for rating in ["complete", "partial", "none"]:
        response = json.dumps({
            "who_match": "yes",
            "where_match": "yes",
            "when_match": "yes",
            "overall_rating": rating,
            "explanation": "Test"
        })
        parsed = judge.parse_response(response)
        assert parsed["overall_rating"] == rating


def test_parse_response_with_markdown_code_fences() -> None:
    """Test that markdown code fences are stripped from response."""
    judge = ConsistencyVerificationJudge(model_runner=mock_runner)

    # Response with markdown code fences
    response = """```json
{
  "who_match": "yes",
  "where_match": "no",
  "when_match": "unclear",
  "overall_rating": "partial",
  "explanation": "Test explanation"
}
```"""

    parsed = judge.parse_response(response)
    assert parsed["who_match"] == "yes"
    assert parsed["where_match"] == "no"
    assert parsed["when_match"] == "unclear"
    assert parsed["overall_rating"] == "partial"
    assert parsed["explanation"] == "Test explanation"
