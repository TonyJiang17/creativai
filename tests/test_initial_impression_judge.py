#!/usr/bin/env python3
"""Tests for InitialImpressionJudge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.initial_impression_judge import InitialImpressionJudge


def mock_runner(prompt: str, **kwargs: Any) -> str:
    """Mock model runner that returns test JSON."""
    return json.dumps({
        "who": "A young detective",
        "where": "A foggy London street",
        "when": "Victorian era"
    })


def mock_runner_invalid_json(prompt: str, **kwargs: Any) -> str:
    """Mock model runner that returns invalid JSON."""
    return "This is not JSON"


def mock_runner_missing_field(prompt: str, **kwargs: Any) -> str:
    """Mock model runner that returns JSON missing a required field."""
    return json.dumps({
        "who": "A young detective",
        "where": "A foggy London street"
        # missing "when"
    })


class TestInitialImpressionJudge:
    """Test suite for InitialImpressionJudge."""

    def test_parse_response_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        judge = InitialImpressionJudge(model_runner=mock_runner)
        response = json.dumps({
            "who": "A young detective",
            "where": "A foggy London street",
            "when": "Victorian era"
        })
        result = judge.parse_response(response)

        assert isinstance(result, dict)
        assert result["who"] == "A young detective"
        assert result["where"] == "A foggy London street"
        assert result["when"] == "Victorian era"

    def test_parse_response_invalid_json(self) -> None:
        """Test that invalid JSON raises ValueError."""
        judge = InitialImpressionJudge(model_runner=mock_runner)

        with pytest.raises(ValueError, match="Invalid JSON from LLM"):
            judge.parse_response("This is not JSON")

    def test_parse_response_missing_who(self) -> None:
        """Test that missing 'who' field raises ValueError."""
        judge = InitialImpressionJudge(model_runner=mock_runner)
        response = json.dumps({"where": "London", "when": "past"})

        with pytest.raises(ValueError, match="Missing required field 'who'"):
            judge.parse_response(response)

    def test_parse_response_missing_where(self) -> None:
        """Test that missing 'where' field raises ValueError."""
        judge = InitialImpressionJudge(model_runner=mock_runner)
        response = json.dumps({"who": "Detective", "when": "past"})

        with pytest.raises(ValueError, match="Missing required field 'where'"):
            judge.parse_response(response)

    def test_parse_response_missing_when(self) -> None:
        """Test that missing 'when' field raises ValueError."""
        judge = InitialImpressionJudge(model_runner=mock_runner)
        response = json.dumps({"who": "Detective", "where": "London"})

        with pytest.raises(ValueError, match="Missing required field 'when'"):
            judge.parse_response(response)

    def test_evaluate_returns_structured_dict(self, tmp_path: Path) -> None:
        """Test that evaluate() returns properly structured dict."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("The detective walked down the foggy London street.")

        judge = InitialImpressionJudge(model_runner=mock_runner)
        result = judge.evaluate(test_file)

        assert "source" in result
        assert "prompt" in result
        assert "raw_response" in result
        assert "result" in result
        assert "metadata" in result

        assert isinstance(result["result"], dict)
        assert "who" in result["result"]
        assert "where" in result["result"]
        assert "when" in result["result"]

    def test_evaluate_nonexistent_file(self) -> None:
        """Test that evaluate() raises FileNotFoundError for nonexistent file."""
        judge = InitialImpressionJudge(model_runner=mock_runner)

        with pytest.raises(FileNotFoundError):
            judge.evaluate(Path("/nonexistent/file.txt"))

    def test_custom_prompt_template(self) -> None:
        """Test that custom prompt template can be provided."""
        custom_template = "Custom template: {text}"
        judge = InitialImpressionJudge(
            model_runner=mock_runner,
            prompt_template=custom_template
        )

        assert judge.prompt_template == custom_template

    def test_model_kwargs_passed_through(self, tmp_path: Path) -> None:
        """Test that model_kwargs are stored and accessible."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("The detective walked down the foggy London street.")

        model_kwargs = {"temperature": 0.7, "max_tokens": 100}
        judge = InitialImpressionJudge(
            model_runner=mock_runner,
            model_kwargs=model_kwargs
        )

        assert judge.model_kwargs == model_kwargs

    def test_metadata_contains_metric_name(self) -> None:
        """Test that metadata contains the metric name."""
        judge = InitialImpressionJudge(model_runner=mock_runner)

        assert "metric" in judge.metadata
        assert judge.metadata["metric"] == "initial_impression"
