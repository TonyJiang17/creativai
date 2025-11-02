#!/usr/bin/env python3
"""Unit tests for TextLLMJudge class."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from eval.text_llm_judge import TextLLMJudge


def mock_model_runner_json(answer: bool, reason: str) -> Any:
    """Create a mock model runner that returns JSON with specified answer and reason."""

    def runner(prompt: str, **kwargs: Any) -> str:
        return json.dumps({"answer": answer, "reason": reason})

    return runner


def mock_model_runner_raw(response: str) -> Any:
    """Create a mock model runner that returns a raw string response."""

    def runner(prompt: str, **kwargs: Any) -> str:
        return response

    return runner


class TestTextLLMJudgeInstantiation:
    """Test TextLLMJudge constructor and initialization."""

    def test_instantiation_with_question_only(self) -> None:
        """Test creating judge with only a question."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
        )

        assert judge.model_runner is runner
        assert judge.question == question
        assert judge.true_example is None
        assert judge.false_example is None
        assert judge.model_kwargs == {}

    def test_instantiation_with_question_and_examples(self) -> None:
        """Test creating judge with question and both examples."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        true_ex = "The stars danced in the velvet sky."
        false_ex = "The sky was dark at night."

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            true_example=true_ex,
            false_example=false_ex,
        )

        assert judge.model_runner is runner
        assert judge.question == question
        assert judge.true_example == true_ex
        assert judge.false_example == false_ex
        assert judge.model_kwargs == {}

    def test_instantiation_with_model_kwargs(self) -> None:
        """Test creating judge with model kwargs."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        kwargs = {"temperature": 0.7, "max_tokens": 100}

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            model_kwargs=kwargs,
        )

        assert judge.model_kwargs == kwargs


class TestBuildPrompt:
    """Test the build_prompt method."""

    def test_build_prompt_question_only(self) -> None:
        """Test prompt building with only a question."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        text = "Sample text to evaluate."

        judge = TextLLMJudge(model_runner=runner, question=question)
        prompt = judge.build_prompt(text)

        assert "Question: Is this text creative?" in prompt
        assert "Text to evaluate:" in prompt
        assert "Sample text to evaluate." in prompt
        assert "Please respond with JSON containing:" in prompt
        assert '"answer": true or false' in prompt
        assert '"reason": brief explanation' in prompt
        assert "Example where the answer is TRUE:" not in prompt
        assert "Example where the answer is FALSE:" not in prompt

    def test_build_prompt_with_true_example(self) -> None:
        """Test prompt building with true example."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        true_ex = "The stars danced."
        text = "Sample text."

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            true_example=true_ex,
        )
        prompt = judge.build_prompt(text)

        assert "Question: Is this text creative?" in prompt
        assert "Example where the answer is TRUE:" in prompt
        assert "The stars danced." in prompt
        assert "Text to evaluate:" in prompt
        assert "Sample text." in prompt
        assert "Example where the answer is FALSE:" not in prompt

    def test_build_prompt_with_false_example(self) -> None:
        """Test prompt building with false example."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        false_ex = "The sky was dark."
        text = "Sample text."

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            false_example=false_ex,
        )
        prompt = judge.build_prompt(text)

        assert "Question: Is this text creative?" in prompt
        assert "Example where the answer is FALSE:" in prompt
        assert "The sky was dark." in prompt
        assert "Text to evaluate:" in prompt
        assert "Sample text." in prompt
        assert "Example where the answer is TRUE:" not in prompt

    def test_build_prompt_with_both_examples(self) -> None:
        """Test prompt building with both examples."""
        runner = mock_model_runner_json(True, "test")
        question = "Is this text creative?"
        true_ex = "The stars danced."
        false_ex = "The sky was dark."
        text = "Sample text."

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            true_example=true_ex,
            false_example=false_ex,
        )
        prompt = judge.build_prompt(text)

        assert "Question: Is this text creative?" in prompt
        assert "Example where the answer is TRUE:" in prompt
        assert "The stars danced." in prompt
        assert "Example where the answer is FALSE:" in prompt
        assert "The sky was dark." in prompt
        assert "Text to evaluate:" in prompt
        assert "Sample text." in prompt


class TestParseResponse:
    """Test the parse_response method."""

    def test_parse_response_answer_true(self) -> None:
        """Test parsing valid JSON with answer=true."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"answer": True, "reason": "It has creative metaphors."})
        result = judge.parse_response(response)

        assert result["answer"] is True
        assert result["reason"] == "It has creative metaphors."

    def test_parse_response_answer_false(self) -> None:
        """Test parsing valid JSON with answer=false."""
        runner = mock_model_runner_json(False, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"answer": False, "reason": "It lacks creativity."})
        result = judge.parse_response(response)

        assert result["answer"] is False
        assert result["reason"] == "It lacks creativity."

    def test_parse_response_malformed_json(self) -> None:
        """Test that malformed JSON raises ValueError."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            judge.parse_response("This is not valid JSON {")

    def test_parse_response_missing_answer_field(self) -> None:
        """Test that missing answer field raises ValueError."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"reason": "Some reason"})
        with pytest.raises(ValueError, match="Response missing 'answer' field"):
            judge.parse_response(response)

    def test_parse_response_missing_reason_field(self) -> None:
        """Test that missing reason field raises ValueError."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"answer": True})
        with pytest.raises(ValueError, match="Response missing 'reason' field"):
            judge.parse_response(response)

    def test_parse_response_answer_not_bool(self) -> None:
        """Test that non-boolean answer raises TypeError."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"answer": "yes", "reason": "Some reason"})
        with pytest.raises(TypeError, match="Expected 'answer' to be bool"):
            judge.parse_response(response)

    def test_parse_response_reason_not_string(self) -> None:
        """Test that non-string reason raises TypeError."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        response = json.dumps({"answer": True, "reason": 123})
        with pytest.raises(TypeError, match="Expected 'reason' to be str"):
            judge.parse_response(response)


class TestEvaluate:
    """Test the evaluate method end-to-end."""

    def test_evaluate_end_to_end_true(self) -> None:
        """Test full evaluation workflow returning true."""
        runner = mock_model_runner_json(True, "Contains creative metaphors.")
        question = "Is this text creative?"

        judge = TextLLMJudge(model_runner=runner, question=question)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The moon whispered secrets to the ocean.")
            temp_path = Path(f.name)

        try:
            result = judge.evaluate(temp_path)

            assert result["source"] == str(temp_path)
            assert "Question: Is this text creative?" in result["prompt"]
            assert "The moon whispered secrets to the ocean." in result["prompt"]
            assert result["raw_response"] == json.dumps(
                {"answer": True, "reason": "Contains creative metaphors."}
            )
            assert result["result"]["answer"] is True
            assert result["result"]["reason"] == "Contains creative metaphors."
            assert result["metadata"]["question"] == question
        finally:
            temp_path.unlink()

    def test_evaluate_end_to_end_false(self) -> None:
        """Test full evaluation workflow returning false."""
        runner = mock_model_runner_json(False, "Lacks creative elements.")
        question = "Is this text creative?"

        judge = TextLLMJudge(model_runner=runner, question=question)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The car was parked.")
            temp_path = Path(f.name)

        try:
            result = judge.evaluate(temp_path)

            assert result["source"] == str(temp_path)
            assert result["result"]["answer"] is False
            assert result["result"]["reason"] == "Lacks creative elements."
            assert result["metadata"]["question"] == question
        finally:
            temp_path.unlink()

    def test_evaluate_with_examples(self) -> None:
        """Test evaluation with both examples provided."""
        runner = mock_model_runner_json(True, "Uses metaphor like the true example.")
        question = "Is this text creative?"
        true_ex = "The stars danced."
        false_ex = "The sky was dark."

        judge = TextLLMJudge(
            model_runner=runner,
            question=question,
            true_example=true_ex,
            false_example=false_ex,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The clouds sang a melody.")
            temp_path = Path(f.name)

        try:
            result = judge.evaluate(temp_path)

            assert "Example where the answer is TRUE:" in result["prompt"]
            assert "The stars danced." in result["prompt"]
            assert "Example where the answer is FALSE:" in result["prompt"]
            assert "The sky was dark." in result["prompt"]
            assert result["result"]["answer"] is True
        finally:
            temp_path.unlink()

    def test_evaluate_with_model_kwargs(self) -> None:
        """Test that model_kwargs are passed to model_runner."""
        received_kwargs: dict[str, Any] = {}

        def capturing_runner(prompt: str, **kwargs: Any) -> str:
            received_kwargs.update(kwargs)
            return json.dumps({"answer": True, "reason": "Test"})

        judge = TextLLMJudge(
            model_runner=capturing_runner,
            question="Test?",
            model_kwargs={"temperature": 0.5, "max_tokens": 50},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test text.")
            temp_path = Path(f.name)

        try:
            judge.evaluate(temp_path)
            assert received_kwargs["temperature"] == 0.5
            assert received_kwargs["max_tokens"] == 50
        finally:
            temp_path.unlink()

    def test_evaluate_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            judge.evaluate(Path("/nonexistent/file.txt"))

    def test_evaluate_directory_error(self) -> None:
        """Test that IsADirectoryError is raised for directory path."""
        runner = mock_model_runner_json(True, "test")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(IsADirectoryError, match="Expected a file path, got directory"):
                judge.evaluate(Path(tmpdir))

    def test_evaluate_malformed_response(self) -> None:
        """Test that ValueError is raised for malformed LLM response."""
        runner = mock_model_runner_raw("This is not JSON")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test text.")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to parse JSON response"):
                judge.evaluate(temp_path)
        finally:
            temp_path.unlink()
