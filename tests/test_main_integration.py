#!/usr/bin/env python3
"""Integration tests for the JSON-driven judge system."""

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from eval.main import create_judges, load_judge_configs, run_all_judges
from eval.text_llm_judge import TextLLMJudge


class TestLoadJudgeConfigs:
    """Test config loading functionality."""

    def test_load_judge_configs_valid(self, tmp_path: Path) -> None:
        """Test loading valid config with one question."""
        config = [{"question": "Is this creative?"}]
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = load_judge_configs(config_file)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "question" in result[0]
        assert result[0]["question"] == "Is this creative?"

    def test_load_judge_configs_multiple_questions(self, tmp_path: Path) -> None:
        """Test loading config with multiple questions."""
        config = [
            {"question": "Is this creative?"},
            {"question": "Is this compelling?"},
            {"question": "Is this well-formatted?"},
        ]
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = load_judge_configs(config_file)

        assert len(result) == 3
        assert result[0]["question"] == "Is this creative?"
        assert result[1]["question"] == "Is this compelling?"
        assert result[2]["question"] == "Is this well-formatted?"

    def test_load_judge_configs_with_examples(self, tmp_path: Path) -> None:
        """Test config loading with example files."""
        # Create directory structure matching project layout
        # Config will be in eval/, so base_dir will be parent.parent
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        true_example = texts_dir / "true_example.txt"
        true_example.write_text("The stars danced beautifully.", encoding="utf-8")

        false_example = texts_dir / "false_example.txt"
        false_example.write_text("The sky was dark.", encoding="utf-8")

        # Create config in eval/ directory
        config = [
            {
                "question": "Does this have creative metaphors?",
                "true_example_file": "texts/true_example.txt",
                "false_example_file": "texts/false_example.txt",
            }
        ]
        config_file = eval_dir / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = load_judge_configs(config_file)

        assert len(result) == 1
        assert result[0]["question"] == "Does this have creative metaphors?"
        assert result[0]["true_example"] == "The stars danced beautifully."
        assert result[0]["false_example"] == "The sky was dark."
        # Verify file path keys are removed
        assert "true_example_file" not in result[0]
        assert "false_example_file" not in result[0]

    def test_load_judge_configs_with_only_true_example(self, tmp_path: Path) -> None:
        """Test config with only true example file."""
        # Create directory structure matching project layout
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        true_example = texts_dir / "true_example.txt"
        true_example.write_text("Example text.", encoding="utf-8")

        config = [
            {
                "question": "Test question?",
                "true_example_file": "texts/true_example.txt",
            }
        ]
        config_file = eval_dir / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = load_judge_configs(config_file)

        assert result[0]["true_example"] == "Example text."
        assert "false_example" not in result[0]

    def test_load_judge_configs_missing_example_file_warning(
        self, tmp_path: Path, capfd: Any
    ) -> None:
        """Test that missing example files generate warnings but don't fail."""
        config = [
            {
                "question": "Test question?",
                "true_example_file": "texts/nonexistent.txt",
            }
        ]
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = load_judge_configs(config_file)

        # Should still load, but with None for missing example
        assert len(result) == 1
        assert result[0]["true_example"] is None

        # Check warning was printed
        captured = capfd.readouterr()
        assert "Warning: true_example_file not found" in captured.out

    def test_load_judge_configs_missing_config_file(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing config file."""
        config_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_judge_configs(config_file)

    def test_load_judge_configs_invalid_json(self, tmp_path: Path) -> None:
        """Test error handling for invalid JSON."""
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text("This is not valid JSON {", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_judge_configs(config_file)

    def test_load_judge_configs_not_a_list(self, tmp_path: Path) -> None:
        """Test error when config is not a JSON array."""
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps({"question": "Test?"}), encoding="utf-8")

        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_judge_configs(config_file)

    def test_load_judge_configs_missing_question_field(self, tmp_path: Path) -> None:
        """Test error when config entry is missing question."""
        config = [{"answer": "no question here"}]
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="missing required 'question' field"):
            load_judge_configs(config_file)

    def test_load_judge_configs_entry_not_dict(self, tmp_path: Path) -> None:
        """Test error when config entry is not a dict."""
        config = ["This should be a dict"]
        config_file = tmp_path / "llm_judge_prompts.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="Each config entry must be a dict"):
            load_judge_configs(config_file)


class TestCreateJudges:
    """Test judge creation from configs."""

    def test_create_judges_from_configs(self, mock_openai_runner: Callable[..., str]) -> None:
        """Test creating judges from config list."""
        configs = [
            {"question": "Is this creative?"},
            {"question": "Is this compelling?"},
        ]

        judges = create_judges(configs, mock_openai_runner)

        assert len(judges) == 2
        assert all(isinstance(j, TextLLMJudge) for j in judges)
        assert judges[0].question == "Is this creative?"
        assert judges[1].question == "Is this compelling?"
        assert judges[0].model_runner is mock_openai_runner
        assert judges[1].model_runner is mock_openai_runner

    def test_create_judges_with_examples(self, mock_openai_runner: Callable[..., str]) -> None:
        """Test creating judges with example content."""
        configs = [
            {
                "question": "Is this creative?",
                "true_example": "The stars danced.",
                "false_example": "The sky was dark.",
            }
        ]

        judges = create_judges(configs, mock_openai_runner)

        assert len(judges) == 1
        assert judges[0].true_example == "The stars danced."
        assert judges[0].false_example == "The sky was dark."

    def test_create_judges_mixed_with_and_without_examples(
        self, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test creating judges where some have examples and some don't."""
        configs = [
            {"question": "Question 1?", "true_example": "Example 1"},
            {"question": "Question 2?"},
            {"question": "Question 3?", "false_example": "Example 3"},
        ]

        judges = create_judges(configs, mock_openai_runner)

        assert len(judges) == 3
        assert judges[0].true_example == "Example 1"
        assert judges[0].false_example is None
        assert judges[1].true_example is None
        assert judges[1].false_example is None
        assert judges[2].true_example is None
        assert judges[2].false_example == "Example 3"

    def test_create_judges_empty_list(self, mock_openai_runner: Callable[..., str]) -> None:
        """Test creating judges from empty config list."""
        configs: list[dict[str, Any]] = []

        judges = create_judges(configs, mock_openai_runner)

        assert len(judges) == 0
        assert isinstance(judges, list)


class TestRunAllJudges:
    """Test single file evaluation."""

    def test_run_all_judges_single_file(
        self, sample_text_file: Path, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test running all judges on a single file."""
        configs = [
            {"question": "Is this creative?"},
            {"question": "Is this compelling?"},
        ]
        judges = create_judges(configs, mock_openai_runner)

        result = run_all_judges(sample_text_file, judges)

        assert "source" in result
        assert result["source"] == str(sample_text_file)
        assert "judges" in result
        assert isinstance(result["judges"], dict)
        assert len(result["judges"]) == 2
        assert "judge_0" in result["judges"]
        assert "judge_1" in result["judges"]

    def test_run_all_judges_output_format(
        self, sample_text_file: Path, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test output format matches specification exactly."""
        configs = [{"question": "Is this creative?"}]
        judges = create_judges(configs, mock_openai_runner)

        result = run_all_judges(sample_text_file, judges)

        # Check top-level structure
        assert set(result.keys()) == {"source", "judges"}

        # Check judge result structure
        judge_result = result["judges"]["judge_0"]
        assert set(judge_result.keys()) == {"question", "answer", "reason", "raw_response"}

        # Check types
        assert isinstance(judge_result["question"], str)
        assert isinstance(judge_result["answer"], bool)
        assert isinstance(judge_result["reason"], str)
        assert isinstance(judge_result["raw_response"], str)

        # Check values
        assert judge_result["question"] == "Is this creative?"
        assert judge_result["answer"] is True  # From mock
        assert judge_result["reason"] == "Test reason from mock"

    def test_run_all_judges_with_different_answers(
        self, sample_text_file: Path, mock_runner_with_response: Callable[[bool, str], Callable[..., str]]
    ) -> None:
        """Test judges with different answer values."""
        runner_true = mock_runner_with_response(True, "Reason for true")
        runner_false = mock_runner_with_response(False, "Reason for false")

        judge_true = TextLLMJudge(model_runner=runner_true, question="Question 1?")
        judge_false = TextLLMJudge(model_runner=runner_false, question="Question 2?")

        result = run_all_judges(sample_text_file, [judge_true, judge_false])

        assert result["judges"]["judge_0"]["answer"] is True
        assert result["judges"]["judge_0"]["reason"] == "Reason for true"
        assert result["judges"]["judge_1"]["answer"] is False
        assert result["judges"]["judge_1"]["reason"] == "Reason for false"

    def test_run_all_judges_no_judges(self, sample_text_file: Path) -> None:
        """Test running with empty judges list."""
        result = run_all_judges(sample_text_file, [])

        assert result["source"] == str(sample_text_file)
        assert result["judges"] == {}


class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    def test_batch_evaluation_directory(
        self, sample_text_directory: Path, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test batch evaluation on a directory of text files."""
        configs = [{"question": "Is this creative?"}]
        judges = create_judges(configs, mock_openai_runner)

        # Simulate batch processing workflow
        from eval.main import _iter_text_files

        files = list(_iter_text_files(sample_text_directory))
        assert len(files) == 3  # Should find 3 .txt files (not the .md)

        aggregate_report: dict[str, Any] = {
            "source_directory": str(sample_text_directory),
            "files": {},
        }

        for text_file in files:
            aggregate_report["files"][str(text_file)] = run_all_judges(text_file, judges)

        # Verify structure
        assert "source_directory" in aggregate_report
        assert "files" in aggregate_report
        assert len(aggregate_report["files"]) == 3

        # Verify each file entry has correct structure
        for file_path, file_result in aggregate_report["files"].items():
            assert "source" in file_result
            assert "judges" in file_result
            assert "judge_0" in file_result["judges"]

    def test_iter_text_files_ordering(self, sample_text_directory: Path) -> None:
        """Test that text files are returned in sorted order."""
        from eval.main import _iter_text_files

        files = list(_iter_text_files(sample_text_directory))
        file_names = [f.name for f in files]

        assert file_names == ["file1.txt", "file2.txt", "file3.txt"]

    def test_iter_text_files_ignores_non_txt(self, tmp_path: Path) -> None:
        """Test that non-.txt files are ignored."""
        from eval.main import _iter_text_files

        test_dir = tmp_path / "test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("Text 1", encoding="utf-8")
        (test_dir / "file2.md").write_text("Markdown", encoding="utf-8")
        (test_dir / "file3.py").write_text("Python", encoding="utf-8")
        (test_dir / "file4.txt").write_text("Text 2", encoding="utf-8")

        files = list(_iter_text_files(test_dir))
        file_names = [f.name for f in files]

        assert len(files) == 2
        assert file_names == ["file1.txt", "file4.txt"]

    def test_iter_text_files_ignores_directories(self, tmp_path: Path) -> None:
        """Test that directories named *.txt are ignored."""
        from eval.main import _iter_text_files

        test_dir = tmp_path / "test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("Text", encoding="utf-8")
        (test_dir / "directory.txt").mkdir()  # Directory with .txt extension

        files = list(_iter_text_files(test_dir))

        assert len(files) == 1
        assert files[0].name == "file1.txt"

    def test_iter_text_files_empty_directory(self, tmp_path: Path) -> None:
        """Test iteration over empty directory."""
        from eval.main import _iter_text_files

        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        files = list(_iter_text_files(test_dir))

        assert len(files) == 0


class TestErrorHandling:
    """Test error handling in evaluation workflow."""

    def test_error_missing_text_file(self, mock_openai_runner: Callable[..., str]) -> None:
        """Test error when text file doesn't exist."""
        judge = TextLLMJudge(model_runner=mock_openai_runner, question="Test?")
        nonexistent_path = Path("/nonexistent/file.txt")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            judge.evaluate(nonexistent_path)

    def test_error_directory_as_text_file(
        self, tmp_path: Path, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test error when trying to evaluate a directory."""
        judge = TextLLMJudge(model_runner=mock_openai_runner, question="Test?")

        with pytest.raises(IsADirectoryError, match="Expected a file path, got directory"):
            judge.evaluate(tmp_path)

    def test_error_malformed_llm_response(
        self, sample_text_file: Path, mock_runner_raw_response: Callable[[str], Callable[..., str]]
    ) -> None:
        """Test error when LLM returns invalid JSON."""
        runner = mock_runner_raw_response("This is not valid JSON")
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            judge.evaluate(sample_text_file)

    def test_error_missing_answer_field(
        self, sample_text_file: Path, mock_runner_raw_response: Callable[[str], Callable[..., str]]
    ) -> None:
        """Test error when response is missing answer field."""
        runner = mock_runner_raw_response(json.dumps({"reason": "No answer field"}))
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(ValueError, match="Response missing 'answer' field"):
            judge.evaluate(sample_text_file)

    def test_error_missing_reason_field(
        self, sample_text_file: Path, mock_runner_raw_response: Callable[[str], Callable[..., str]]
    ) -> None:
        """Test error when response is missing reason field."""
        runner = mock_runner_raw_response(json.dumps({"answer": True}))
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(ValueError, match="Response missing 'reason' field"):
            judge.evaluate(sample_text_file)

    def test_error_wrong_answer_type(
        self, sample_text_file: Path, mock_runner_raw_response: Callable[[str], Callable[..., str]]
    ) -> None:
        """Test error when answer is not a boolean."""
        runner = mock_runner_raw_response(json.dumps({"answer": "yes", "reason": "Test"}))
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(TypeError, match="Expected 'answer' to be bool"):
            judge.evaluate(sample_text_file)

    def test_error_wrong_reason_type(
        self, sample_text_file: Path, mock_runner_raw_response: Callable[[str], Callable[..., str]]
    ) -> None:
        """Test error when reason is not a string."""
        runner = mock_runner_raw_response(json.dumps({"answer": True, "reason": 123}))
        judge = TextLLMJudge(model_runner=runner, question="Test?")

        with pytest.raises(TypeError, match="Expected 'reason' to be str"):
            judge.evaluate(sample_text_file)


class TestEndToEndIntegration:
    """End-to-end integration tests using real sample files."""

    def test_single_file_evaluation_with_real_files(
        self, mock_openai_runner: Callable[..., str]
    ) -> None:
        """Test complete workflow from config loading to output with real files."""
        # Use actual sample file from the project
        sample_file = Path("/Users/richy/code/creativai/texts/weak_hook.txt")

        if not sample_file.exists():
            pytest.skip(f"Sample file not found: {sample_file}")

        configs = [{"question": "Does this text have a compelling hook?"}]
        judges = create_judges(configs, mock_openai_runner)
        result = run_all_judges(sample_file, judges)

        # Verify complete output structure
        assert result["source"] == str(sample_file)
        assert "judges" in result
        assert len(result["judges"]) == 1

        judge_result = result["judges"]["judge_0"]
        assert judge_result["question"] == "Does this text have a compelling hook?"
        assert isinstance(judge_result["answer"], bool)
        assert isinstance(judge_result["reason"], str)
        assert isinstance(judge_result["raw_response"], str)

    def test_config_loading_with_actual_example_files(self) -> None:
        """Test loading real config with actual example files."""
        # Create a temporary config that references real example files
        weak_hook_path = Path("/Users/richy/code/creativai/texts/weak_hook.txt")

        if not weak_hook_path.exists():
            pytest.skip("Sample files not found")

        # Create a temporary config
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create directory structure matching project layout
            eval_dir = tmp_path / "eval"
            eval_dir.mkdir()

            texts_dir = tmp_path / "texts"
            texts_dir.mkdir()

            # Copy real file to temp location
            example_file = texts_dir / "weak_hook.txt"
            example_file.write_text(weak_hook_path.read_text(encoding="utf-8"), encoding="utf-8")

            # Create config in eval/ directory
            config = [
                {
                    "question": "Test question?",
                    "false_example_file": "texts/weak_hook.txt",
                }
            ]
            config_file = eval_dir / "llm_judge_prompts.json"
            config_file.write_text(json.dumps(config), encoding="utf-8")

            # Load and verify
            result = load_judge_configs(config_file)

            assert len(result) == 1
            assert result[0]["false_example"] == weak_hook_path.read_text(encoding="utf-8")

    def test_multiple_judges_integration(
        self, sample_text_file: Path, mock_runner_with_response: Callable[[bool, str], Callable[..., str]]
    ) -> None:
        """Test running multiple judges with different responses."""
        configs = [
            {"question": "Is this creative?"},
            {"question": "Is this compelling?"},
            {"question": "Is this well-formatted?"},
        ]

        # Create different runners for each judge
        runners = [
            mock_runner_with_response(True, "Yes, very creative"),
            mock_runner_with_response(False, "Not compelling"),
            mock_runner_with_response(True, "Well formatted"),
        ]

        judges = [
            TextLLMJudge(model_runner=runners[i], question=configs[i]["question"])
            for i in range(3)
        ]

        result = run_all_judges(sample_text_file, judges)

        assert len(result["judges"]) == 3
        assert result["judges"]["judge_0"]["answer"] is True
        assert result["judges"]["judge_0"]["reason"] == "Yes, very creative"
        assert result["judges"]["judge_1"]["answer"] is False
        assert result["judges"]["judge_1"]["reason"] == "Not compelling"
        assert result["judges"]["judge_2"]["answer"] is True
        assert result["judges"]["judge_2"]["reason"] == "Well formatted"
