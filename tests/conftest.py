#!/usr/bin/env python3
"""Shared pytest fixtures for integration tests."""

import json
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def mock_openai_runner() -> Callable[..., str]:
    """
    Create a mock OpenAI runner that returns valid JSON response.

    Returns a callable that accepts a prompt and kwargs and returns
    a JSON string with answer=True and a test reason.
    """
    def runner(prompt: str, **kwargs: Any) -> str:
        return json.dumps({"answer": True, "reason": "Test reason from mock"})

    return runner


@pytest.fixture
def mock_runner_with_response() -> Callable[[bool, str], Callable[..., str]]:
    """
    Factory fixture to create mock runners with specific responses.

    Returns a function that creates mock runners with custom answer/reason.

    Usage:
        mock_runner = mock_runner_with_response(False, "Custom reason")
        result = mock_runner("test prompt")
    """
    def create_runner(answer: bool, reason: str) -> Callable[..., str]:
        def runner(prompt: str, **kwargs: Any) -> str:
            return json.dumps({"answer": answer, "reason": reason})
        return runner

    return create_runner


@pytest.fixture
def mock_runner_raw_response() -> Callable[[str], Callable[..., str]]:
    """
    Factory fixture to create mock runners that return raw string responses.

    Useful for testing error handling with malformed responses.

    Usage:
        mock_runner = mock_runner_raw_response("not valid json")
        result = mock_runner("test prompt")
    """
    def create_runner(response: str) -> Callable[..., str]:
        def runner(prompt: str, **kwargs: Any) -> str:
            return response
        return runner

    return create_runner


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """
    Create a temporary text file with sample content.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to the created temporary file
    """
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is sample text for testing.\nIt has multiple lines.", encoding="utf-8")
    return text_file


@pytest.fixture
def sample_json_config(tmp_path: Path) -> Path:
    """
    Create a test llm_judge_prompts.json configuration file.

    Creates a simple config with one question and no examples.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to the created config file
    """
    config = [
        {
            "question": "Is this text creative?",
        }
    ]

    config_file = tmp_path / "llm_judge_prompts.json"
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_file


@pytest.fixture
def sample_judge_config_with_examples(tmp_path: Path) -> Path:
    """
    Create a test config with example files.

    Creates a config that references true_example_file and false_example_file,
    and also creates those actual example files in the tmp_path.
    Directory structure matches project layout (config in eval/, examples in texts/).

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to the created config file
    """
    # Create directory structure matching project layout
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    texts_dir = tmp_path / "texts"
    texts_dir.mkdir()

    # Create example text files
    true_example = texts_dir / "true_example.txt"
    true_example.write_text("The stars danced in the velvet sky.", encoding="utf-8")

    false_example = texts_dir / "false_example.txt"
    false_example.write_text("The sky was dark at night.", encoding="utf-8")

    # Create config that references these files
    config = [
        {
            "question": "Does this text contain creative metaphors?",
            "true_example_file": "texts/true_example.txt",
            "false_example_file": "texts/false_example.txt",
        }
    ]

    config_file = eval_dir / "llm_judge_prompts.json"
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_file


@pytest.fixture
def sample_text_directory(tmp_path: Path) -> Path:
    """
    Create a temporary directory with multiple text files for batch testing.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to the directory containing text files
    """
    text_dir = tmp_path / "texts"
    text_dir.mkdir()

    (text_dir / "file1.txt").write_text("First sample text.", encoding="utf-8")
    (text_dir / "file2.txt").write_text("Second sample text.", encoding="utf-8")
    (text_dir / "file3.txt").write_text("Third sample text.", encoding="utf-8")

    # Also add a non-txt file to ensure it's ignored
    (text_dir / "README.md").write_text("# Not a text file", encoding="utf-8")

    return text_dir
