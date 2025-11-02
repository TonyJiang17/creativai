# Testing Documentation

This directory contains comprehensive tests for the CreativAI evaluation system, focusing on the JSON-driven LLM judge framework.

## Test Structure

```
tests/
├── conftest.py                      # Shared pytest fixtures
├── test_text_llm_judge.py          # Unit tests for TextLLMJudge class
├── test_main_integration.py        # Integration tests for eval/main.py
└── README.md                        # This file
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
# Run only TextLLMJudge unit tests
pytest tests/test_text_llm_judge.py -v

# Run only integration tests
pytest tests/test_main_integration.py -v
```

### Run Specific Test Class or Method

```bash
# Run all config loading tests
pytest tests/test_main_integration.py::TestLoadJudgeConfigs -v

# Run a specific test method
pytest tests/test_main_integration.py::TestLoadJudgeConfigs::test_load_judge_configs_valid -v
```

### Run Tests with Coverage

```bash
# Get coverage report for eval module
pytest tests/ --cov=eval --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=eval --cov-report=html
# Open htmlcov/index.html in browser
```

### Run Tests in Verbose Mode with Output

```bash
# Show print statements and detailed output
pytest tests/ -v -s
```

## Test Categories

### Unit Tests (`test_text_llm_judge.py`)

Tests for the core `TextLLMJudge` class that provides the foundation for LLM-based evaluations.

**Test Classes:**
- `TestTextLLMJudgeInstantiation` - Constructor and initialization
- `TestBuildPrompt` - Prompt building with questions and examples
- `TestParseResponse` - JSON response parsing and validation
- `TestEvaluate` - End-to-end evaluation workflow

**Key Coverage:**
- Judge creation with various configurations
- Prompt construction with optional examples
- Response parsing with error handling
- File I/O and error cases (missing files, directories)

### Integration Tests (`test_main_integration.py`)

Tests for the complete JSON-driven evaluation system including config loading, judge creation, and batch processing.

**Test Classes:**
- `TestLoadJudgeConfigs` - JSON config loading and example file resolution
- `TestCreateJudges` - Judge instantiation from configs
- `TestRunAllJudges` - Single file evaluation with multiple judges
- `TestBatchEvaluation` - Directory batch processing
- `TestErrorHandling` - Error scenarios and edge cases
- `TestEndToEndIntegration` - Complete workflows with real files

**Key Coverage:**
- Config file parsing and validation
- Example file loading and content substitution
- Multi-judge evaluation orchestration
- Batch processing workflows
- Error handling for malformed inputs
- Output format validation

## Mocking Strategy

### Mock LLM Responses

All tests use mocked LLM runners to ensure deterministic, fast tests without requiring actual API calls.

**Available Fixtures (from `conftest.py`):**

```python
# Simple mock that returns True answer
def test_example(mock_openai_runner):
    judge = TextLLMJudge(model_runner=mock_openai_runner, question="Test?")
    result = judge.evaluate(text_file)
    # Returns: {"answer": True, "reason": "Test reason from mock"}

# Factory for custom responses
def test_custom(mock_runner_with_response):
    runner = mock_runner_with_response(False, "Custom reason")
    judge = TextLLMJudge(model_runner=runner, question="Test?")
    # Returns: {"answer": False, "reason": "Custom reason"}

# Factory for raw/malformed responses
def test_error(mock_runner_raw_response):
    runner = mock_runner_raw_response("invalid json")
    judge = TextLLMJudge(model_runner=runner, question="Test?")
    # Useful for testing error handling
```

### Temporary Files and Directories

Tests use pytest's built-in `tmp_path` fixture for creating temporary files and directories.

**Available Fixtures:**

```python
# Single text file
def test_with_file(sample_text_file):
    # sample_text_file is a Path to a temp .txt file
    result = judge.evaluate(sample_text_file)

# JSON config file
def test_with_config(sample_json_config):
    # sample_json_config is a Path to temp llm_judge_prompts.json
    configs = load_judge_configs(sample_json_config)

# Config with example files
def test_with_examples(sample_judge_config_with_examples):
    # Creates config + example text files
    configs = load_judge_configs(sample_judge_config_with_examples)

# Directory of text files
def test_batch(sample_text_directory):
    # sample_text_directory contains file1.txt, file2.txt, file3.txt
    for file in sample_text_directory.glob("*.txt"):
        result = judge.evaluate(file)
```

## Adding New Tests

### Adding a Unit Test

1. Add test method to appropriate class in `test_text_llm_judge.py`
2. Use descriptive test name: `test_<feature>_<scenario>`
3. Follow AAA pattern: Arrange, Act, Assert
4. Use fixtures from `conftest.py` for common setup

Example:

```python
def test_parse_response_with_extra_fields(self) -> None:
    """Test that extra fields in response are ignored."""
    runner = mock_model_runner_json(True, "test")
    judge = TextLLMJudge(model_runner=runner, question="Test?")

    response = json.dumps({
        "answer": True,
        "reason": "Test reason",
        "extra_field": "ignored"
    })
    result = judge.parse_response(response)

    assert result["answer"] is True
    assert result["reason"] == "Test reason"
    assert "extra_field" not in result
```

### Adding an Integration Test

1. Add test method to appropriate class in `test_main_integration.py`
2. Use fixtures for setup (mocks, temp files)
3. Test complete workflows, not isolated units
4. Verify output structure and types

Example:

```python
def test_config_with_multiple_examples(
    self, tmp_path: Path, mock_openai_runner: Callable[..., str]
) -> None:
    """Test config loading with multiple judges each having examples."""
    # Create example files
    texts_dir = tmp_path / "texts"
    texts_dir.mkdir()

    (texts_dir / "true1.txt").write_text("Example 1", encoding="utf-8")
    (texts_dir / "false1.txt").write_text("Example 2", encoding="utf-8")

    # Create config
    config = [
        {
            "question": "Question 1?",
            "true_example_file": "texts/true1.txt",
            "false_example_file": "texts/false1.txt",
        }
    ]
    config_file = tmp_path / "llm_judge_prompts.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")

    # Test loading
    result = load_judge_configs(config_file)

    assert result[0]["true_example"] == "Example 1"
    assert result[0]["false_example"] == "Example 2"
```

## Test Naming Conventions

- **Test files**: `test_<module_name>.py`
- **Test classes**: `Test<FeatureName>` (PascalCase)
- **Test methods**: `test_<feature>_<scenario>` (snake_case)
- Use descriptive names that explain what is being tested

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Deterministic**: Use mocks to ensure tests always produce same results
3. **Fast**: Tests should run quickly (no actual API calls, minimal I/O)
4. **Readable**: Test names and code should clearly show intent
5. **Comprehensive**: Cover happy paths, edge cases, and error conditions
6. **Maintainable**: Use fixtures for common setup to avoid duplication

## Debugging Failed Tests

### View detailed output

```bash
pytest tests/test_main_integration.py::TestLoadJudgeConfigs::test_load_judge_configs_valid -v -s
```

### Drop into debugger on failure

```bash
pytest tests/ --pdb
```

### Run last failed tests only

```bash
pytest tests/ --lf
```

### Show local variables on failure

```bash
pytest tests/ -l
```

## Continuous Integration

These tests are designed to run in CI environments without requiring:
- OpenAI API keys
- External network access
- Real text files (uses temporary files)

All external dependencies are mocked, making tests fast and reliable.

## Test Coverage Goals

- **Critical paths**: 100% coverage
  - Config loading and validation
  - Judge creation and initialization
  - Single file evaluation
  - Response parsing
  - Error handling
- **Secondary paths**: 90%+ coverage
  - Batch processing
  - CLI workflows
  - Edge cases

Run coverage checks regularly to ensure new code is tested:

```bash
pytest tests/ --cov=eval --cov-report=term-missing --cov-fail-under=90
```
