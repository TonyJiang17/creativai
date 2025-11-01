#!/usr/bin/env python3
"""Test script to validate the SettingJudge system end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_imports() -> None:
    """Test that all three modules can be imported."""
    print("Testing imports...")
    try:
        from eval.initial_impression_judge import InitialImpressionJudge
        print("  ✓ InitialImpressionJudge imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import InitialImpressionJudge: {e}")
        sys.exit(1)

    try:
        from eval.consistency_verification_judge import ConsistencyVerificationJudge
        print("  ✓ ConsistencyVerificationJudge imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import ConsistencyVerificationJudge: {e}")
        sys.exit(1)

    try:
        from eval.setting_judge import SettingJudge
        print("  ✓ SettingJudge imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import SettingJudge: {e}")
        sys.exit(1)

    print("All imports successful!\n")


def test_mock_runner() -> None:
    """Test with a mock runner to validate logic without API calls."""
    print("Testing with mock runner...")

    from eval.setting_judge import SettingJudge

    # Create a mock runner that returns predictable JSON responses
    def mock_runner(prompt: str, **kwargs) -> str:
        if "Answer these questions based ONLY on what you just read:" in prompt:
            # This is InitialImpressionJudge
            return """{
                "who": "Detective Sarah Chen",
                "where": "abandoned warehouse on Pier 47",
                "when": "early morning hours (present day)"
            }"""
        elif "INITIAL IMPRESSIONS FROM FIRST 50 WORDS:" in prompt:
            # This is ConsistencyVerificationJudge
            return """{
                "who_match": "yes",
                "where_match": "yes",
                "when_match": "yes",
                "overall_rating": "complete",
                "explanation": "All elements match the full story"
            }"""
        else:
            # Fallback - shouldn't happen
            raise ValueError(f"Unexpected prompt format: {prompt[:100]}")

    judge = SettingJudge(model_runner=mock_runner)

    # Test with file path
    test_file = Path(__file__).parent.parent / "texts" / "setting_complete.txt"
    if not test_file.exists():
        print(f"  ✗ Test file not found: {test_file}")
        sys.exit(1)

    result = judge.evaluate(test_file)

    # Validate result structure
    assert "source" in result, "Missing 'source' in result"
    assert "first_50_words" in result, "Missing 'first_50_words' in result"
    assert "initial_impression" in result, "Missing 'initial_impression' in result"
    assert "consistency_verification" in result, "Missing 'consistency_verification' in result"
    assert "result" in result, "Missing 'result' in result"

    # Validate initial impression structure
    initial = result["initial_impression"]
    assert "result" in initial, "Missing 'result' in initial_impression"
    assert initial["result"]["who"] == "Detective Sarah Chen"
    assert initial["result"]["where"] == "abandoned warehouse on Pier 47"
    assert initial["result"]["when"] == "early morning hours (present day)"

    # Validate consistency verification structure
    consistency = result["consistency_verification"]
    assert "result" in consistency, "Missing 'result' in consistency_verification"
    assert consistency["result"]["who_match"] == "yes"
    assert consistency["result"]["where_match"] == "yes"
    assert consistency["result"]["when_match"] == "yes"
    assert consistency["result"]["overall_rating"] == "complete"

    # Validate final result
    assert result["result"] == "complete", f"Expected 'complete' rating, got {result['result']}"

    print("  ✓ Mock runner test passed")
    print("  ✓ Result structure validated")
    print("  ✓ Rating logic validated\n")


def test_string_input() -> None:
    """Test that SettingJudge can accept string input."""
    print("Testing string input...")

    from eval.setting_judge import SettingJudge

    # Create a mock runner
    def mock_runner(prompt: str, **kwargs) -> str:
        if "Answer these questions based ONLY on what you just read:" in prompt:
            return '{"who": "test", "where": "test location", "when": "test time"}'
        elif "INITIAL IMPRESSIONS FROM FIRST 50 WORDS:" in prompt:
            return """{
                "who_match": "yes",
                "where_match": "yes",
                "when_match": "yes",
                "overall_rating": "complete",
                "explanation": "Test"
            }"""
        else:
            raise ValueError(f"Unexpected prompt format: {prompt[:100]}")

    judge = SettingJudge(model_runner=mock_runner)

    # Test with string input
    test_text = "This is a test story with at least fifty words to make sure we have enough content for the first fifty words extraction. We need to ensure that the string input path works correctly and creates temporary files as needed."

    result = judge.evaluate(test_text)

    assert result["source"] == "<string>", f"Expected source '<string>', got {result['source']}"
    assert "first_50_words" in result
    assert len(result["first_50_words"].split()) <= 50

    print("  ✓ String input test passed\n")


def test_error_handling() -> None:
    """Test error handling for various edge cases."""
    print("Testing error handling...")

    from eval.setting_judge import SettingJudge
    from eval.initial_impression_judge import InitialImpressionJudge
    from eval.consistency_verification_judge import ConsistencyVerificationJudge

    def mock_runner(prompt: str, **kwargs) -> str:
        return '{"who": "test", "where": "test", "when": "test"}'

    # Test FileNotFoundError
    judge = SettingJudge(model_runner=mock_runner)
    nonexistent_file = Path("/nonexistent/file.txt")

    try:
        judge.evaluate(nonexistent_file)
        print("  ✗ Should have raised FileNotFoundError")
        sys.exit(1)
    except FileNotFoundError:
        print("  ✓ FileNotFoundError raised correctly")

    # Test InitialImpressionJudge with invalid JSON
    initial_judge = InitialImpressionJudge(model_runner=lambda p, **k: "not json")
    test_file = Path(__file__).parent.parent / "texts" / "setting_complete.txt"

    try:
        initial_judge.evaluate(test_file)
        print("  ✗ Should have raised ValueError for invalid JSON")
        sys.exit(1)
    except ValueError as e:
        if "Invalid JSON" in str(e):
            print("  ✓ Invalid JSON error raised correctly")
        else:
            print(f"  ✗ Wrong error message: {e}")
            sys.exit(1)

    # Test InitialImpressionJudge with missing fields
    initial_judge = InitialImpressionJudge(
        model_runner=lambda p, **k: '{"who": "test", "where": "test"}'
    )

    try:
        initial_judge.evaluate(test_file)
        print("  ✗ Should have raised ValueError for missing field")
        sys.exit(1)
    except ValueError as e:
        if "Missing required field" in str(e):
            print("  ✓ Missing field error raised correctly")
        else:
            print(f"  ✗ Wrong error message: {e}")
            sys.exit(1)

    # Test ConsistencyVerificationJudge with invalid rating
    consistency_judge = ConsistencyVerificationJudge(
        model_runner=lambda p, **k: """{
            "who_match": "yes",
            "where_match": "yes",
            "when_match": "yes",
            "overall_rating": "invalid",
            "explanation": "test"
        }"""
    )

    try:
        consistency_judge.evaluate(
            text_path=test_file,
            initial_answers={"who": "test", "where": "test", "when": "test"}
        )
        print("  ✗ Should have raised ValueError for invalid rating")
        sys.exit(1)
    except ValueError as e:
        if "overall_rating must be one of" in str(e):
            print("  ✓ Invalid rating error raised correctly")
        else:
            print(f"  ✗ Wrong error message: {e}")
            sys.exit(1)

    print("All error handling tests passed!\n")


def main() -> None:
    """Run all validation tests."""
    print("=" * 60)
    print("SettingJudge System Validation")
    print("=" * 60 + "\n")

    test_imports()
    test_mock_runner()
    test_string_input()
    test_error_handling()

    print("=" * 60)
    print("All validation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
