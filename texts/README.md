# Test Text Files

This directory contains text files for testing the SettingJudge system and other evaluators.

## SettingJudge Test Files

The following files are designed to test the SettingJudge system's ability to evaluate whether the opening of a story accurately represents the complete story:

### setting_complete.txt
- **Purpose**: Test case with clear, accurate setting information in the first 50 words
- **Expected behavior**: Should receive "complete" rating
- **First 50 words establish**:
  - Who: Detective Sarah Chen
  - Where: Abandoned warehouse on Pier 47
  - When: Early morning hours (present day)
- **Full story**: Maintains these same elements throughout

### setting_misleading.txt
- **Purpose**: Test case where opening suggests one setting but full story reveals another
- **Expected behavior**: Should receive "partial" or "none" rating
- **First 50 words suggest**:
  - Who: A ship captain and crew
  - Where: A ship in the Atlantic Ocean
  - When: 1847 (historical period)
- **Full story reveals**:
  - Who: Film director Maria Santos and actors
  - Where: Hollywood soundstage
  - When: July 2024 (present day)
- **Key point**: Opening is actually a scene being filmed, not the real setting

### setting_unclear.txt
- **Purpose**: Test case with vague, ambiguous opening that doesn't establish clear setting
- **Expected behavior**: Should receive "unclear" matches, possibly "partial" or "none" rating
- **First 50 words**:
  - Who: Not specified (generic "someone")
  - Where: "Empty space" with walls (very vague)
  - When: "That day" (no temporal context)
- **Full story**: Deliberately maintains vagueness throughout

## Other Test Files

### compelling_hook.txt
- Test file for evaluating story hooks and opening engagement

### weak_hook.txt
- Test file with less engaging opening for comparison

## Usage with SettingJudge

```bash
# Test with complete setting
python3 eval/setting_judge.py --file texts/setting_complete.txt

# Test with misleading opening
python3 eval/setting_judge.py --file texts/setting_misleading.txt

# Test with unclear opening
python3 eval/setting_judge.py --file texts/setting_unclear.txt
```

## Validation

To run comprehensive validation tests on all components:

```bash
# Run unit and integration tests
python3 test_setting_judge.py

# Run CLI tests (requires OPENAI_API_KEY)
./test_cli.sh
```
