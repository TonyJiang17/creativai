#!/bin/bash
# CLI test script for SettingJudge system
# Note: These tests require OPENAI_API_KEY to be set

set -e

echo "============================================================"
echo "CLI Tests for SettingJudge System"
echo "============================================================"
echo ""

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Skipping API-dependent tests."
    echo "To run full tests, set OPENAI_API_KEY environment variable."
    echo ""

    # Run only import tests
    echo "Running import-only tests..."
    python3 /Users/richy/code/creativai/test_setting_judge.py
    exit 0
fi

echo "Test 1: InitialImpressionJudge with --file flag"
python3 /Users/richy/code/creativai/eval/initial_impression_judge.py \
    /Users/richy/code/creativai/texts/setting_complete.txt
echo ""

echo "Test 2: InitialImpressionJudge with --show-raw flag"
python3 /Users/richy/code/creativai/eval/initial_impression_judge.py \
    /Users/richy/code/creativai/texts/setting_complete.txt \
    --show-raw
echo ""

echo "Test 3: InitialImpressionJudge with --model flag"
python3 /Users/richy/code/creativai/eval/initial_impression_judge.py \
    /Users/richy/code/creativai/texts/setting_complete.txt \
    --model gpt-4o-mini
echo ""

echo "Test 4: ConsistencyVerificationJudge with all required args"
python3 /Users/richy/code/creativai/eval/consistency_verification_judge.py \
    --file /Users/richy/code/creativai/texts/setting_complete.txt \
    --who "Detective Sarah Chen" \
    --where "abandoned warehouse on Pier 47" \
    --when "early morning hours"
echo ""

echo "Test 5: ConsistencyVerificationJudge with --show-raw"
python3 /Users/richy/code/creativai/eval/consistency_verification_judge.py \
    --file /Users/richy/code/creativai/texts/setting_complete.txt \
    --who "Detective Sarah Chen" \
    --where "abandoned warehouse" \
    --when "morning" \
    --show-raw
echo ""

echo "Test 6: SettingJudge with --file flag"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --file /Users/richy/code/creativai/texts/setting_complete.txt
echo ""

echo "Test 7: SettingJudge with --text flag"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --text "Detective Sarah Chen stood in the abandoned warehouse on Pier 47, flashlight cutting through the darkness of the early morning hours. She was investigating a crime scene."
echo ""

echo "Test 8: SettingJudge with --json flag"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --file /Users/richy/code/creativai/texts/setting_complete.txt \
    --json
echo ""

echo "Test 9: SettingJudge with --show-raw flag"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --file /Users/richy/code/creativai/texts/setting_complete.txt \
    --show-raw
echo ""

echo "Test 10: SettingJudge with misleading text (should get partial/none rating)"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --file /Users/richy/code/creativai/texts/setting_misleading.txt
echo ""

echo "Test 11: SettingJudge with unclear text"
python3 /Users/richy/code/creativai/eval/setting_judge.py \
    --file /Users/richy/code/creativai/texts/setting_unclear.txt
echo ""

echo "============================================================"
echo "All CLI tests completed successfully!"
echo "============================================================"
