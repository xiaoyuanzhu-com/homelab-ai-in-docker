#!/bin/bash

# Run Task Integration Tests
# Runs all model/task-based tests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test-utils.sh"

# Parse command line arguments
TASK_TYPE=""
if [ $# -gt 0 ]; then
    TASK_TYPE="$1"
fi

echo -e "${COLOR_BLUE}========================================"
if [ -n "$TASK_TYPE" ]; then
    echo "Task Integration Tests: $TASK_TYPE"
else
    echo "Task Integration Tests (All)"
fi
echo -e "========================================${COLOR_RESET}"
echo ""

# Wait for API to be ready
wait_for_api || exit 1

# Track overall results
TOTAL_TESTS=0
FAILED_TESTS=0

# Determine which tasks to run
if [ -n "$TASK_TYPE" ]; then
    TASK_DIRS=("${SCRIPT_DIR}/tasks/${TASK_TYPE}")
else
    TASK_DIRS=(
        "${SCRIPT_DIR}/tasks/image-ocr"
        "${SCRIPT_DIR}/tasks/image-captioning"
        "${SCRIPT_DIR}/tasks/text-embedding"
        "${SCRIPT_DIR}/tasks/text-generation"
        "${SCRIPT_DIR}/tasks/asr"
    )
fi

# Run tests for each task type
for task_dir in "${TASK_DIRS[@]}"; do
    if [ ! -d "$task_dir" ]; then
        echo -e "${COLOR_YELLOW}Warning: Task directory not found: $task_dir${COLOR_RESET}"
        continue
    fi

    task_name=$(basename "$task_dir")
    echo ""
    echo -e "${COLOR_BLUE}========================================"
    echo "Task: $task_name"
    echo -e "========================================${COLOR_RESET}"

    # Run all test scripts in this task directory
    for test_script in "$task_dir"/*.sh; do
        if [ -f "$test_script" ]; then
            echo ""
            echo -e "${COLOR_BLUE}Running: $(basename "$test_script")${COLOR_RESET}"
            echo ""

            if bash "$test_script"; then
                echo -e "${COLOR_GREEN}✓ Test passed${COLOR_RESET}"
            else
                echo -e "${COLOR_RED}✗ Test failed${COLOR_RESET}"
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi

            TOTAL_TESTS=$((TOTAL_TESTS + 1))
        fi
    done
done

# Print final summary
echo ""
echo -e "${COLOR_BLUE}========================================"
echo "Final Summary"
echo -e "========================================${COLOR_RESET}"
echo "Total test suites: $TOTAL_TESTS"
echo -e "${COLOR_GREEN}Passed: $((TOTAL_TESTS - FAILED_TESTS))${COLOR_RESET}"

if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${COLOR_RED}Failed: $FAILED_TESTS${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_GREEN}All test suites passed!${COLOR_RESET}"
    exit 0
fi
