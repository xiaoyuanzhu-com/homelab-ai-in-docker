#!/bin/bash

# Run API Integration Tests
# Runs only the basic API tests (non-model tests)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test-utils.sh"

echo -e "${COLOR_BLUE}========================================"
echo "API Integration Tests"
echo -e "========================================${COLOR_RESET}"
echo ""

# Wait for API to be ready
wait_for_api || exit 1

# Track overall results
TOTAL_TESTS=0
FAILED_TESTS=0

# Run all API tests in order
for test_script in "${SCRIPT_DIR}/api"/*.sh; do
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
