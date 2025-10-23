#!/bin/bash

# Run All Integration Tests
# Runs both API tests and task tests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test-utils.sh"

echo -e "${COLOR_BLUE}========================================"
echo "Homelab AI Integration Tests"
echo "Running ALL Tests"
echo -e "========================================${COLOR_RESET}"
echo ""

# Wait for API to be ready
wait_for_api || exit 1

# Track overall results
TOTAL_SUITES=0
FAILED_SUITES=0

# Run API tests
echo ""
echo -e "${COLOR_BLUE}========================================"
echo "Phase 1: API Tests"
echo -e "========================================${COLOR_RESET}"
if bash "${SCRIPT_DIR}/run-api-tests.sh"; then
    echo -e "${COLOR_GREEN}✓ API tests passed${COLOR_RESET}"
else
    echo -e "${COLOR_RED}✗ API tests failed${COLOR_RESET}"
    FAILED_SUITES=$((FAILED_SUITES + 1))
fi
TOTAL_SUITES=$((TOTAL_SUITES + 1))

# Run task tests
echo ""
echo -e "${COLOR_BLUE}========================================"
echo "Phase 2: Task Tests"
echo -e "========================================${COLOR_RESET}"
if bash "${SCRIPT_DIR}/run-task-tests.sh"; then
    echo -e "${COLOR_GREEN}✓ Task tests passed${COLOR_RESET}"
else
    echo -e "${COLOR_RED}✗ Task tests failed${COLOR_RESET}"
    FAILED_SUITES=$((FAILED_SUITES + 1))
fi
TOTAL_SUITES=$((TOTAL_SUITES + 1))

# Print final summary
echo ""
echo -e "${COLOR_BLUE}========================================"
echo "Overall Summary"
echo -e "========================================${COLOR_RESET}"
echo "Total test phases: $TOTAL_SUITES"
echo -e "${COLOR_GREEN}Passed: $((TOTAL_SUITES - FAILED_SUITES))${COLOR_RESET}"

if [ $FAILED_SUITES -gt 0 ]; then
    echo -e "${COLOR_RED}Failed: $FAILED_SUITES${COLOR_RESET}"
    echo ""
    echo "Some tests failed. Please review the output above for details."
    exit 1
else
    echo -e "${COLOR_GREEN}All tests passed!${COLOR_RESET}"
    echo ""
    echo "Your Homelab AI API is working correctly!"
    exit 0
fi
