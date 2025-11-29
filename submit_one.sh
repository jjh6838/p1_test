#!/bin/bash
# Run a single parallel script by number (e.g., ./submit_one.sh 06)
# Usage: ./submit_one.sh <script_number> [--run-all-scenarios]

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <script_number> [--run-all-scenarios]"
    echo "Example: $0 06"
    echo "Example: $0 06 --run-all-scenarios"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

SCRIPT_NUM=$(printf "%02d" $1)
SCRIPT_FILE="parallel_scripts/submit_parallel_${SCRIPT_NUM}.sh"

# Check for --run-all-scenarios flag
RUN_ALL_SCENARIOS=""
if [ "$2" == "--run-all-scenarios" ]; then
    RUN_ALL_SCENARIOS="1"
    echo "[INFO] Running all supply scenarios: 100%, 90%, 80%, 70%, 60%"
else
    echo "[INFO] Running default 100% supply scenario only"
fi
echo ""

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script not found: $SCRIPT_FILE"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

echo "[INFO] Submitting script ${SCRIPT_NUM}..."
if [ -n "$RUN_ALL_SCENARIOS" ]; then
    sbatch --export=ALL,RUN_ALL_SCENARIOS=1 "$SCRIPT_FILE"
else
    sbatch "$SCRIPT_FILE"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
