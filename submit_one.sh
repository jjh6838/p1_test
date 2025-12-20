#!/bin/bash
# Run a single parallel script by number (e.g., ./submit_one.sh 06)
# Usage: ./submit_one.sh <script_number> [--run-all-scenarios]

# --- Parse arguments ---
RUN_ALL_SCENARIOS=""
SBATCH_EXPORT=""
SCRIPT_NUM=""

for arg in "$@"; do
    case $arg in
        --run-all-scenarios)
            RUN_ALL_SCENARIOS="1"
            SBATCH_EXPORT="--export=ALL,RUN_ALL_SCENARIOS=1"
            ;;
        *)
            if [ -z "$SCRIPT_NUM" ]; then
                SCRIPT_NUM="$arg"
            else
                echo "Unknown argument: $arg"
                echo "Usage: $0 <script_number> [--run-all-scenarios]"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$SCRIPT_NUM" ]; then
    echo "Usage: $0 <script_number> [--run-all-scenarios]"
    echo "Example: $0 06"
    echo "Example: $0 06 --run-all-scenarios"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

SCRIPT_NUM=$(printf "%02d" $SCRIPT_NUM)
SCRIPT_FILE="parallel_scripts/submit_parallel_${SCRIPT_NUM}.sh"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script not found: $SCRIPT_FILE"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

echo "[INFO] Submitting script ${SCRIPT_NUM}..."
if [ -n "$RUN_ALL_SCENARIOS" ]; then
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi
sbatch $SBATCH_EXPORT "$SCRIPT_FILE"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
