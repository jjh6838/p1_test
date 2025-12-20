#!/bin/bash
# Run a single parallel script by number (e.g., ./submit_one.sh 06)
# Usage: ./submit_one.sh <script_number> [--run-all-scenarios] [--supply-factor <value>]

# --- Parse arguments ---
RUN_ALL_SCENARIOS=""
SUPPLY_FACTOR=""
SBATCH_EXPORT=""
SCRIPT_NUM=""

while [ $# -gt 0 ]; do
    case $1 in
        --run-all-scenarios)
            RUN_ALL_SCENARIOS="1"
            shift
            ;;
        --supply-factor)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --supply-factor requires a value (e.g., 0.9)"
                exit 1
            fi
            SUPPLY_FACTOR="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 <script_number> [--run-all-scenarios] [--supply-factor <value>]"
            exit 1
            ;;
        *)
            if [ -z "$SCRIPT_NUM" ]; then
                SCRIPT_NUM="$1"
            else
                echo "Unknown argument: $1"
                echo "Usage: $0 <script_number> [--run-all-scenarios] [--supply-factor <value>]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Build SBATCH_EXPORT based on flags
if [ -n "$SUPPLY_FACTOR" ]; then
    SBATCH_EXPORT="--export=ALL,SUPPLY_FACTOR=$SUPPLY_FACTOR"
elif [ -n "$RUN_ALL_SCENARIOS" ]; then
    SBATCH_EXPORT="--export=ALL,RUN_ALL_SCENARIOS=1"
fi

if [ -z "$SCRIPT_NUM" ]; then
    echo "Usage: $0 <script_number> [--run-all-scenarios] [--supply-factor <value>]"
    echo "Example: $0 06"
    echo "Example: $0 06 --run-all-scenarios"
    echo "Example: $0 06 --supply-factor 0.9"
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
if [ -n "$SUPPLY_FACTOR" ]; then
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor)"
elif [ -n "$RUN_ALL_SCENARIOS" ]; then
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi
sbatch $SBATCH_EXPORT "$SCRIPT_FILE"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
