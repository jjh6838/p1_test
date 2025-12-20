#!/bin/bash
# Run a single parallel siting script by number
# Usage: ./submit_one_siting.sh <script_number> [--run-all-scenarios] [--supply-factor <value>]

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
    ls -1 parallel_scripts_siting/submit_parallel_siting_*.sh | sed 's/.*submit_parallel_siting_/  /' | sed 's/.sh//'
    exit 1
fi

SCRIPT_NUM=$(printf "%02d" $SCRIPT_NUM)
SCRIPT_FILE="parallel_scripts_siting/submit_parallel_siting_${SCRIPT_NUM}.sh"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script not found: $SCRIPT_FILE"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts_siting/submit_parallel_siting_*.sh | sed 's/.*submit_parallel_siting_/  /' | sed 's/.sh//'
    exit 1
fi

# --- Determine log directory based on scenario ---
if [ -n "$SUPPLY_FACTOR" ]; then
    # Convert supply factor to percentage (e.g., 0.9 -> 90)
    SCENARIO_PCT=$(echo "$SUPPLY_FACTOR * 100" | bc | cut -d. -f1)
    LOG_DIR="outputs_per_country/parquet/2030_supply_${SCENARIO_PCT}%/logs"
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor ${SCENARIO_PCT}%)"
elif [ -n "$RUN_ALL_SCENARIOS" ]; then
    LOG_DIR="outputs_per_country/parquet/logs_run_all_scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
else
    LOG_DIR="outputs_per_country/parquet/2030_supply_100%/logs"
    echo "[INFO] Running default scenario: 100%"
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "[INFO] Submitting siting script ${SCRIPT_NUM}..."
echo "[INFO] Logs will be saved to: ${LOG_DIR}/"
sbatch --output="${LOG_DIR}/siting_${SCRIPT_NUM}_%j.out" \
       --error="${LOG_DIR}/siting_${SCRIPT_NUM}_%j.err" \
       $SBATCH_EXPORT "$SCRIPT_FILE"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
