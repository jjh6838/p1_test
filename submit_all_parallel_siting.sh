#!/bin/bash
# Submit all parallel siting analysis jobs
# Usage: ./submit_all_parallel_siting.sh [--run-all-scenarios] [--supply-factor <value>]

# --- Parse arguments ---
RUN_ALL_SCENARIOS=""
SUPPLY_FACTOR=""
SBATCH_EXPORT=""

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
            echo "Usage: $0 [--run-all-scenarios] [--supply-factor <value>]"
            exit 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--run-all-scenarios] [--supply-factor <value>]"
            exit 1
            ;;
    esac
done

# Build SBATCH_EXPORT based on flags
if [ -n "$SUPPLY_FACTOR" ]; then
    SBATCH_EXPORT="--export=ALL,SUPPLY_FACTOR=$SUPPLY_FACTOR"
    echo "[INFO] Running single scenario: $SUPPLY_FACTOR (supply factor)"
elif [ -n "$RUN_ALL_SCENARIOS" ]; then
    SBATCH_EXPORT="--export=ALL,RUN_ALL_SCENARIOS=1"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting 25 parallel siting analysis jobs..."
echo "[INFO] SLURM will automatically queue and manage job execution"
if [ -n "$RUN_ALL_SCENARIOS" ]; then
    echo "[INFO] Each job will run 5 scenarios (100%, 90%, 80%, 70%, 60%)"
fi
echo ""

# Submit all jobs
for i in {01..25}; do
    echo "[$(date +%H:%M:%S)] Submitting siting job $i..."
    sbatch $SBATCH_EXPORT parallel_scripts_siting/submit_parallel_siting_${i}.sh
    sleep 1
done

echo ""
echo "[INFO] All 25 siting jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name 'siting_*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (siting analysis):"
echo "  Tier 1 (CHN, USA):              1 country/script  | Medium partition (48h) | 95G, 56 CPUs"
echo "  Tier 2 (IND, CAN, etc.):        2 countries/script | Medium partition (24h) | 95G, 40 CPUs"
echo "  Tier 3 (all others):           11 countries/script | Short partition (12h)  | 25G, 40 CPUs"
