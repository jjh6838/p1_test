#!/bin/bash
# Submit all parallel jobs immediately (SLURM will queue them automatically)
# Usage: ./submit_all_parallel.sh [--run-all-scenarios]

# --- Parse arguments ---
RUN_ALL_SCENARIOS=""
SBATCH_EXPORT=""

for arg in "$@"; do
    case $arg in
        --run-all-scenarios)
            RUN_ALL_SCENARIOS="1"
            SBATCH_EXPORT="--export=ALL,RUN_ALL_SCENARIOS=1"
            echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--run-all-scenarios]"
            exit 1
            ;;
    esac
done

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting 40 parallel jobs..."
echo "[INFO] SLURM will automatically queue and manage job execution (max 8 running at once)"
if [ -n "$RUN_ALL_SCENARIOS" ]; then
    echo "[INFO] Each job will run 5 scenarios (100%, 90%, 80%, 70%, 60%)"
fi
echo ""

# Submit all jobs
for i in {01..40}; do
    echo "[$(date +%H:%M:%S)] Submitting job $i..."
    sbatch $SBATCH_EXPORT parallel_scripts/submit_parallel_${i}.sh
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "[INFO] All 40 jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name '*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (tiered partition strategy):"
echo "  Tier 1 (CHN, USA):              1 country/script  | Interactive partition (168h) | 170G, 36 CPUs"
echo "  Tier 2 (IND, CAN, MEX):         1 country/script  | Medium partition (48h)       | 95G, 40 CPUs"
echo "  Tier 3 (RUS, BRA, AUS, etc.):   1 country/script  | Medium partition (48h)       | 95G, 40 CPUs"
echo "  Tier 4 (TUR, NGA, COL, etc.):   2 countries/script | Short partition (12h)        | 95G, 40 CPUs"
echo "  Tier 5 (all others):           11 countries/script | Short partition (12h)        | 30G, 40 CPUs"
