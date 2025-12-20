#!/bin/bash --login
#SBATCH --job-name=p14_t3
#SBATCH --partition=Medium
#SBATCH --time=48:00:00
#SBATCH --mem=95G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_14_%j.out
#SBATCH --error=outputs_global/logs/parallel_14_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 14/40 (T3) at $(date)"
echo "[INFO] Processing 1 countries in this batch: RUS"
echo "[INFO] Tier: T3 | Memory: 95G | CPUs: 40 | Time: 48:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path to avoid activation issues in batch shells
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Check scenario flags (passed via sbatch --export)
SCENARIO_FLAG=""
if [ -n "$SUPPLY_FACTOR" ]; then
    SCENARIO_FLAG="--supply-factor $SUPPLY_FACTOR"
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor)"
elif [ "$RUN_ALL_SCENARIOS" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing RUS (T3)..."
if $PY process_country_supply.py RUS $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] RUS completed"
else
    echo "[ERROR] RUS failed"
fi

echo "[INFO] Batch 14/40 (T3) completed at $(date)"
