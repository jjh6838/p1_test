#!/bin/bash --login
#SBATCH --job-name=p01s_t1
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=95G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output=outputs_global/logs/siting_01_%j.out
#SBATCH --error=outputs_global/logs/siting_01_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 1/25 (T1) at $(date)"
echo "[INFO] Processing 1 countries in this batch: CHN"
echo "[INFO] Tier: T1 | Memory: 95G | CPUs: 56 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
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

echo "[INFO] Processing siting analysis for CHN (T1)..."
if $PY process_country_siting.py CHN $SCENARIO_FLAG; then
    echo "[SUCCESS] CHN siting analysis completed"
else
    echo "[ERROR] CHN siting analysis failed"
fi

echo "[INFO] Siting batch 1/25 (T1) completed at $(date)"
