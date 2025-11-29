#!/bin/bash --login
#SBATCH --job-name=p15_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_15_%j.out
#SBATCH --error=outputs_global/logs/parallel_15_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 15/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: BWA, CAF, CHL, CIV"
echo "[INFO] Tier: T3 | Memory: 100G | CPUs: 40 | Time: 12:00:00"

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

# Check for scenario flag
SCENARIO_FLAG=""
if [ "${RUN_ALL_SCENARIOS:-0}" == "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running all supply scenarios: 100%, 90%, 80%, 70%, 60%"
else
    echo "[INFO] Running default 100% supply scenario"
fi
echo ""

# Process countries in this batch

echo "[INFO] Processing BWA (T3)..."
$PY process_country_supply.py BWA --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BWA completed"
else
    echo "[ERROR] BWA failed"
fi

echo "[INFO] Processing CAF (T3)..."
$PY process_country_supply.py CAF --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CAF completed"
else
    echo "[ERROR] CAF failed"
fi

echo "[INFO] Processing CHL (T3)..."
$PY process_country_supply.py CHL --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CHL completed"
else
    echo "[ERROR] CHL failed"
fi

echo "[INFO] Processing CIV (T3)..."
$PY process_country_supply.py CIV --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CIV completed"
else
    echo "[ERROR] CIV failed"
fi

echo "[INFO] Batch 15/40 (T3) completed at $(date)"
