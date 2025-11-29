#!/bin/bash --login
#SBATCH --job-name=p39_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_39_%j.out
#SBATCH --error=outputs_global/logs/parallel_39_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 39/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: PSE, QAT, ROU, RWA, SEN, SGP, SLE, SLV"
echo "[INFO] Tier: OTHER | Memory: 100G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing PSE (OTHER)..."
$PY process_country_supply.py PSE --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PSE completed"
else
    echo "[ERROR] PSE failed"
fi

echo "[INFO] Processing QAT (OTHER)..."
$PY process_country_supply.py QAT --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] QAT completed"
else
    echo "[ERROR] QAT failed"
fi

echo "[INFO] Processing ROU (OTHER)..."
$PY process_country_supply.py ROU --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ROU completed"
else
    echo "[ERROR] ROU failed"
fi

echo "[INFO] Processing RWA (OTHER)..."
$PY process_country_supply.py RWA --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] RWA completed"
else
    echo "[ERROR] RWA failed"
fi

echo "[INFO] Processing SEN (OTHER)..."
$PY process_country_supply.py SEN --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SEN completed"
else
    echo "[ERROR] SEN failed"
fi

echo "[INFO] Processing SGP (OTHER)..."
$PY process_country_supply.py SGP --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SGP completed"
else
    echo "[ERROR] SGP failed"
fi

echo "[INFO] Processing SLE (OTHER)..."
$PY process_country_supply.py SLE --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLE completed"
else
    echo "[ERROR] SLE failed"
fi

echo "[INFO] Processing SLV (OTHER)..."
$PY process_country_supply.py SLV --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLV completed"
else
    echo "[ERROR] SLV failed"
fi

echo "[INFO] Batch 39/40 (OTHER) completed at $(date)"
