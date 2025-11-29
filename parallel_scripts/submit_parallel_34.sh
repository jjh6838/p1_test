#!/bin/bash --login
#SBATCH --job-name=p34_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_34_%j.out
#SBATCH --error=outputs_global/logs/parallel_34_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 34/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: GUY, HND, HRV, HUN, IRL, ISL, ISR, JAM"
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

echo "[INFO] Processing GUY (OTHER)..."
$PY process_country_supply.py GUY --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GUY completed"
else
    echo "[ERROR] GUY failed"
fi

echo "[INFO] Processing HND (OTHER)..."
$PY process_country_supply.py HND --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HND completed"
else
    echo "[ERROR] HND failed"
fi

echo "[INFO] Processing HRV (OTHER)..."
$PY process_country_supply.py HRV --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HRV completed"
else
    echo "[ERROR] HRV failed"
fi

echo "[INFO] Processing HUN (OTHER)..."
$PY process_country_supply.py HUN --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HUN completed"
else
    echo "[ERROR] HUN failed"
fi

echo "[INFO] Processing IRL (OTHER)..."
$PY process_country_supply.py IRL --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] IRL completed"
else
    echo "[ERROR] IRL failed"
fi

echo "[INFO] Processing ISL (OTHER)..."
$PY process_country_supply.py ISL --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ISL completed"
else
    echo "[ERROR] ISL failed"
fi

echo "[INFO] Processing ISR (OTHER)..."
$PY process_country_supply.py ISR --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ISR completed"
else
    echo "[ERROR] ISR failed"
fi

echo "[INFO] Processing JAM (OTHER)..."
$PY process_country_supply.py JAM --output-dir outputs_per_country $SCENARIO_FLAG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] JAM completed"
else
    echo "[ERROR] JAM failed"
fi

echo "[INFO] Batch 34/40 (OTHER) completed at $(date)"
