#!/bin/bash --login
#SBATCH --job-name=p27_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_27_%j.out
#SBATCH --error=outputs_global/logs/parallel_27_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 27/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: BEL, BEN, BFA, BGD, BGR, BHR, BHS, BIH, BLR, BLZ, BMU"
echo "[INFO] Tier: T5 | Memory: 30G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing BEL (T5)..."
if $PY process_country_supply.py BEL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BEL completed"
else
    echo "[ERROR] BEL failed"
fi

echo "[INFO] Processing BEN (T5)..."
if $PY process_country_supply.py BEN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BEN completed"
else
    echo "[ERROR] BEN failed"
fi

echo "[INFO] Processing BFA (T5)..."
if $PY process_country_supply.py BFA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BFA completed"
else
    echo "[ERROR] BFA failed"
fi

echo "[INFO] Processing BGD (T5)..."
if $PY process_country_supply.py BGD $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BGD completed"
else
    echo "[ERROR] BGD failed"
fi

echo "[INFO] Processing BGR (T5)..."
if $PY process_country_supply.py BGR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BGR completed"
else
    echo "[ERROR] BGR failed"
fi

echo "[INFO] Processing BHR (T5)..."
if $PY process_country_supply.py BHR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BHR completed"
else
    echo "[ERROR] BHR failed"
fi

echo "[INFO] Processing BHS (T5)..."
if $PY process_country_supply.py BHS $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BHS completed"
else
    echo "[ERROR] BHS failed"
fi

echo "[INFO] Processing BIH (T5)..."
if $PY process_country_supply.py BIH $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BIH completed"
else
    echo "[ERROR] BIH failed"
fi

echo "[INFO] Processing BLR (T5)..."
if $PY process_country_supply.py BLR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BLR completed"
else
    echo "[ERROR] BLR failed"
fi

echo "[INFO] Processing BLZ (T5)..."
if $PY process_country_supply.py BLZ $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BLZ completed"
else
    echo "[ERROR] BLZ failed"
fi

echo "[INFO] Processing BMU (T5)..."
if $PY process_country_supply.py BMU $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BMU completed"
else
    echo "[ERROR] BMU failed"
fi

echo "[INFO] Batch 27/40 (T5) completed at $(date)"
