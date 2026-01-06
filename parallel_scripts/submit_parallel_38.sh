#!/bin/bash --login
#SBATCH --job-name=p38_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_38_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_38_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 38/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: SEN, SGP, SLE, SLV, SOM, SRB, SSD, SUR, SVK, SVN, SWZ"
echo "[INFO] Tier: T5 | Memory: 25G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country/logs outputs_global

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
# Use ${VAR:-} syntax to avoid 'unbound variable' errors with set -u
SCENARIO_FLAG=""
if [ -n "${SUPPLY_FACTOR:-}" ]; then
    SCENARIO_FLAG="--supply-factor $SUPPLY_FACTOR"
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor)"
elif [ "${RUN_ALL_SCENARIOS:-}" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing SEN (T5)..."
if $PY process_country_supply.py SEN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SEN completed"
else
    echo "[ERROR] SEN failed"
fi

echo "[INFO] Processing SGP (T5)..."
if $PY process_country_supply.py SGP $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SGP completed"
else
    echo "[ERROR] SGP failed"
fi

echo "[INFO] Processing SLE (T5)..."
if $PY process_country_supply.py SLE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SLE completed"
else
    echo "[ERROR] SLE failed"
fi

echo "[INFO] Processing SLV (T5)..."
if $PY process_country_supply.py SLV $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SLV completed"
else
    echo "[ERROR] SLV failed"
fi

echo "[INFO] Processing SOM (T5)..."
if $PY process_country_supply.py SOM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SOM completed"
else
    echo "[ERROR] SOM failed"
fi

echo "[INFO] Processing SRB (T5)..."
if $PY process_country_supply.py SRB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SRB completed"
else
    echo "[ERROR] SRB failed"
fi

echo "[INFO] Processing SSD (T5)..."
if $PY process_country_supply.py SSD $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SSD completed"
else
    echo "[ERROR] SSD failed"
fi

echo "[INFO] Processing SUR (T5)..."
if $PY process_country_supply.py SUR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SUR completed"
else
    echo "[ERROR] SUR failed"
fi

echo "[INFO] Processing SVK (T5)..."
if $PY process_country_supply.py SVK $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SVK completed"
else
    echo "[ERROR] SVK failed"
fi

echo "[INFO] Processing SVN (T5)..."
if $PY process_country_supply.py SVN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SVN completed"
else
    echo "[ERROR] SVN failed"
fi

echo "[INFO] Processing SWZ (T5)..."
if $PY process_country_supply.py SWZ $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] SWZ completed"
else
    echo "[ERROR] SWZ failed"
fi

echo "[INFO] Batch 38/40 (T5) completed at $(date)"
