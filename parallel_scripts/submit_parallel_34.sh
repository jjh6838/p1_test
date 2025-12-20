#!/bin/bash --login
#SBATCH --job-name=p34_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_34_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_34_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 34/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: LKA, LSO, LTU, LUX, LVA, MAR, MDA, MDG, MDV, MKD"
echo "[INFO] Tier: T5 | Memory: 30G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing LKA (T5)..."
if $PY process_country_supply.py LKA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LKA completed"
else
    echo "[ERROR] LKA failed"
fi

echo "[INFO] Processing LSO (T5)..."
if $PY process_country_supply.py LSO $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LSO completed"
else
    echo "[ERROR] LSO failed"
fi

echo "[INFO] Processing LTU (T5)..."
if $PY process_country_supply.py LTU $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LTU completed"
else
    echo "[ERROR] LTU failed"
fi

echo "[INFO] Processing LUX (T5)..."
if $PY process_country_supply.py LUX $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LUX completed"
else
    echo "[ERROR] LUX failed"
fi

echo "[INFO] Processing LVA (T5)..."
if $PY process_country_supply.py LVA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LVA completed"
else
    echo "[ERROR] LVA failed"
fi

echo "[INFO] Processing MAR (T5)..."
if $PY process_country_supply.py MAR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MAR completed"
else
    echo "[ERROR] MAR failed"
fi

echo "[INFO] Processing MDA (T5)..."
if $PY process_country_supply.py MDA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MDA completed"
else
    echo "[ERROR] MDA failed"
fi

echo "[INFO] Processing MDG (T5)..."
if $PY process_country_supply.py MDG $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MDG completed"
else
    echo "[ERROR] MDG failed"
fi

echo "[INFO] Processing MDV (T5)..."
if $PY process_country_supply.py MDV $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MDV completed"
else
    echo "[ERROR] MDV failed"
fi

echo "[INFO] Processing MKD (T5)..."
if $PY process_country_supply.py MKD $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MKD completed"
else
    echo "[ERROR] MKD failed"
fi

echo "[INFO] Batch 34/40 (T5) completed at $(date)"
