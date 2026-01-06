#!/bin/bash --login
#SBATCH --job-name=p32_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_32_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_32_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 32/40 (T5) at $(date)"
echo "[INFO] Processing 12 countries in this batch: FIN, FJI, FRO, GAB, GBR, GIN, GMB, GNB, GNQ, GRC, GRL, GTM"
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

echo "[INFO] Processing FIN (T5)..."
if $PY process_country_supply.py FIN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] FIN completed"
else
    echo "[ERROR] FIN failed"
fi

echo "[INFO] Processing FJI (T5)..."
if $PY process_country_supply.py FJI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] FJI completed"
else
    echo "[ERROR] FJI failed"
fi

echo "[INFO] Processing FRO (T5)..."
if $PY process_country_supply.py FRO $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] FRO completed"
else
    echo "[ERROR] FRO failed"
fi

echo "[INFO] Processing GAB (T5)..."
if $PY process_country_supply.py GAB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GAB completed"
else
    echo "[ERROR] GAB failed"
fi

echo "[INFO] Processing GBR (T5)..."
if $PY process_country_supply.py GBR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GBR completed"
else
    echo "[ERROR] GBR failed"
fi

echo "[INFO] Processing GIN (T5)..."
if $PY process_country_supply.py GIN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GIN completed"
else
    echo "[ERROR] GIN failed"
fi

echo "[INFO] Processing GMB (T5)..."
if $PY process_country_supply.py GMB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GMB completed"
else
    echo "[ERROR] GMB failed"
fi

echo "[INFO] Processing GNB (T5)..."
if $PY process_country_supply.py GNB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GNB completed"
else
    echo "[ERROR] GNB failed"
fi

echo "[INFO] Processing GNQ (T5)..."
if $PY process_country_supply.py GNQ $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GNQ completed"
else
    echo "[ERROR] GNQ failed"
fi

echo "[INFO] Processing GRC (T5)..."
if $PY process_country_supply.py GRC $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GRC completed"
else
    echo "[ERROR] GRC failed"
fi

echo "[INFO] Processing GRL (T5)..."
if $PY process_country_supply.py GRL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GRL completed"
else
    echo "[ERROR] GRL failed"
fi

echo "[INFO] Processing GTM (T5)..."
if $PY process_country_supply.py GTM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GTM completed"
else
    echo "[ERROR] GTM failed"
fi

echo "[INFO] Batch 32/40 (T5) completed at $(date)"
