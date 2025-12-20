#!/bin/bash --login
#SBATCH --job-name=p33_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_33_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_33_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 33/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: KHM, KIR, KNA, KOR, KWT, LAO, LBN, LBR, LBY, LCA"
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

echo "[INFO] Processing KHM (T5)..."
if $PY process_country_supply.py KHM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KHM completed"
else
    echo "[ERROR] KHM failed"
fi

echo "[INFO] Processing KIR (T5)..."
if $PY process_country_supply.py KIR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KIR completed"
else
    echo "[ERROR] KIR failed"
fi

echo "[INFO] Processing KNA (T5)..."
if $PY process_country_supply.py KNA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KNA completed"
else
    echo "[ERROR] KNA failed"
fi

echo "[INFO] Processing KOR (T5)..."
if $PY process_country_supply.py KOR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KOR completed"
else
    echo "[ERROR] KOR failed"
fi

echo "[INFO] Processing KWT (T5)..."
if $PY process_country_supply.py KWT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KWT completed"
else
    echo "[ERROR] KWT failed"
fi

echo "[INFO] Processing LAO (T5)..."
if $PY process_country_supply.py LAO $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LAO completed"
else
    echo "[ERROR] LAO failed"
fi

echo "[INFO] Processing LBN (T5)..."
if $PY process_country_supply.py LBN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LBN completed"
else
    echo "[ERROR] LBN failed"
fi

echo "[INFO] Processing LBR (T5)..."
if $PY process_country_supply.py LBR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LBR completed"
else
    echo "[ERROR] LBR failed"
fi

echo "[INFO] Processing LBY (T5)..."
if $PY process_country_supply.py LBY $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LBY completed"
else
    echo "[ERROR] LBY failed"
fi

echo "[INFO] Processing LCA (T5)..."
if $PY process_country_supply.py LCA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] LCA completed"
else
    echo "[ERROR] LCA failed"
fi

echo "[INFO] Batch 33/40 (T5) completed at $(date)"
