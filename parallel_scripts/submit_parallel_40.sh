#!/bin/bash --login
#SBATCH --job-name=p40_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_40_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_40_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 40/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: TZA, UGA, URY, UZB, VIR, VNM, VUT, WSM, YEM, ZMB, ZWE"
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

echo "[INFO] Processing TZA (T5)..."
if $PY process_country_supply.py TZA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] TZA completed"
else
    echo "[ERROR] TZA failed"
fi

echo "[INFO] Processing UGA (T5)..."
if $PY process_country_supply.py UGA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] UGA completed"
else
    echo "[ERROR] UGA failed"
fi

echo "[INFO] Processing URY (T5)..."
if $PY process_country_supply.py URY $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] URY completed"
else
    echo "[ERROR] URY failed"
fi

echo "[INFO] Processing UZB (T5)..."
if $PY process_country_supply.py UZB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] UZB completed"
else
    echo "[ERROR] UZB failed"
fi

echo "[INFO] Processing VIR (T5)..."
if $PY process_country_supply.py VIR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] VIR completed"
else
    echo "[ERROR] VIR failed"
fi

echo "[INFO] Processing VNM (T5)..."
if $PY process_country_supply.py VNM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] VNM completed"
else
    echo "[ERROR] VNM failed"
fi

echo "[INFO] Processing VUT (T5)..."
if $PY process_country_supply.py VUT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] VUT completed"
else
    echo "[ERROR] VUT failed"
fi

echo "[INFO] Processing WSM (T5)..."
if $PY process_country_supply.py WSM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] WSM completed"
else
    echo "[ERROR] WSM failed"
fi

echo "[INFO] Processing YEM (T5)..."
if $PY process_country_supply.py YEM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] YEM completed"
else
    echo "[ERROR] YEM failed"
fi

echo "[INFO] Processing ZMB (T5)..."
if $PY process_country_supply.py ZMB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ZMB completed"
else
    echo "[ERROR] ZMB failed"
fi

echo "[INFO] Processing ZWE (T5)..."
if $PY process_country_supply.py ZWE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ZWE completed"
else
    echo "[ERROR] ZWE failed"
fi

echo "[INFO] Batch 40/40 (T5) completed at $(date)"
