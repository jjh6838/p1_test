#!/bin/bash --login
#SBATCH --job-name=p32_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_32_%j.out
#SBATCH --error=outputs_global/logs/parallel_32_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 32/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: HRV, HUN, IRL, ISL, ISR, ITA, JAM, JOR, KEN, KGZ"
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

echo "[INFO] Processing HRV (T5)..."
if $PY process_country_supply.py HRV $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] HRV completed"
else
    echo "[ERROR] HRV failed"
fi

echo "[INFO] Processing HUN (T5)..."
if $PY process_country_supply.py HUN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] HUN completed"
else
    echo "[ERROR] HUN failed"
fi

echo "[INFO] Processing IRL (T5)..."
if $PY process_country_supply.py IRL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] IRL completed"
else
    echo "[ERROR] IRL failed"
fi

echo "[INFO] Processing ISL (T5)..."
if $PY process_country_supply.py ISL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ISL completed"
else
    echo "[ERROR] ISL failed"
fi

echo "[INFO] Processing ISR (T5)..."
if $PY process_country_supply.py ISR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ISR completed"
else
    echo "[ERROR] ISR failed"
fi

echo "[INFO] Processing ITA (T5)..."
if $PY process_country_supply.py ITA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ITA completed"
else
    echo "[ERROR] ITA failed"
fi

echo "[INFO] Processing JAM (T5)..."
if $PY process_country_supply.py JAM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] JAM completed"
else
    echo "[ERROR] JAM failed"
fi

echo "[INFO] Processing JOR (T5)..."
if $PY process_country_supply.py JOR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] JOR completed"
else
    echo "[ERROR] JOR failed"
fi

echo "[INFO] Processing KEN (T5)..."
if $PY process_country_supply.py KEN $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KEN completed"
else
    echo "[ERROR] KEN failed"
fi

echo "[INFO] Processing KGZ (T5)..."
if $PY process_country_supply.py KGZ $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] KGZ completed"
else
    echo "[ERROR] KGZ failed"
fi

echo "[INFO] Batch 32/40 (T5) completed at $(date)"
