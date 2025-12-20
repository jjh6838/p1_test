#!/bin/bash --login
#SBATCH --job-name=p29_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_29_%j.out
#SBATCH --error=outputs_global/logs/parallel_29_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 29/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: COG, COM, CPV, CRI, CUB, CYM, CYP, CZE, DJI, DNK, DOM"
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

# Check if running all scenarios (passed via sbatch --export)
SCENARIO_FLAG=""
if [ "$RUN_ALL_SCENARIOS" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing COG (T5)..."
if $PY process_country_supply.py COG $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] COG completed"
else
    echo "[ERROR] COG failed"
fi

echo "[INFO] Processing COM (T5)..."
if $PY process_country_supply.py COM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] COM completed"
else
    echo "[ERROR] COM failed"
fi

echo "[INFO] Processing CPV (T5)..."
if $PY process_country_supply.py CPV $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CPV completed"
else
    echo "[ERROR] CPV failed"
fi

echo "[INFO] Processing CRI (T5)..."
if $PY process_country_supply.py CRI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CRI completed"
else
    echo "[ERROR] CRI failed"
fi

echo "[INFO] Processing CUB (T5)..."
if $PY process_country_supply.py CUB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CUB completed"
else
    echo "[ERROR] CUB failed"
fi

echo "[INFO] Processing CYM (T5)..."
if $PY process_country_supply.py CYM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CYM completed"
else
    echo "[ERROR] CYM failed"
fi

echo "[INFO] Processing CYP (T5)..."
if $PY process_country_supply.py CYP $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CYP completed"
else
    echo "[ERROR] CYP failed"
fi

echo "[INFO] Processing CZE (T5)..."
if $PY process_country_supply.py CZE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] CZE completed"
else
    echo "[ERROR] CZE failed"
fi

echo "[INFO] Processing DJI (T5)..."
if $PY process_country_supply.py DJI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] DJI completed"
else
    echo "[ERROR] DJI failed"
fi

echo "[INFO] Processing DNK (T5)..."
if $PY process_country_supply.py DNK $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] DNK completed"
else
    echo "[ERROR] DNK failed"
fi

echo "[INFO] Processing DOM (T5)..."
if $PY process_country_supply.py DOM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] DOM completed"
else
    echo "[ERROR] DOM failed"
fi

echo "[INFO] Batch 29/40 (T5) completed at $(date)"
