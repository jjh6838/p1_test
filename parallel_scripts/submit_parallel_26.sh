#!/bin/bash --login
#SBATCH --job-name=p26_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_26_%j.out
#SBATCH --error=outputs_global/logs/parallel_26_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 26/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: ABW, AFG, AGO, ALB, ARE, ARM, ASM, ATG, AUT, AZE, BDI"
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

echo "[INFO] Processing ABW (T5)..."
if $PY process_country_supply.py ABW $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ABW completed"
else
    echo "[ERROR] ABW failed"
fi

echo "[INFO] Processing AFG (T5)..."
if $PY process_country_supply.py AFG $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] AFG completed"
else
    echo "[ERROR] AFG failed"
fi

echo "[INFO] Processing AGO (T5)..."
if $PY process_country_supply.py AGO $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] AGO completed"
else
    echo "[ERROR] AGO failed"
fi

echo "[INFO] Processing ALB (T5)..."
if $PY process_country_supply.py ALB $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ALB completed"
else
    echo "[ERROR] ALB failed"
fi

echo "[INFO] Processing ARE (T5)..."
if $PY process_country_supply.py ARE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ARE completed"
else
    echo "[ERROR] ARE failed"
fi

echo "[INFO] Processing ARM (T5)..."
if $PY process_country_supply.py ARM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ARM completed"
else
    echo "[ERROR] ARM failed"
fi

echo "[INFO] Processing ASM (T5)..."
if $PY process_country_supply.py ASM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ASM completed"
else
    echo "[ERROR] ASM failed"
fi

echo "[INFO] Processing ATG (T5)..."
if $PY process_country_supply.py ATG $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ATG completed"
else
    echo "[ERROR] ATG failed"
fi

echo "[INFO] Processing AUT (T5)..."
if $PY process_country_supply.py AUT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] AUT completed"
else
    echo "[ERROR] AUT failed"
fi

echo "[INFO] Processing AZE (T5)..."
if $PY process_country_supply.py AZE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] AZE completed"
else
    echo "[ERROR] AZE failed"
fi

echo "[INFO] Processing BDI (T5)..."
if $PY process_country_supply.py BDI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] BDI completed"
else
    echo "[ERROR] BDI failed"
fi

echo "[INFO] Batch 26/40 (T5) completed at $(date)"
