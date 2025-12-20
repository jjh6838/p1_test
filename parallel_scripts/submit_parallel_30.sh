#!/bin/bash --login
#SBATCH --job-name=p30_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_30_%j.out
#SBATCH --error=outputs_global/logs/parallel_30_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 30/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: ECU, ERI, ESP, EST, FIN, FJI, FRO, GAB, GBR, GEO, GHA"
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

echo "[INFO] Processing ECU (T5)..."
if $PY process_country_supply.py ECU $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ECU completed"
else
    echo "[ERROR] ECU failed"
fi

echo "[INFO] Processing ERI (T5)..."
if $PY process_country_supply.py ERI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ERI completed"
else
    echo "[ERROR] ERI failed"
fi

echo "[INFO] Processing ESP (T5)..."
if $PY process_country_supply.py ESP $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ESP completed"
else
    echo "[ERROR] ESP failed"
fi

echo "[INFO] Processing EST (T5)..."
if $PY process_country_supply.py EST $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] EST completed"
else
    echo "[ERROR] EST failed"
fi

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

echo "[INFO] Processing GEO (T5)..."
if $PY process_country_supply.py GEO $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GEO completed"
else
    echo "[ERROR] GEO failed"
fi

echo "[INFO] Processing GHA (T5)..."
if $PY process_country_supply.py GHA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] GHA completed"
else
    echo "[ERROR] GHA failed"
fi

echo "[INFO] Batch 30/40 (T5) completed at $(date)"
