#!/bin/bash --login
#SBATCH --job-name=p37_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_37_%j.out
#SBATCH --error=outputs_global/logs/parallel_37_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 37/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: PRK, PRT, PRY, PSE, QAT, ROU, RWA, SEN, SGP, SLE"
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

echo "[INFO] Processing PRK (T5)..."
if $PY process_country_supply.py PRK $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] PRK completed"
else
    echo "[ERROR] PRK failed"
fi

echo "[INFO] Processing PRT (T5)..."
if $PY process_country_supply.py PRT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] PRT completed"
else
    echo "[ERROR] PRT failed"
fi

echo "[INFO] Processing PRY (T5)..."
if $PY process_country_supply.py PRY $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] PRY completed"
else
    echo "[ERROR] PRY failed"
fi

echo "[INFO] Processing PSE (T5)..."
if $PY process_country_supply.py PSE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] PSE completed"
else
    echo "[ERROR] PSE failed"
fi

echo "[INFO] Processing QAT (T5)..."
if $PY process_country_supply.py QAT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] QAT completed"
else
    echo "[ERROR] QAT failed"
fi

echo "[INFO] Processing ROU (T5)..."
if $PY process_country_supply.py ROU $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] ROU completed"
else
    echo "[ERROR] ROU failed"
fi

echo "[INFO] Processing RWA (T5)..."
if $PY process_country_supply.py RWA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] RWA completed"
else
    echo "[ERROR] RWA failed"
fi

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

echo "[INFO] Batch 37/40 (T5) completed at $(date)"
