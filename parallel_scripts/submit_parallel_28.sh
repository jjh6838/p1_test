#!/bin/bash --login
#SBATCH --job-name=p28_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_28_%j.out
#SBATCH --error=outputs_global/logs/parallel_28_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 28/40 (T5) at $(date)"
echo "[INFO] Processing 11 countries in this batch: BOL, BRB, BRN, BTN, BWA, CAF, CHE, CHL, CIV, CMR, COD"
echo "[INFO] Tier: T5 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

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

# Process countries in this batch

echo "[INFO] Processing BOL (T5)..."
if $PY process_country_supply.py BOL --output-dir outputs_per_country; then
    echo "[SUCCESS] BOL completed"
else
    echo "[ERROR] BOL failed"
fi

echo "[INFO] Processing BRB (T5)..."
if $PY process_country_supply.py BRB --output-dir outputs_per_country; then
    echo "[SUCCESS] BRB completed"
else
    echo "[ERROR] BRB failed"
fi

echo "[INFO] Processing BRN (T5)..."
if $PY process_country_supply.py BRN --output-dir outputs_per_country; then
    echo "[SUCCESS] BRN completed"
else
    echo "[ERROR] BRN failed"
fi

echo "[INFO] Processing BTN (T5)..."
if $PY process_country_supply.py BTN --output-dir outputs_per_country; then
    echo "[SUCCESS] BTN completed"
else
    echo "[ERROR] BTN failed"
fi

echo "[INFO] Processing BWA (T5)..."
if $PY process_country_supply.py BWA --output-dir outputs_per_country; then
    echo "[SUCCESS] BWA completed"
else
    echo "[ERROR] BWA failed"
fi

echo "[INFO] Processing CAF (T5)..."
if $PY process_country_supply.py CAF --output-dir outputs_per_country; then
    echo "[SUCCESS] CAF completed"
else
    echo "[ERROR] CAF failed"
fi

echo "[INFO] Processing CHE (T5)..."
if $PY process_country_supply.py CHE --output-dir outputs_per_country; then
    echo "[SUCCESS] CHE completed"
else
    echo "[ERROR] CHE failed"
fi

echo "[INFO] Processing CHL (T5)..."
if $PY process_country_supply.py CHL --output-dir outputs_per_country; then
    echo "[SUCCESS] CHL completed"
else
    echo "[ERROR] CHL failed"
fi

echo "[INFO] Processing CIV (T5)..."
if $PY process_country_supply.py CIV --output-dir outputs_per_country; then
    echo "[SUCCESS] CIV completed"
else
    echo "[ERROR] CIV failed"
fi

echo "[INFO] Processing CMR (T5)..."
if $PY process_country_supply.py CMR --output-dir outputs_per_country; then
    echo "[SUCCESS] CMR completed"
else
    echo "[ERROR] CMR failed"
fi

echo "[INFO] Processing COD (T5)..."
if $PY process_country_supply.py COD --output-dir outputs_per_country; then
    echo "[SUCCESS] COD completed"
else
    echo "[ERROR] COD failed"
fi

echo "[INFO] Batch 28/40 (T5) completed at $(date)"
