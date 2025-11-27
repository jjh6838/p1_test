#!/bin/bash --login
#SBATCH --job-name=p30_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_30_%j.out
#SBATCH --error=outputs_global/logs/parallel_30_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 30/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: BLR, BLZ, BMU, BRB, BRN, BTN, CHE, COG"
echo "[INFO] Tier: OTHER | Memory: 100G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing BLR (OTHER)..."
$PY process_country_supply.py BLR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BLR completed"
else
    echo "[ERROR] BLR failed"
fi

echo "[INFO] Processing BLZ (OTHER)..."
$PY process_country_supply.py BLZ --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BLZ completed"
else
    echo "[ERROR] BLZ failed"
fi

echo "[INFO] Processing BMU (OTHER)..."
$PY process_country_supply.py BMU --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BMU completed"
else
    echo "[ERROR] BMU failed"
fi

echo "[INFO] Processing BRB (OTHER)..."
$PY process_country_supply.py BRB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BRB completed"
else
    echo "[ERROR] BRB failed"
fi

echo "[INFO] Processing BRN (OTHER)..."
$PY process_country_supply.py BRN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BRN completed"
else
    echo "[ERROR] BRN failed"
fi

echo "[INFO] Processing BTN (OTHER)..."
$PY process_country_supply.py BTN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BTN completed"
else
    echo "[ERROR] BTN failed"
fi

echo "[INFO] Processing CHE (OTHER)..."
$PY process_country_supply.py CHE --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CHE completed"
else
    echo "[ERROR] CHE failed"
fi

echo "[INFO] Processing COG (OTHER)..."
$PY process_country_supply.py COG --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COG completed"
else
    echo "[ERROR] COG failed"
fi

echo "[INFO] Batch 30/40 (OTHER) completed at $(date)"
