#!/bin/bash --login
#SBATCH --job-name=p13_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_13_%j.out
#SBATCH --error=outputs_global/logs/parallel_13_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 13/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: SRB, SSD, SUR, SVK, SVN, SWZ, SYC, SYR"
echo "[INFO] Tier: OTHER | Memory: 64G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing SRB (OTHER)..."
$PY process_country_supply.py SRB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SRB completed"
else
    echo "[ERROR] SRB failed"
fi

echo "[INFO] Processing SSD (OTHER)..."
$PY process_country_supply.py SSD --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SSD completed"
else
    echo "[ERROR] SSD failed"
fi

echo "[INFO] Processing SUR (OTHER)..."
$PY process_country_supply.py SUR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SUR completed"
else
    echo "[ERROR] SUR failed"
fi

echo "[INFO] Processing SVK (OTHER)..."
$PY process_country_supply.py SVK --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVK completed"
else
    echo "[ERROR] SVK failed"
fi

echo "[INFO] Processing SVN (OTHER)..."
$PY process_country_supply.py SVN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVN completed"
else
    echo "[ERROR] SVN failed"
fi

echo "[INFO] Processing SWZ (OTHER)..."
$PY process_country_supply.py SWZ --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SWZ completed"
else
    echo "[ERROR] SWZ failed"
fi

echo "[INFO] Processing SYC (OTHER)..."
$PY process_country_supply.py SYC --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYC completed"
else
    echo "[ERROR] SYC failed"
fi

echo "[INFO] Processing SYR (OTHER)..."
$PY process_country_supply.py SYR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYR completed"
else
    echo "[ERROR] SYR failed"
fi

echo "[INFO] Batch 13/40 (OTHER) completed at $(date)"
