#!/bin/bash --login
#SBATCH --job-name=p29_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=896G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output=outputs_global/logs/parallel_29_%j.out
#SBATCH --error=outputs_global/logs/parallel_29_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 29/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: BDI, BEL, BEN, BGD, BGR, BHR, BHS, BIH"
echo "[INFO] Tier: OTHER | Memory: 896G | CPUs: 56 | Time: 12:00:00"

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

echo "[INFO] Processing BDI (OTHER)..."
$PY process_country_supply.py BDI --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BDI completed"
else
    echo "[ERROR] BDI failed"
fi

echo "[INFO] Processing BEL (OTHER)..."
$PY process_country_supply.py BEL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BEL completed"
else
    echo "[ERROR] BEL failed"
fi

echo "[INFO] Processing BEN (OTHER)..."
$PY process_country_supply.py BEN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BEN completed"
else
    echo "[ERROR] BEN failed"
fi

echo "[INFO] Processing BGD (OTHER)..."
$PY process_country_supply.py BGD --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BGD completed"
else
    echo "[ERROR] BGD failed"
fi

echo "[INFO] Processing BGR (OTHER)..."
$PY process_country_supply.py BGR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BGR completed"
else
    echo "[ERROR] BGR failed"
fi

echo "[INFO] Processing BHR (OTHER)..."
$PY process_country_supply.py BHR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BHR completed"
else
    echo "[ERROR] BHR failed"
fi

echo "[INFO] Processing BHS (OTHER)..."
$PY process_country_supply.py BHS --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BHS completed"
else
    echo "[ERROR] BHS failed"
fi

echo "[INFO] Processing BIH (OTHER)..."
$PY process_country_supply.py BIH --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BIH completed"
else
    echo "[ERROR] BIH failed"
fi

echo "[INFO] Batch 29/40 (OTHER) completed at $(date)"
