#!/bin/bash --login
#SBATCH --job-name=p33_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_33_%j.out
#SBATCH --error=outputs_global/logs/parallel_33_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 33/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: GEO, GHA, GMB, GNB, GNQ, GRC, GRD, GRL"
echo "[INFO] Tier: OTHER | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing GEO (OTHER)..."
$PY process_country_supply.py GEO --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GEO completed"
else
    echo "[ERROR] GEO failed"
fi

echo "[INFO] Processing GHA (OTHER)..."
$PY process_country_supply.py GHA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GHA completed"
else
    echo "[ERROR] GHA failed"
fi

echo "[INFO] Processing GMB (OTHER)..."
$PY process_country_supply.py GMB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GMB completed"
else
    echo "[ERROR] GMB failed"
fi

echo "[INFO] Processing GNB (OTHER)..."
$PY process_country_supply.py GNB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GNB completed"
else
    echo "[ERROR] GNB failed"
fi

echo "[INFO] Processing GNQ (OTHER)..."
$PY process_country_supply.py GNQ --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GNQ completed"
else
    echo "[ERROR] GNQ failed"
fi

echo "[INFO] Processing GRC (OTHER)..."
$PY process_country_supply.py GRC --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GRC completed"
else
    echo "[ERROR] GRC failed"
fi

echo "[INFO] Processing GRD (OTHER)..."
$PY process_country_supply.py GRD --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GRD completed"
else
    echo "[ERROR] GRD failed"
fi

echo "[INFO] Processing GRL (OTHER)..."
$PY process_country_supply.py GRL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GRL completed"
else
    echo "[ERROR] GRL failed"
fi

echo "[INFO] Batch 33/40 (OTHER) completed at $(date)"
