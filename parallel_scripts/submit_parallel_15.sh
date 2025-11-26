#!/bin/bash --login
#SBATCH --job-name=p15_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_15_%j.out
#SBATCH --error=outputs_global/logs/parallel_15_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 15/40 (OTHER) at $(date)"
echo "[INFO] Processing 5 countries in this batch: URY, VIR, VNM, VUT, WSM"
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

echo "[INFO] Processing URY (OTHER)..."
$PY process_country_supply.py URY --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] URY completed"
else
    echo "[ERROR] URY failed"
fi

echo "[INFO] Processing VIR (OTHER)..."
$PY process_country_supply.py VIR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VIR completed"
else
    echo "[ERROR] VIR failed"
fi

echo "[INFO] Processing VNM (OTHER)..."
$PY process_country_supply.py VNM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VNM completed"
else
    echo "[ERROR] VNM failed"
fi

echo "[INFO] Processing VUT (OTHER)..."
$PY process_country_supply.py VUT --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VUT completed"
else
    echo "[ERROR] VUT failed"
fi

echo "[INFO] Processing WSM (OTHER)..."
$PY process_country_supply.py WSM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] WSM completed"
else
    echo "[ERROR] WSM failed"
fi

echo "[INFO] Batch 15/40 (OTHER) completed at $(date)"
