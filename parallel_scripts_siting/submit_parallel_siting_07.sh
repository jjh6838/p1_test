#!/bin/bash --login
#SBATCH --job-name=siting_07_t2
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_07_%j.out
#SBATCH --error=outputs_global/logs/siting_07_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 7/24 (T2) at $(date)"
echo "[INFO] Processing 2 countries in this batch: RUS, SAU"
echo "[INFO] Tier: T2 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Process countries in this batch

echo "[INFO] Processing siting analysis for RUS (T2)..."
$PY process_country_siting.py RUS
if [ $? -eq 0 ]; then
    echo "[SUCCESS] RUS siting analysis completed"
else
    echo "[ERROR] RUS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SAU (T2)..."
$PY process_country_siting.py SAU
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SAU siting analysis completed"
else
    echo "[ERROR] SAU siting analysis failed"
fi

echo "[INFO] Siting batch 7/24 (T2) completed at $(date)"
