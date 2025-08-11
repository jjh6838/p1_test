#!/bin/bash --login
#SBATCH --job-name=supply_p16_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_16_%j.out
#SBATCH --error=outputs_global/logs/parallel_16_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 16/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: CMR, COL, DEU, ECU"
echo "[INFO] Tier: T3 | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing CMR (T3)..."
$PY process_country_supply.py CMR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CMR completed"
else
    echo "[ERROR] CMR failed"
fi

echo "[INFO] Processing COL (T3)..."
$PY process_country_supply.py COL --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COL completed"
else
    echo "[ERROR] COL failed"
fi

echo "[INFO] Processing DEU (T3)..."
$PY process_country_supply.py DEU --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DEU completed"
else
    echo "[ERROR] DEU failed"
fi

echo "[INFO] Processing ECU (T3)..."
$PY process_country_supply.py ECU --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ECU completed"
else
    echo "[ERROR] ECU failed"
fi

echo "[INFO] Batch 16/40 (T3) completed at $(date)"
