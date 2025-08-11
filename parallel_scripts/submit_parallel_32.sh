#!/bin/bash --login
#SBATCH --job-name=supply_p32_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_32_%j.out
#SBATCH --error=outputs_global/logs/parallel_32_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 32/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: DMA, DNK, DOM, ERI, EST, FIN, FJI, FRO"
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

echo "[INFO] Processing DMA (OTHER)..."
$PY process_country_supply.py DMA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DMA completed"
else
    echo "[ERROR] DMA failed"
fi

echo "[INFO] Processing DNK (OTHER)..."
$PY process_country_supply.py DNK --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DNK completed"
else
    echo "[ERROR] DNK failed"
fi

echo "[INFO] Processing DOM (OTHER)..."
$PY process_country_supply.py DOM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DOM completed"
else
    echo "[ERROR] DOM failed"
fi

echo "[INFO] Processing ERI (OTHER)..."
$PY process_country_supply.py ERI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ERI completed"
else
    echo "[ERROR] ERI failed"
fi

echo "[INFO] Processing EST (OTHER)..."
$PY process_country_supply.py EST --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] EST completed"
else
    echo "[ERROR] EST failed"
fi

echo "[INFO] Processing FIN (OTHER)..."
$PY process_country_supply.py FIN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FIN completed"
else
    echo "[ERROR] FIN failed"
fi

echo "[INFO] Processing FJI (OTHER)..."
$PY process_country_supply.py FJI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FJI completed"
else
    echo "[ERROR] FJI failed"
fi

echo "[INFO] Processing FRO (OTHER)..."
$PY process_country_supply.py FRO --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FRO completed"
else
    echo "[ERROR] FRO failed"
fi

echo "[INFO] Batch 32/40 (OTHER) completed at $(date)"
