#!/bin/bash --login
#SBATCH --job-name=supply_p31_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_31_%j.out
#SBATCH --error=outputs_global/logs/parallel_31_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 31/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: COM, CPV, CRI, CUB, CYM, CYP, CZE, DJI"
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

echo "[INFO] Processing COM (OTHER)..."
$PY process_country_supply.py COM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COM completed"
else
    echo "[ERROR] COM failed"
fi

echo "[INFO] Processing CPV (OTHER)..."
$PY process_country_supply.py CPV --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CPV completed"
else
    echo "[ERROR] CPV failed"
fi

echo "[INFO] Processing CRI (OTHER)..."
$PY process_country_supply.py CRI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CRI completed"
else
    echo "[ERROR] CRI failed"
fi

echo "[INFO] Processing CUB (OTHER)..."
$PY process_country_supply.py CUB --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CUB completed"
else
    echo "[ERROR] CUB failed"
fi

echo "[INFO] Processing CYM (OTHER)..."
$PY process_country_supply.py CYM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CYM completed"
else
    echo "[ERROR] CYM failed"
fi

echo "[INFO] Processing CYP (OTHER)..."
$PY process_country_supply.py CYP --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CYP completed"
else
    echo "[ERROR] CYP failed"
fi

echo "[INFO] Processing CZE (OTHER)..."
$PY process_country_supply.py CZE --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CZE completed"
else
    echo "[ERROR] CZE failed"
fi

echo "[INFO] Processing DJI (OTHER)..."
$PY process_country_supply.py DJI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DJI completed"
else
    echo "[ERROR] DJI failed"
fi

echo "[INFO] Batch 31/40 (OTHER) completed at $(date)"
