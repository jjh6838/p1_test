#!/bin/bash --login
#SBATCH --job-name=p20_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_20_%j.out
#SBATCH --error=outputs_global/logs/parallel_20_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 20/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: MAR, MDG, MLI, MMR"
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

echo "[INFO] Processing MAR (T3)..."
$PY process_country_supply.py MAR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MAR completed"
else
    echo "[ERROR] MAR failed"
fi

echo "[INFO] Processing MDG (T3)..."
$PY process_country_supply.py MDG --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDG completed"
else
    echo "[ERROR] MDG failed"
fi

echo "[INFO] Processing MLI (T3)..."
$PY process_country_supply.py MLI --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MLI completed"
else
    echo "[ERROR] MLI failed"
fi

echo "[INFO] Processing MMR (T3)..."
$PY process_country_supply.py MMR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MMR completed"
else
    echo "[ERROR] MMR failed"
fi

echo "[INFO] Batch 20/40 (T3) completed at $(date)"
