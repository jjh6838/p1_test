#!/bin/bash --login
#SBATCH --job-name=p19_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_19_%j.out
#SBATCH --error=outputs_global/logs/parallel_19_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 19/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: EGY, ESP, ETH, FRA"
echo "[INFO] Tier: T3 | Memory: 64G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing EGY (T3)..."
$PY process_country_supply.py EGY --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] EGY completed"
else
    echo "[ERROR] EGY failed"
fi

echo "[INFO] Processing ESP (T3)..."
$PY process_country_supply.py ESP --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ESP completed"
else
    echo "[ERROR] ESP failed"
fi

echo "[INFO] Processing ETH (T3)..."
$PY process_country_supply.py ETH --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ETH completed"
else
    echo "[ERROR] ETH failed"
fi

echo "[INFO] Processing FRA (T3)..."
$PY process_country_supply.py FRA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FRA completed"
else
    echo "[ERROR] FRA failed"
fi

echo "[INFO] Batch 19/40 (T3) completed at $(date)"
