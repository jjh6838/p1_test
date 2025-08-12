#!/bin/bash --login
#SBATCH --job-name=p11_t2
#SBATCH --partition=Medium
#SBATCH --time=48:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_11_%j.out
#SBATCH --error=outputs_global/logs/parallel_11_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 11/40 (T2) at $(date)"
echo "[INFO] Processing 2 countries in this batch: LBY, MEX"
echo "[INFO] Tier: T2 | Memory: 340G | CPUs: 72 | Time: 48:00:00"

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

echo "[INFO] Processing LBY (T2)..."
$PY process_country_supply.py LBY --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LBY completed"
else
    echo "[ERROR] LBY failed"
fi

echo "[INFO] Processing MEX (T2)..."
$PY process_country_supply.py MEX --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MEX completed"
else
    echo "[ERROR] MEX failed"
fi

echo "[INFO] Batch 11/40 (T2) completed at $(date)"
