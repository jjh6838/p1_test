#!/bin/bash --login
#SBATCH --job-name=p29_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_29_%j.out
#SBATCH --error=outputs_global/logs/parallel_29_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 29/40 (T3) at $(date)"
echo "[INFO] Processing 2 countries in this batch: ZMB, ZWE"
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

echo "[INFO] Processing ZMB (T3)..."
$PY process_country_supply.py ZMB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZMB completed"
else
    echo "[ERROR] ZMB failed"
fi

echo "[INFO] Processing ZWE (T3)..."
$PY process_country_supply.py ZWE --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZWE completed"
else
    echo "[ERROR] ZWE failed"
fi

echo "[INFO] Batch 29/40 (T3) completed at $(date)"
