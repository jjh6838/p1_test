#!/bin/bash --login
#SBATCH --job-name=p28_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_28_%j.out
#SBATCH --error=outputs_global/logs/parallel_28_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 28/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: UKR, UZB, YEM, ZAF"
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

echo "[INFO] Processing UKR (T3)..."
$PY process_country_supply.py UKR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UKR completed"
else
    echo "[ERROR] UKR failed"
fi

echo "[INFO] Processing UZB (T3)..."
$PY process_country_supply.py UZB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UZB completed"
else
    echo "[ERROR] UZB failed"
fi

echo "[INFO] Processing YEM (T3)..."
$PY process_country_supply.py YEM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] YEM completed"
else
    echo "[ERROR] YEM failed"
fi

echo "[INFO] Processing ZAF (T3)..."
$PY process_country_supply.py ZAF --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZAF completed"
else
    echo "[ERROR] ZAF failed"
fi

echo "[INFO] Batch 28/40 (T3) completed at $(date)"
