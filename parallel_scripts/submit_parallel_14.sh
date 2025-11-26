#!/bin/bash --login
#SBATCH --job-name=p14_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=896G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output=outputs_global/logs/parallel_14_%j.out
#SBATCH --error=outputs_global/logs/parallel_14_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 14/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: AFG, AGO, BFA, BOL"
echo "[INFO] Tier: T3 | Memory: 896G | CPUs: 56 | Time: 12:00:00"

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

echo "[INFO] Processing AFG (T3)..."
$PY process_country_supply.py AFG --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] AFG completed"
else
    echo "[ERROR] AFG failed"
fi

echo "[INFO] Processing AGO (T3)..."
$PY process_country_supply.py AGO --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] AGO completed"
else
    echo "[ERROR] AGO failed"
fi

echo "[INFO] Processing BFA (T3)..."
$PY process_country_supply.py BFA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BFA completed"
else
    echo "[ERROR] BFA failed"
fi

echo "[INFO] Processing BOL (T3)..."
$PY process_country_supply.py BOL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BOL completed"
else
    echo "[ERROR] BOL failed"
fi

echo "[INFO] Batch 14/40 (T3) completed at $(date)"
