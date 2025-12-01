#!/bin/bash --login
#SBATCH --job-name=p22_t4
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=98G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_22_%j.out
#SBATCH --error=outputs_global/logs/parallel_22_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 22/40 (T4) at $(date)"
echo "[INFO] Processing 2 countries in this batch: PER, PHL"
echo "[INFO] Tier: T4 | Memory: 98G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing PER (T4)..."
$PY process_country_supply.py PER --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PER completed"
else
    echo "[ERROR] PER failed"
fi

echo "[INFO] Processing PHL (T4)..."
$PY process_country_supply.py PHL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PHL completed"
else
    echo "[ERROR] PHL failed"
fi

echo "[INFO] Batch 22/40 (T4) completed at $(date)"
