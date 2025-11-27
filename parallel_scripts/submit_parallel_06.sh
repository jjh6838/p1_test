#!/bin/bash --login
#SBATCH --job-name=p06_t1
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output=outputs_global/logs/parallel_06_%j.out
#SBATCH --error=outputs_global/logs/parallel_06_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 6/40 (T1) at $(date)"
echo "[INFO] Processing 1 countries in this batch: RUS"
echo "[INFO] Tier: T1 | Memory: 100G | CPUs: 56 | Time: 12:00:00"

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

echo "[INFO] Processing RUS (T1)..."
$PY process_country_supply.py RUS --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] RUS completed"
else
    echo "[ERROR] RUS failed"
fi

echo "[INFO] Batch 6/40 (T1) completed at $(date)"
