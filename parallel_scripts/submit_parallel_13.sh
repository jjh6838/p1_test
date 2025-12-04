#!/bin/bash --login
#SBATCH --job-name=p13_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_13_%j.out
#SBATCH --error=outputs_global/logs/parallel_13_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 13/40 (T3) at $(date)"
echo "[INFO] Processing 1 countries in this batch: RUS"
echo "[INFO] Tier: T3 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing RUS (T3)..."
if $PY process_country_supply.py RUS --output-dir outputs_per_country; then
    echo "[SUCCESS] RUS completed"
else
    echo "[ERROR] RUS failed"
fi

echo "[INFO] Batch 13/40 (T3) completed at $(date)"
