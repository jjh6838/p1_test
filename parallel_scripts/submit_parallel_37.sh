#!/bin/bash --login
#SBATCH --job-name=p37_t1
#SBATCH --partition=Medium
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_37_%j.out
#SBATCH --error=outputs_global/logs/parallel_37_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 37/40 (T1) at $(date)"
echo "[INFO] Processing 1 countries in this batch: BRA"
echo "[INFO] Tier: T1 | Memory: 64G | CPUs: 40 | Time: 48:00:00"

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

echo "[INFO] Processing BRA (T1)..."
$PY process_country_supply.py BRA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] BRA completed"
else
    echo "[ERROR] BRA failed"
fi

echo "[INFO] Batch 37/40 (T1) completed at $(date)"
