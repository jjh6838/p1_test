#!/bin/bash --login
#SBATCH --job-name=p02s_t1
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=98G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_02_%j.out
#SBATCH --error=outputs_global/logs/siting_02_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 2/24 (T1) at $(date)"
echo "[INFO] Processing 1 countries in this batch: USA"
echo "[INFO] Tier: T1 | Memory: 98G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Process countries in this batch

echo "[INFO] Processing siting analysis for USA (T1)..."
if $PY process_country_siting.py USA; then
    echo "[SUCCESS] USA siting analysis completed"
else
    echo "[ERROR] USA siting analysis failed"
fi

echo "[INFO] Siting batch 2/24 (T1) completed at $(date)"
