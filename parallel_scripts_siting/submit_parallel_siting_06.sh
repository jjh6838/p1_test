#!/bin/bash --login
#SBATCH --job-name=p06s_t2
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=95G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_06_%j.out
#SBATCH --error=outputs_global/logs/siting_06_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 6/24 (T2) at $(date)"
echo "[INFO] Processing 2 countries in this batch: KAZ, MEX"
echo "[INFO] Tier: T2 | Memory: 95G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing siting analysis for KAZ (T2)..."
if $PY process_country_siting.py KAZ; then
    echo "[SUCCESS] KAZ siting analysis completed"
else
    echo "[ERROR] KAZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MEX (T2)..."
if $PY process_country_siting.py MEX; then
    echo "[SUCCESS] MEX siting analysis completed"
else
    echo "[ERROR] MEX siting analysis failed"
fi

echo "[INFO] Siting batch 6/24 (T2) completed at $(date)"
