#!/bin/bash --login
#SBATCH --job-name=supply_p25_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_25_%j.out
#SBATCH --error=outputs_global/logs/parallel_25_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 25/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: THA, TKM, TUR, TZA"
echo "[INFO] Tier: T3 | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing THA (T3)..."
$PY process_country_supply.py THA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] THA completed"
else
    echo "[ERROR] THA failed"
fi

echo "[INFO] Processing TKM (T3)..."
$PY process_country_supply.py TKM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TKM completed"
else
    echo "[ERROR] TKM failed"
fi

echo "[INFO] Processing TUR (T3)..."
$PY process_country_supply.py TUR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TUR completed"
else
    echo "[ERROR] TUR failed"
fi

echo "[INFO] Processing TZA (T3)..."
$PY process_country_supply.py TZA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TZA completed"
else
    echo "[ERROR] TZA failed"
fi

echo "[INFO] Batch 25/40 (T3) completed at $(date)"
