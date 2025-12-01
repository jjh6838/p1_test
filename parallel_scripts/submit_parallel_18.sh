#!/bin/bash --login
#SBATCH --job-name=p18_t4
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=98G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_18_%j.out
#SBATCH --error=outputs_global/logs/parallel_18_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 18/40 (T4) at $(date)"
echo "[INFO] Processing 2 countries in this batch: FRA, IRQ"
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

echo "[INFO] Processing FRA (T4)..."
$PY process_country_supply.py FRA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FRA completed"
else
    echo "[ERROR] FRA failed"
fi

echo "[INFO] Processing IRQ (T4)..."
$PY process_country_supply.py IRQ --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] IRQ completed"
else
    echo "[ERROR] IRQ failed"
fi

echo "[INFO] Batch 18/40 (T4) completed at $(date)"
