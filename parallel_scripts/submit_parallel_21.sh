#!/bin/bash --login
#SBATCH --job-name=p21_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_21_%j.out
#SBATCH --error=outputs_global/logs/parallel_21_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 21/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: MRT, MYS, NER, NGA"
echo "[INFO] Tier: T3 | Memory: 100G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing MRT (T3)..."
$PY process_country_supply.py MRT --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MRT completed"
else
    echo "[ERROR] MRT failed"
fi

echo "[INFO] Processing MYS (T3)..."
$PY process_country_supply.py MYS --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MYS completed"
else
    echo "[ERROR] MYS failed"
fi

echo "[INFO] Processing NER (T3)..."
$PY process_country_supply.py NER --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NER completed"
else
    echo "[ERROR] NER failed"
fi

echo "[INFO] Processing NGA (T3)..."
$PY process_country_supply.py NGA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NGA completed"
else
    echo "[ERROR] NGA failed"
fi

echo "[INFO] Batch 21/40 (T3) completed at $(date)"
