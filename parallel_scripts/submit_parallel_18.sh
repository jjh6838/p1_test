#!/bin/bash --login
#SBATCH --job-name=supply_p18_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_18_%j.out
#SBATCH --error=outputs_global/logs/parallel_18_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 18/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: GAB, GBR, GIN, IRQ"
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

echo "[INFO] Processing GAB (T3)..."
$PY process_country_supply.py GAB --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GAB completed"
else
    echo "[ERROR] GAB failed"
fi

echo "[INFO] Processing GBR (T3)..."
$PY process_country_supply.py GBR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GBR completed"
else
    echo "[ERROR] GBR failed"
fi

echo "[INFO] Processing GIN (T3)..."
$PY process_country_supply.py GIN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GIN completed"
else
    echo "[ERROR] GIN failed"
fi

echo "[INFO] Processing IRQ (T3)..."
$PY process_country_supply.py IRQ --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] IRQ completed"
else
    echo "[ERROR] IRQ failed"
fi

echo "[INFO] Batch 18/40 (T3) completed at $(date)"
