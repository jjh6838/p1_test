#!/bin/bash --login
#SBATCH --job-name=p36_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=896G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --output=outputs_global/logs/parallel_36_%j.out
#SBATCH --error=outputs_global/logs/parallel_36_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 36/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: LBR, LCA, LKA, LSO, LTU, LUX, LVA, MDA"
echo "[INFO] Tier: OTHER | Memory: 896G | CPUs: 56 | Time: 12:00:00"

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

echo "[INFO] Processing LBR (OTHER)..."
$PY process_country_supply.py LBR --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LBR completed"
else
    echo "[ERROR] LBR failed"
fi

echo "[INFO] Processing LCA (OTHER)..."
$PY process_country_supply.py LCA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LCA completed"
else
    echo "[ERROR] LCA failed"
fi

echo "[INFO] Processing LKA (OTHER)..."
$PY process_country_supply.py LKA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LKA completed"
else
    echo "[ERROR] LKA failed"
fi

echo "[INFO] Processing LSO (OTHER)..."
$PY process_country_supply.py LSO --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LSO completed"
else
    echo "[ERROR] LSO failed"
fi

echo "[INFO] Processing LTU (OTHER)..."
$PY process_country_supply.py LTU --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LTU completed"
else
    echo "[ERROR] LTU failed"
fi

echo "[INFO] Processing LUX (OTHER)..."
$PY process_country_supply.py LUX --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LUX completed"
else
    echo "[ERROR] LUX failed"
fi

echo "[INFO] Processing LVA (OTHER)..."
$PY process_country_supply.py LVA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LVA completed"
else
    echo "[ERROR] LVA failed"
fi

echo "[INFO] Processing MDA (OTHER)..."
$PY process_country_supply.py MDA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDA completed"
else
    echo "[ERROR] MDA failed"
fi

echo "[INFO] Batch 36/40 (OTHER) completed at $(date)"
