#!/bin/bash --login
#SBATCH --job-name=p36_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_36_%j.out
#SBATCH --error=outputs_global/logs/parallel_36_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 36/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: KWT, LAO, LBN, LBR, LCA, LKA, LSO, LTU"
echo "[INFO] Tier: OTHER | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing KWT (OTHER)..."
$PY process_country_supply.py KWT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KWT completed"
else
    echo "[ERROR] KWT failed"
fi

echo "[INFO] Processing LAO (OTHER)..."
$PY process_country_supply.py LAO --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LAO completed"
else
    echo "[ERROR] LAO failed"
fi

echo "[INFO] Processing LBN (OTHER)..."
$PY process_country_supply.py LBN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LBN completed"
else
    echo "[ERROR] LBN failed"
fi

echo "[INFO] Processing LBR (OTHER)..."
$PY process_country_supply.py LBR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LBR completed"
else
    echo "[ERROR] LBR failed"
fi

echo "[INFO] Processing LCA (OTHER)..."
$PY process_country_supply.py LCA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LCA completed"
else
    echo "[ERROR] LCA failed"
fi

echo "[INFO] Processing LKA (OTHER)..."
$PY process_country_supply.py LKA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LKA completed"
else
    echo "[ERROR] LKA failed"
fi

echo "[INFO] Processing LSO (OTHER)..."
$PY process_country_supply.py LSO --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LSO completed"
else
    echo "[ERROR] LSO failed"
fi

echo "[INFO] Processing LTU (OTHER)..."
$PY process_country_supply.py LTU --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LTU completed"
else
    echo "[ERROR] LTU failed"
fi

echo "[INFO] Batch 36/40 (OTHER) completed at $(date)"
