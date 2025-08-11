#!/bin/bash --login
#SBATCH --job-name=supply_p38_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_38_%j.out
#SBATCH --error=outputs_global/logs/parallel_38_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 38/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: MUS, MWI, NAM, NCL, NIC, NLD, NPL, NRU"
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

echo "[INFO] Processing MUS (OTHER)..."
$PY process_country_supply.py MUS --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MUS completed"
else
    echo "[ERROR] MUS failed"
fi

echo "[INFO] Processing MWI (OTHER)..."
$PY process_country_supply.py MWI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MWI completed"
else
    echo "[ERROR] MWI failed"
fi

echo "[INFO] Processing NAM (OTHER)..."
$PY process_country_supply.py NAM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NAM completed"
else
    echo "[ERROR] NAM failed"
fi

echo "[INFO] Processing NCL (OTHER)..."
$PY process_country_supply.py NCL --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NCL completed"
else
    echo "[ERROR] NCL failed"
fi

echo "[INFO] Processing NIC (OTHER)..."
$PY process_country_supply.py NIC --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NIC completed"
else
    echo "[ERROR] NIC failed"
fi

echo "[INFO] Processing NLD (OTHER)..."
$PY process_country_supply.py NLD --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NLD completed"
else
    echo "[ERROR] NLD failed"
fi

echo "[INFO] Processing NPL (OTHER)..."
$PY process_country_supply.py NPL --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NPL completed"
else
    echo "[ERROR] NPL failed"
fi

echo "[INFO] Processing NRU (OTHER)..."
$PY process_country_supply.py NRU --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NRU completed"
else
    echo "[ERROR] NRU failed"
fi

echo "[INFO] Batch 38/40 (OTHER) completed at $(date)"
