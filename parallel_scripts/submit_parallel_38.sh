#!/bin/bash --login
#SBATCH --job-name=p38_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_38_%j.out
#SBATCH --error=outputs_global/logs/parallel_38_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 38/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: NCL, NIC, NLD, NPL, PAN, PRI, PRK, PRT"
echo "[INFO] Tier: OTHER | Memory: 100G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing NCL (OTHER)..."
$PY process_country_supply.py NCL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NCL completed"
else
    echo "[ERROR] NCL failed"
fi

echo "[INFO] Processing NIC (OTHER)..."
$PY process_country_supply.py NIC --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NIC completed"
else
    echo "[ERROR] NIC failed"
fi

echo "[INFO] Processing NLD (OTHER)..."
$PY process_country_supply.py NLD --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NLD completed"
else
    echo "[ERROR] NLD failed"
fi

echo "[INFO] Processing NPL (OTHER)..."
$PY process_country_supply.py NPL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NPL completed"
else
    echo "[ERROR] NPL failed"
fi

echo "[INFO] Processing PAN (OTHER)..."
$PY process_country_supply.py PAN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PAN completed"
else
    echo "[ERROR] PAN failed"
fi

echo "[INFO] Processing PRI (OTHER)..."
$PY process_country_supply.py PRI --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRI completed"
else
    echo "[ERROR] PRI failed"
fi

echo "[INFO] Processing PRK (OTHER)..."
$PY process_country_supply.py PRK --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRK completed"
else
    echo "[ERROR] PRK failed"
fi

echo "[INFO] Processing PRT (OTHER)..."
$PY process_country_supply.py PRT --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRT completed"
else
    echo "[ERROR] PRT failed"
fi

echo "[INFO] Batch 38/40 (OTHER) completed at $(date)"
