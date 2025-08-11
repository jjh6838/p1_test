#!/bin/bash --login
#SBATCH --job-name=supply_p39_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_39_%j.out
#SBATCH --error=outputs_global/logs/parallel_39_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 39/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: PAN, PRI, PRK, PRT, PSE, PYF, QAT, ROU"
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

echo "[INFO] Processing PAN (OTHER)..."
$PY process_country_supply.py PAN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PAN completed"
else
    echo "[ERROR] PAN failed"
fi

echo "[INFO] Processing PRI (OTHER)..."
$PY process_country_supply.py PRI --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRI completed"
else
    echo "[ERROR] PRI failed"
fi

echo "[INFO] Processing PRK (OTHER)..."
$PY process_country_supply.py PRK --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRK completed"
else
    echo "[ERROR] PRK failed"
fi

echo "[INFO] Processing PRT (OTHER)..."
$PY process_country_supply.py PRT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRT completed"
else
    echo "[ERROR] PRT failed"
fi

echo "[INFO] Processing PSE (OTHER)..."
$PY process_country_supply.py PSE --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PSE completed"
else
    echo "[ERROR] PSE failed"
fi

echo "[INFO] Processing PYF (OTHER)..."
$PY process_country_supply.py PYF --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PYF completed"
else
    echo "[ERROR] PYF failed"
fi

echo "[INFO] Processing QAT (OTHER)..."
$PY process_country_supply.py QAT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] QAT completed"
else
    echo "[ERROR] QAT failed"
fi

echo "[INFO] Processing ROU (OTHER)..."
$PY process_country_supply.py ROU --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ROU completed"
else
    echo "[ERROR] ROU failed"
fi

echo "[INFO] Batch 39/40 (OTHER) completed at $(date)"
