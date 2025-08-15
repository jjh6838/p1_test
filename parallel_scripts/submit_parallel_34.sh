#!/bin/bash --login
#SBATCH --job-name=p34_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_34_%j.out
#SBATCH --error=outputs_global/logs/parallel_34_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 34/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: GTM, GUM, GUY, HND, HRV, HTI, HUN, IRL"
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

echo "[INFO] Processing GTM (OTHER)..."
$PY process_country_supply.py GTM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GTM completed"
else
    echo "[ERROR] GTM failed"
fi

echo "[INFO] Processing GUM (OTHER)..."
$PY process_country_supply.py GUM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GUM completed"
else
    echo "[ERROR] GUM failed"
fi

echo "[INFO] Processing GUY (OTHER)..."
$PY process_country_supply.py GUY --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GUY completed"
else
    echo "[ERROR] GUY failed"
fi

echo "[INFO] Processing HND (OTHER)..."
$PY process_country_supply.py HND --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HND completed"
else
    echo "[ERROR] HND failed"
fi

echo "[INFO] Processing HRV (OTHER)..."
$PY process_country_supply.py HRV --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HRV completed"
else
    echo "[ERROR] HRV failed"
fi

echo "[INFO] Processing HTI (OTHER)..."
$PY process_country_supply.py HTI --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HTI completed"
else
    echo "[ERROR] HTI failed"
fi

echo "[INFO] Processing HUN (OTHER)..."
$PY process_country_supply.py HUN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HUN completed"
else
    echo "[ERROR] HUN failed"
fi

echo "[INFO] Processing IRL (OTHER)..."
$PY process_country_supply.py IRL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] IRL completed"
else
    echo "[ERROR] IRL failed"
fi

echo "[INFO] Batch 34/40 (OTHER) completed at $(date)"
