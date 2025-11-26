#!/bin/bash --login
#SBATCH --job-name=p01_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_01_%j.out
#SBATCH --error=outputs_global/logs/parallel_01_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 1/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: ABW, ALB, ARE, ARM, ASM, ATG, AUT, AZE"
echo "[INFO] Tier: OTHER | Memory: 64G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing ABW (OTHER)..."
$PY process_country_supply.py ABW --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ABW completed"
else
    echo "[ERROR] ABW failed"
fi

echo "[INFO] Processing ALB (OTHER)..."
$PY process_country_supply.py ALB --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ALB completed"
else
    echo "[ERROR] ALB failed"
fi

echo "[INFO] Processing ARE (OTHER)..."
$PY process_country_supply.py ARE --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ARE completed"
else
    echo "[ERROR] ARE failed"
fi

echo "[INFO] Processing ARM (OTHER)..."
$PY process_country_supply.py ARM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ARM completed"
else
    echo "[ERROR] ARM failed"
fi

echo "[INFO] Processing ASM (OTHER)..."
$PY process_country_supply.py ASM --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ASM completed"
else
    echo "[ERROR] ASM failed"
fi

echo "[INFO] Processing ATG (OTHER)..."
$PY process_country_supply.py ATG --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ATG completed"
else
    echo "[ERROR] ATG failed"
fi

echo "[INFO] Processing AUT (OTHER)..."
$PY process_country_supply.py AUT --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] AUT completed"
else
    echo "[ERROR] AUT failed"
fi

echo "[INFO] Processing AZE (OTHER)..."
$PY process_country_supply.py AZE --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] AZE completed"
else
    echo "[ERROR] AZE failed"
fi

echo "[INFO] Batch 1/40 (OTHER) completed at $(date)"
