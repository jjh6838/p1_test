#!/bin/bash --login
#SBATCH --job-name=supply_p35_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_35_%j.out
#SBATCH --error=outputs_global/logs/parallel_35_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 35/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: ISL, ISR, JAM, JOR, KGZ, KHM, KIR, KNA"
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

echo "[INFO] Processing ISL (OTHER)..."
$PY process_country_supply.py ISL --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ISL completed"
else
    echo "[ERROR] ISL failed"
fi

echo "[INFO] Processing ISR (OTHER)..."
$PY process_country_supply.py ISR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ISR completed"
else
    echo "[ERROR] ISR failed"
fi

echo "[INFO] Processing JAM (OTHER)..."
$PY process_country_supply.py JAM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] JAM completed"
else
    echo "[ERROR] JAM failed"
fi

echo "[INFO] Processing JOR (OTHER)..."
$PY process_country_supply.py JOR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] JOR completed"
else
    echo "[ERROR] JOR failed"
fi

echo "[INFO] Processing KGZ (OTHER)..."
$PY process_country_supply.py KGZ --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KGZ completed"
else
    echo "[ERROR] KGZ failed"
fi

echo "[INFO] Processing KHM (OTHER)..."
$PY process_country_supply.py KHM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KHM completed"
else
    echo "[ERROR] KHM failed"
fi

echo "[INFO] Processing KIR (OTHER)..."
$PY process_country_supply.py KIR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KIR completed"
else
    echo "[ERROR] KIR failed"
fi

echo "[INFO] Processing KNA (OTHER)..."
$PY process_country_supply.py KNA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KNA completed"
else
    echo "[ERROR] KNA failed"
fi

echo "[INFO] Batch 35/40 (OTHER) completed at $(date)"
