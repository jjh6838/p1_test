#!/bin/bash --login
#SBATCH --job-name=supply_p37_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_37_%j.out
#SBATCH --error=outputs_global/logs/parallel_37_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 37/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: LUX, LVA, MDA, MDV, MKD, MLT, MNE, MOZ"
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

echo "[INFO] Processing LUX (OTHER)..."
$PY process_country_supply.py LUX --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LUX completed"
else
    echo "[ERROR] LUX failed"
fi

echo "[INFO] Processing LVA (OTHER)..."
$PY process_country_supply.py LVA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LVA completed"
else
    echo "[ERROR] LVA failed"
fi

echo "[INFO] Processing MDA (OTHER)..."
$PY process_country_supply.py MDA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDA completed"
else
    echo "[ERROR] MDA failed"
fi

echo "[INFO] Processing MDV (OTHER)..."
$PY process_country_supply.py MDV --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDV completed"
else
    echo "[ERROR] MDV failed"
fi

echo "[INFO] Processing MKD (OTHER)..."
$PY process_country_supply.py MKD --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MKD completed"
else
    echo "[ERROR] MKD failed"
fi

echo "[INFO] Processing MLT (OTHER)..."
$PY process_country_supply.py MLT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MLT completed"
else
    echo "[ERROR] MLT failed"
fi

echo "[INFO] Processing MNE (OTHER)..."
$PY process_country_supply.py MNE --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MNE completed"
else
    echo "[ERROR] MNE failed"
fi

echo "[INFO] Processing MOZ (OTHER)..."
$PY process_country_supply.py MOZ --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MOZ completed"
else
    echo "[ERROR] MOZ failed"
fi

echo "[INFO] Batch 37/40 (OTHER) completed at $(date)"
