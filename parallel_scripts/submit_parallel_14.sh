#!/bin/bash --login
#SBATCH --job-name=p14_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_14_%j.out
#SBATCH --error=outputs_global/logs/parallel_14_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 14/40 (OTHER) at $(date)"
echo "[INFO] Processing 8 countries in this batch: TGO, TJK, TLS, TON, TTO, TUN, TWN, UGA"
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

echo "[INFO] Processing TGO (OTHER)..."
$PY process_country_supply.py TGO --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TGO completed"
else
    echo "[ERROR] TGO failed"
fi

echo "[INFO] Processing TJK (OTHER)..."
$PY process_country_supply.py TJK --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TJK completed"
else
    echo "[ERROR] TJK failed"
fi

echo "[INFO] Processing TLS (OTHER)..."
$PY process_country_supply.py TLS --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TLS completed"
else
    echo "[ERROR] TLS failed"
fi

echo "[INFO] Processing TON (OTHER)..."
$PY process_country_supply.py TON --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TON completed"
else
    echo "[ERROR] TON failed"
fi

echo "[INFO] Processing TTO (OTHER)..."
$PY process_country_supply.py TTO --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TTO completed"
else
    echo "[ERROR] TTO failed"
fi

echo "[INFO] Processing TUN (OTHER)..."
$PY process_country_supply.py TUN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TUN completed"
else
    echo "[ERROR] TUN failed"
fi

echo "[INFO] Processing TWN (OTHER)..."
$PY process_country_supply.py TWN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TWN completed"
else
    echo "[ERROR] TWN failed"
fi

echo "[INFO] Processing UGA (OTHER)..."
$PY process_country_supply.py UGA --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UGA completed"
else
    echo "[ERROR] UGA failed"
fi

echo "[INFO] Batch 14/40 (OTHER) completed at $(date)"
