#!/bin/bash --login
#SBATCH --job-name=p36_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_36_%j.out
#SBATCH --error=outputs_global/logs/parallel_36_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 36/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: NER, NIC, NLD, NPL, NZL, OMN, PAN, PNG, POL, PRI"
echo "[INFO] Tier: T5 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing NER (T5)..."
$PY process_country_supply.py NER --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NER completed"
else
    echo "[ERROR] NER failed"
fi

echo "[INFO] Processing NIC (T5)..."
$PY process_country_supply.py NIC --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NIC completed"
else
    echo "[ERROR] NIC failed"
fi

echo "[INFO] Processing NLD (T5)..."
$PY process_country_supply.py NLD --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NLD completed"
else
    echo "[ERROR] NLD failed"
fi

echo "[INFO] Processing NPL (T5)..."
$PY process_country_supply.py NPL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NPL completed"
else
    echo "[ERROR] NPL failed"
fi

echo "[INFO] Processing NZL (T5)..."
$PY process_country_supply.py NZL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] NZL completed"
else
    echo "[ERROR] NZL failed"
fi

echo "[INFO] Processing OMN (T5)..."
$PY process_country_supply.py OMN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] OMN completed"
else
    echo "[ERROR] OMN failed"
fi

echo "[INFO] Processing PAN (T5)..."
$PY process_country_supply.py PAN --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PAN completed"
else
    echo "[ERROR] PAN failed"
fi

echo "[INFO] Processing PNG (T5)..."
$PY process_country_supply.py PNG --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PNG completed"
else
    echo "[ERROR] PNG failed"
fi

echo "[INFO] Processing POL (T5)..."
$PY process_country_supply.py POL --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] POL completed"
else
    echo "[ERROR] POL failed"
fi

echo "[INFO] Processing PRI (T5)..."
$PY process_country_supply.py PRI --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PRI completed"
else
    echo "[ERROR] PRI failed"
fi

echo "[INFO] Batch 36/40 (T5) completed at $(date)"
