#!/bin/bash --login
#SBATCH --job-name=p19_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_19_%j.out
#SBATCH --error=outputs_global/logs/parallel_19_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 19/40 (T3) at $(date)"
echo "[INFO] Processing 4 countries in this batch: ITA, JPN, KEN, KOR"
echo "[INFO] Tier: T3 | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing ITA (T3)..."
$PY process_country_supply.py ITA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ITA completed"
else
    echo "[ERROR] ITA failed"
fi

echo "[INFO] Processing JPN (T3)..."
$PY process_country_supply.py JPN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] JPN completed"
else
    echo "[ERROR] JPN failed"
fi

echo "[INFO] Processing KEN (T3)..."
$PY process_country_supply.py KEN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KEN completed"
else
    echo "[ERROR] KEN failed"
fi

echo "[INFO] Processing KOR (T3)..."
$PY process_country_supply.py KOR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] KOR completed"
else
    echo "[ERROR] KOR failed"
fi

echo "[INFO] Batch 19/40 (T3) completed at $(date)"
