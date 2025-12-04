#!/bin/bash --login
#SBATCH --job-name=p40_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_40_%j.out
#SBATCH --error=outputs_global/logs/parallel_40_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 40/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: UGA, URY, UZB, VIR, VNM, VUT, WSM, YEM, ZMB, ZWE"
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

echo "[INFO] Processing UGA (T5)..."
if $PY process_country_supply.py UGA --output-dir outputs_per_country; then
    echo "[SUCCESS] UGA completed"
else
    echo "[ERROR] UGA failed"
fi

echo "[INFO] Processing URY (T5)..."
if $PY process_country_supply.py URY --output-dir outputs_per_country; then
    echo "[SUCCESS] URY completed"
else
    echo "[ERROR] URY failed"
fi

echo "[INFO] Processing UZB (T5)..."
if $PY process_country_supply.py UZB --output-dir outputs_per_country; then
    echo "[SUCCESS] UZB completed"
else
    echo "[ERROR] UZB failed"
fi

echo "[INFO] Processing VIR (T5)..."
if $PY process_country_supply.py VIR --output-dir outputs_per_country; then
    echo "[SUCCESS] VIR completed"
else
    echo "[ERROR] VIR failed"
fi

echo "[INFO] Processing VNM (T5)..."
if $PY process_country_supply.py VNM --output-dir outputs_per_country; then
    echo "[SUCCESS] VNM completed"
else
    echo "[ERROR] VNM failed"
fi

echo "[INFO] Processing VUT (T5)..."
if $PY process_country_supply.py VUT --output-dir outputs_per_country; then
    echo "[SUCCESS] VUT completed"
else
    echo "[ERROR] VUT failed"
fi

echo "[INFO] Processing WSM (T5)..."
if $PY process_country_supply.py WSM --output-dir outputs_per_country; then
    echo "[SUCCESS] WSM completed"
else
    echo "[ERROR] WSM failed"
fi

echo "[INFO] Processing YEM (T5)..."
if $PY process_country_supply.py YEM --output-dir outputs_per_country; then
    echo "[SUCCESS] YEM completed"
else
    echo "[ERROR] YEM failed"
fi

echo "[INFO] Processing ZMB (T5)..."
if $PY process_country_supply.py ZMB --output-dir outputs_per_country; then
    echo "[SUCCESS] ZMB completed"
else
    echo "[ERROR] ZMB failed"
fi

echo "[INFO] Processing ZWE (T5)..."
if $PY process_country_supply.py ZWE --output-dir outputs_per_country; then
    echo "[SUCCESS] ZWE completed"
else
    echo "[ERROR] ZWE failed"
fi

echo "[INFO] Batch 40/40 (T5) completed at $(date)"
