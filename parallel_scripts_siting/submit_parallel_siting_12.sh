#!/bin/bash --login
#SBATCH --job-name=p12s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_12_%j.out
#SBATCH --error=outputs_global/logs/siting_12_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 12/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: DNK, DOM, DZA, ECU, EGY, ERI, ESP, EST, ETH, FIN, FJI"
echo "[INFO] Tier: T3 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Process countries in this batch

echo "[INFO] Processing siting analysis for DNK (T3)..."
if $PY process_country_siting.py DNK; then
    echo "[SUCCESS] DNK siting analysis completed"
else
    echo "[ERROR] DNK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DOM (T3)..."
if $PY process_country_siting.py DOM; then
    echo "[SUCCESS] DOM siting analysis completed"
else
    echo "[ERROR] DOM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DZA (T3)..."
if $PY process_country_siting.py DZA; then
    echo "[SUCCESS] DZA siting analysis completed"
else
    echo "[ERROR] DZA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ECU (T3)..."
if $PY process_country_siting.py ECU; then
    echo "[SUCCESS] ECU siting analysis completed"
else
    echo "[ERROR] ECU siting analysis failed"
fi

echo "[INFO] Processing siting analysis for EGY (T3)..."
if $PY process_country_siting.py EGY; then
    echo "[SUCCESS] EGY siting analysis completed"
else
    echo "[ERROR] EGY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ERI (T3)..."
if $PY process_country_siting.py ERI; then
    echo "[SUCCESS] ERI siting analysis completed"
else
    echo "[ERROR] ERI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ESP (T3)..."
if $PY process_country_siting.py ESP; then
    echo "[SUCCESS] ESP siting analysis completed"
else
    echo "[ERROR] ESP siting analysis failed"
fi

echo "[INFO] Processing siting analysis for EST (T3)..."
if $PY process_country_siting.py EST; then
    echo "[SUCCESS] EST siting analysis completed"
else
    echo "[ERROR] EST siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ETH (T3)..."
if $PY process_country_siting.py ETH; then
    echo "[SUCCESS] ETH siting analysis completed"
else
    echo "[ERROR] ETH siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FIN (T3)..."
if $PY process_country_siting.py FIN; then
    echo "[SUCCESS] FIN siting analysis completed"
else
    echo "[ERROR] FIN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FJI (T3)..."
if $PY process_country_siting.py FJI; then
    echo "[SUCCESS] FJI siting analysis completed"
else
    echo "[ERROR] FJI siting analysis failed"
fi

echo "[INFO] Siting batch 12/24 (T3) completed at $(date)"
