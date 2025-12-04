#!/bin/bash --login
#SBATCH --job-name=siting_09_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_09_%j.out
#SBATCH --error=outputs_global/logs/siting_09_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 9/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: BEL, BEN, BFA, BGD, BGR, BHR, BHS, BIH, BLR, BLZ, BMU"
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

echo "[INFO] Processing siting analysis for BEL (T3)..."
if $PY process_country_siting.py BEL; then
    echo "[SUCCESS] BEL siting analysis completed"
else
    echo "[ERROR] BEL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BEN (T3)..."
if $PY process_country_siting.py BEN; then
    echo "[SUCCESS] BEN siting analysis completed"
else
    echo "[ERROR] BEN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BFA (T3)..."
if $PY process_country_siting.py BFA; then
    echo "[SUCCESS] BFA siting analysis completed"
else
    echo "[ERROR] BFA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BGD (T3)..."
if $PY process_country_siting.py BGD; then
    echo "[SUCCESS] BGD siting analysis completed"
else
    echo "[ERROR] BGD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BGR (T3)..."
if $PY process_country_siting.py BGR; then
    echo "[SUCCESS] BGR siting analysis completed"
else
    echo "[ERROR] BGR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BHR (T3)..."
if $PY process_country_siting.py BHR; then
    echo "[SUCCESS] BHR siting analysis completed"
else
    echo "[ERROR] BHR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BHS (T3)..."
if $PY process_country_siting.py BHS; then
    echo "[SUCCESS] BHS siting analysis completed"
else
    echo "[ERROR] BHS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BIH (T3)..."
if $PY process_country_siting.py BIH; then
    echo "[SUCCESS] BIH siting analysis completed"
else
    echo "[ERROR] BIH siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BLR (T3)..."
if $PY process_country_siting.py BLR; then
    echo "[SUCCESS] BLR siting analysis completed"
else
    echo "[ERROR] BLR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BLZ (T3)..."
if $PY process_country_siting.py BLZ; then
    echo "[SUCCESS] BLZ siting analysis completed"
else
    echo "[ERROR] BLZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BMU (T3)..."
if $PY process_country_siting.py BMU; then
    echo "[SUCCESS] BMU siting analysis completed"
else
    echo "[ERROR] BMU siting analysis failed"
fi

echo "[INFO] Siting batch 9/24 (T3) completed at $(date)"
