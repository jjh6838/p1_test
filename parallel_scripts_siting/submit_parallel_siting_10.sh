#!/bin/bash --login
#SBATCH --job-name=siting_10_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_10_%j.out
#SBATCH --error=outputs_global/logs/siting_10_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 10/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: BOL, BRB, BRN, BTN, BWA, CAF, CHE, CHL, CIV, CMR, COD"
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

echo "[INFO] Processing siting analysis for BOL (T3)..."
if $PY process_country_siting.py BOL; then
    echo "[SUCCESS] BOL siting analysis completed"
else
    echo "[ERROR] BOL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BRB (T3)..."
if $PY process_country_siting.py BRB; then
    echo "[SUCCESS] BRB siting analysis completed"
else
    echo "[ERROR] BRB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BRN (T3)..."
if $PY process_country_siting.py BRN; then
    echo "[SUCCESS] BRN siting analysis completed"
else
    echo "[ERROR] BRN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BTN (T3)..."
if $PY process_country_siting.py BTN; then
    echo "[SUCCESS] BTN siting analysis completed"
else
    echo "[ERROR] BTN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BWA (T3)..."
if $PY process_country_siting.py BWA; then
    echo "[SUCCESS] BWA siting analysis completed"
else
    echo "[ERROR] BWA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CAF (T3)..."
if $PY process_country_siting.py CAF; then
    echo "[SUCCESS] CAF siting analysis completed"
else
    echo "[ERROR] CAF siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CHE (T3)..."
if $PY process_country_siting.py CHE; then
    echo "[SUCCESS] CHE siting analysis completed"
else
    echo "[ERROR] CHE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CHL (T3)..."
if $PY process_country_siting.py CHL; then
    echo "[SUCCESS] CHL siting analysis completed"
else
    echo "[ERROR] CHL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CIV (T3)..."
if $PY process_country_siting.py CIV; then
    echo "[SUCCESS] CIV siting analysis completed"
else
    echo "[ERROR] CIV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CMR (T3)..."
if $PY process_country_siting.py CMR; then
    echo "[SUCCESS] CMR siting analysis completed"
else
    echo "[ERROR] CMR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for COD (T3)..."
if $PY process_country_siting.py COD; then
    echo "[SUCCESS] COD siting analysis completed"
else
    echo "[ERROR] COD siting analysis failed"
fi

echo "[INFO] Siting batch 10/24 (T3) completed at $(date)"
