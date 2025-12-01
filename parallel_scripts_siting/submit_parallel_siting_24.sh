#!/bin/bash --login
#SBATCH --job-name=siting_24_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_24_%j.out
#SBATCH --error=outputs_global/logs/siting_24_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 24/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: URY, UZB, VIR, VNM, VUT, WSM, YEM, ZAF, ZMB, ZWE"
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

echo "[INFO] Processing siting analysis for URY (T3)..."
$PY process_country_siting.py URY
if [ $? -eq 0 ]; then
    echo "[SUCCESS] URY siting analysis completed"
else
    echo "[ERROR] URY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for UZB (T3)..."
$PY process_country_siting.py UZB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UZB siting analysis completed"
else
    echo "[ERROR] UZB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VIR (T3)..."
$PY process_country_siting.py VIR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VIR siting analysis completed"
else
    echo "[ERROR] VIR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VNM (T3)..."
$PY process_country_siting.py VNM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VNM siting analysis completed"
else
    echo "[ERROR] VNM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VUT (T3)..."
$PY process_country_siting.py VUT
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VUT siting analysis completed"
else
    echo "[ERROR] VUT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for WSM (T3)..."
$PY process_country_siting.py WSM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] WSM siting analysis completed"
else
    echo "[ERROR] WSM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for YEM (T3)..."
$PY process_country_siting.py YEM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] YEM siting analysis completed"
else
    echo "[ERROR] YEM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZAF (T3)..."
$PY process_country_siting.py ZAF
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZAF siting analysis completed"
else
    echo "[ERROR] ZAF siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZMB (T3)..."
$PY process_country_siting.py ZMB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZMB siting analysis completed"
else
    echo "[ERROR] ZMB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZWE (T3)..."
$PY process_country_siting.py ZWE
if [ $? -eq 0 ]; then
    echo "[SUCCESS] ZWE siting analysis completed"
else
    echo "[ERROR] ZWE siting analysis failed"
fi

echo "[INFO] Siting batch 24/24 (T3) completed at $(date)"
