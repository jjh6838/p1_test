#!/bin/bash --login
#SBATCH --job-name=siting_21_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_21_%j.out
#SBATCH --error=outputs_global/logs/siting_21_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 21/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: RWA, SDN, SEN, SGP, SLE, SLV, SOM, SRB, SSD, SUR"
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

echo "[INFO] Processing siting analysis for RWA (T3)..."
$PY process_country_siting.py RWA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] RWA siting analysis completed"
else
    echo "[ERROR] RWA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SDN (T3)..."
$PY process_country_siting.py SDN
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SDN siting analysis completed"
else
    echo "[ERROR] SDN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SEN (T3)..."
$PY process_country_siting.py SEN
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SEN siting analysis completed"
else
    echo "[ERROR] SEN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SGP (T3)..."
$PY process_country_siting.py SGP
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SGP siting analysis completed"
else
    echo "[ERROR] SGP siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SLE (T3)..."
$PY process_country_siting.py SLE
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLE siting analysis completed"
else
    echo "[ERROR] SLE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SLV (T3)..."
$PY process_country_siting.py SLV
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLV siting analysis completed"
else
    echo "[ERROR] SLV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SOM (T3)..."
$PY process_country_siting.py SOM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SOM siting analysis completed"
else
    echo "[ERROR] SOM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SRB (T3)..."
$PY process_country_siting.py SRB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SRB siting analysis completed"
else
    echo "[ERROR] SRB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SSD (T3)..."
$PY process_country_siting.py SSD
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SSD siting analysis completed"
else
    echo "[ERROR] SSD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SUR (T3)..."
$PY process_country_siting.py SUR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SUR siting analysis completed"
else
    echo "[ERROR] SUR siting analysis failed"
fi

echo "[INFO] Siting batch 21/24 (T3) completed at $(date)"
