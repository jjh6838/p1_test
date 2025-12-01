#!/bin/bash --login
#SBATCH --job-name=siting_13_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_13_%j.out
#SBATCH --error=outputs_global/logs/siting_13_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 13/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: FRA, FRO, GAB, GBR, GEO, GHA, GIN, GMB, GNB, GNQ, GRC"
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

echo "[INFO] Processing siting analysis for FRA (T3)..."
$PY process_country_siting.py FRA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FRA siting analysis completed"
else
    echo "[ERROR] FRA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FRO (T3)..."
$PY process_country_siting.py FRO
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FRO siting analysis completed"
else
    echo "[ERROR] FRO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GAB (T3)..."
$PY process_country_siting.py GAB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GAB siting analysis completed"
else
    echo "[ERROR] GAB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GBR (T3)..."
$PY process_country_siting.py GBR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GBR siting analysis completed"
else
    echo "[ERROR] GBR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GEO (T3)..."
$PY process_country_siting.py GEO
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GEO siting analysis completed"
else
    echo "[ERROR] GEO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GHA (T3)..."
$PY process_country_siting.py GHA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GHA siting analysis completed"
else
    echo "[ERROR] GHA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GIN (T3)..."
$PY process_country_siting.py GIN
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GIN siting analysis completed"
else
    echo "[ERROR] GIN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GMB (T3)..."
$PY process_country_siting.py GMB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GMB siting analysis completed"
else
    echo "[ERROR] GMB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GNB (T3)..."
$PY process_country_siting.py GNB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GNB siting analysis completed"
else
    echo "[ERROR] GNB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GNQ (T3)..."
$PY process_country_siting.py GNQ
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GNQ siting analysis completed"
else
    echo "[ERROR] GNQ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GRC (T3)..."
$PY process_country_siting.py GRC
if [ $? -eq 0 ]; then
    echo "[SUCCESS] GRC siting analysis completed"
else
    echo "[ERROR] GRC siting analysis failed"
fi

echo "[INFO] Siting batch 13/24 (T3) completed at $(date)"
