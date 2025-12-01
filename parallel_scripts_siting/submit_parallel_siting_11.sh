#!/bin/bash --login
#SBATCH --job-name=siting_11_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_11_%j.out
#SBATCH --error=outputs_global/logs/siting_11_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 11/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: COG, COL, COM, CPV, CRI, CUB, CYM, CYP, CZE, DEU, DJI"
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

echo "[INFO] Processing siting analysis for COG (T3)..."
$PY process_country_siting.py COG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COG siting analysis completed"
else
    echo "[ERROR] COG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for COL (T3)..."
$PY process_country_siting.py COL
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COL siting analysis completed"
else
    echo "[ERROR] COL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for COM (T3)..."
$PY process_country_siting.py COM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] COM siting analysis completed"
else
    echo "[ERROR] COM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CPV (T3)..."
$PY process_country_siting.py CPV
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CPV siting analysis completed"
else
    echo "[ERROR] CPV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CRI (T3)..."
$PY process_country_siting.py CRI
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CRI siting analysis completed"
else
    echo "[ERROR] CRI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CUB (T3)..."
$PY process_country_siting.py CUB
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CUB siting analysis completed"
else
    echo "[ERROR] CUB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CYM (T3)..."
$PY process_country_siting.py CYM
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CYM siting analysis completed"
else
    echo "[ERROR] CYM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CYP (T3)..."
$PY process_country_siting.py CYP
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CYP siting analysis completed"
else
    echo "[ERROR] CYP siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CZE (T3)..."
$PY process_country_siting.py CZE
if [ $? -eq 0 ]; then
    echo "[SUCCESS] CZE siting analysis completed"
else
    echo "[ERROR] CZE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DEU (T3)..."
$PY process_country_siting.py DEU
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DEU siting analysis completed"
else
    echo "[ERROR] DEU siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DJI (T3)..."
$PY process_country_siting.py DJI
if [ $? -eq 0 ]; then
    echo "[SUCCESS] DJI siting analysis completed"
else
    echo "[ERROR] DJI siting analysis failed"
fi

echo "[INFO] Siting batch 11/24 (T3) completed at $(date)"
