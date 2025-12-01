#!/bin/bash --login
#SBATCH --job-name=siting_17_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_17_%j.out
#SBATCH --error=outputs_global/logs/siting_17_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 17/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: LUX, LVA, MAR, MDA, MDG, MDV, MKD, MLI, MLT, MMR"
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

echo "[INFO] Processing siting analysis for LUX (T3)..."
$PY process_country_siting.py LUX
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LUX siting analysis completed"
else
    echo "[ERROR] LUX siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LVA (T3)..."
$PY process_country_siting.py LVA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] LVA siting analysis completed"
else
    echo "[ERROR] LVA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MAR (T3)..."
$PY process_country_siting.py MAR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MAR siting analysis completed"
else
    echo "[ERROR] MAR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDA (T3)..."
$PY process_country_siting.py MDA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDA siting analysis completed"
else
    echo "[ERROR] MDA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDG (T3)..."
$PY process_country_siting.py MDG
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDG siting analysis completed"
else
    echo "[ERROR] MDG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDV (T3)..."
$PY process_country_siting.py MDV
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MDV siting analysis completed"
else
    echo "[ERROR] MDV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MKD (T3)..."
$PY process_country_siting.py MKD
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MKD siting analysis completed"
else
    echo "[ERROR] MKD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MLI (T3)..."
$PY process_country_siting.py MLI
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MLI siting analysis completed"
else
    echo "[ERROR] MLI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MLT (T3)..."
$PY process_country_siting.py MLT
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MLT siting analysis completed"
else
    echo "[ERROR] MLT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MMR (T3)..."
$PY process_country_siting.py MMR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] MMR siting analysis completed"
else
    echo "[ERROR] MMR siting analysis failed"
fi

echo "[INFO] Siting batch 17/24 (T3) completed at $(date)"
