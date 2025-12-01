#!/bin/bash --login
#SBATCH --job-name=siting_22_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_22_%j.out
#SBATCH --error=outputs_global/logs/siting_22_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 22/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: SVK, SVN, SWE, SWZ, SYC, SYR, TCD, TGO, THA, TJK"
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

echo "[INFO] Processing siting analysis for SVK (T3)..."
$PY process_country_siting.py SVK
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVK siting analysis completed"
else
    echo "[ERROR] SVK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SVN (T3)..."
$PY process_country_siting.py SVN
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVN siting analysis completed"
else
    echo "[ERROR] SVN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SWE (T3)..."
$PY process_country_siting.py SWE
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SWE siting analysis completed"
else
    echo "[ERROR] SWE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SWZ (T3)..."
$PY process_country_siting.py SWZ
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SWZ siting analysis completed"
else
    echo "[ERROR] SWZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SYC (T3)..."
$PY process_country_siting.py SYC
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYC siting analysis completed"
else
    echo "[ERROR] SYC siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SYR (T3)..."
$PY process_country_siting.py SYR
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYR siting analysis completed"
else
    echo "[ERROR] SYR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TCD (T3)..."
$PY process_country_siting.py TCD
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TCD siting analysis completed"
else
    echo "[ERROR] TCD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TGO (T3)..."
$PY process_country_siting.py TGO
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TGO siting analysis completed"
else
    echo "[ERROR] TGO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for THA (T3)..."
$PY process_country_siting.py THA
if [ $? -eq 0 ]; then
    echo "[SUCCESS] THA siting analysis completed"
else
    echo "[ERROR] THA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TJK (T3)..."
$PY process_country_siting.py TJK
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TJK siting analysis completed"
else
    echo "[ERROR] TJK siting analysis failed"
fi

echo "[INFO] Siting batch 22/24 (T3) completed at $(date)"
