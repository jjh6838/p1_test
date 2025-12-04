#!/bin/bash --login
#SBATCH --job-name=p14s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_14_%j.out
#SBATCH --error=outputs_global/logs/siting_14_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 14/24 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: GRL, GTM, GUM, GUY, HND, HRV, HUN, IRL, IRN, IRQ, ISL"
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

echo "[INFO] Processing siting analysis for GRL (T3)..."
if $PY process_country_siting.py GRL; then
    echo "[SUCCESS] GRL siting analysis completed"
else
    echo "[ERROR] GRL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GTM (T3)..."
if $PY process_country_siting.py GTM; then
    echo "[SUCCESS] GTM siting analysis completed"
else
    echo "[ERROR] GTM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GUM (T3)..."
if $PY process_country_siting.py GUM; then
    echo "[SUCCESS] GUM siting analysis completed"
else
    echo "[ERROR] GUM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GUY (T3)..."
if $PY process_country_siting.py GUY; then
    echo "[SUCCESS] GUY siting analysis completed"
else
    echo "[ERROR] GUY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for HND (T3)..."
if $PY process_country_siting.py HND; then
    echo "[SUCCESS] HND siting analysis completed"
else
    echo "[ERROR] HND siting analysis failed"
fi

echo "[INFO] Processing siting analysis for HRV (T3)..."
if $PY process_country_siting.py HRV; then
    echo "[SUCCESS] HRV siting analysis completed"
else
    echo "[ERROR] HRV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for HUN (T3)..."
if $PY process_country_siting.py HUN; then
    echo "[SUCCESS] HUN siting analysis completed"
else
    echo "[ERROR] HUN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for IRL (T3)..."
if $PY process_country_siting.py IRL; then
    echo "[SUCCESS] IRL siting analysis completed"
else
    echo "[ERROR] IRL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for IRN (T3)..."
if $PY process_country_siting.py IRN; then
    echo "[SUCCESS] IRN siting analysis completed"
else
    echo "[ERROR] IRN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for IRQ (T3)..."
if $PY process_country_siting.py IRQ; then
    echo "[SUCCESS] IRQ siting analysis completed"
else
    echo "[ERROR] IRQ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ISL (T3)..."
if $PY process_country_siting.py ISL; then
    echo "[SUCCESS] ISL siting analysis completed"
else
    echo "[ERROR] ISL siting analysis failed"
fi

echo "[INFO] Siting batch 14/24 (T3) completed at $(date)"
