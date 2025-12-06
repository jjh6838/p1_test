#!/bin/bash --login
#SBATCH --job-name=p20s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_20_%j.out
#SBATCH --error=outputs_global/logs/siting_20_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 20/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: PHL, PNG, POL, PRI, PRK, PRT, PRY, PSE, QAT, ROU"
echo "[INFO] Tier: T3 | Memory: 25G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing siting analysis for PHL (T3)..."
if $PY process_country_siting.py PHL; then
    echo "[SUCCESS] PHL siting analysis completed"
else
    echo "[ERROR] PHL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PNG (T3)..."
if $PY process_country_siting.py PNG; then
    echo "[SUCCESS] PNG siting analysis completed"
else
    echo "[ERROR] PNG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for POL (T3)..."
if $PY process_country_siting.py POL; then
    echo "[SUCCESS] POL siting analysis completed"
else
    echo "[ERROR] POL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRI (T3)..."
if $PY process_country_siting.py PRI; then
    echo "[SUCCESS] PRI siting analysis completed"
else
    echo "[ERROR] PRI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRK (T3)..."
if $PY process_country_siting.py PRK; then
    echo "[SUCCESS] PRK siting analysis completed"
else
    echo "[ERROR] PRK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRT (T3)..."
if $PY process_country_siting.py PRT; then
    echo "[SUCCESS] PRT siting analysis completed"
else
    echo "[ERROR] PRT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRY (T3)..."
if $PY process_country_siting.py PRY; then
    echo "[SUCCESS] PRY siting analysis completed"
else
    echo "[ERROR] PRY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PSE (T3)..."
if $PY process_country_siting.py PSE; then
    echo "[SUCCESS] PSE siting analysis completed"
else
    echo "[ERROR] PSE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for QAT (T3)..."
if $PY process_country_siting.py QAT; then
    echo "[SUCCESS] QAT siting analysis completed"
else
    echo "[ERROR] QAT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ROU (T3)..."
if $PY process_country_siting.py ROU; then
    echo "[SUCCESS] ROU siting analysis completed"
else
    echo "[ERROR] ROU siting analysis failed"
fi

echo "[INFO] Siting batch 20/24 (T3) completed at $(date)"
