#!/bin/bash --login
#SBATCH --job-name=p16s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_16_%j.out
#SBATCH --error=outputs_global/logs/siting_16_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 16/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: KOR, KWT, LAO, LBN, LBR, LBY, LCA, LKA, LSO, LTU"
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

echo "[INFO] Processing siting analysis for KOR (T3)..."
if $PY process_country_siting.py KOR; then
    echo "[SUCCESS] KOR siting analysis completed"
else
    echo "[ERROR] KOR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KWT (T3)..."
if $PY process_country_siting.py KWT; then
    echo "[SUCCESS] KWT siting analysis completed"
else
    echo "[ERROR] KWT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LAO (T3)..."
if $PY process_country_siting.py LAO; then
    echo "[SUCCESS] LAO siting analysis completed"
else
    echo "[ERROR] LAO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LBN (T3)..."
if $PY process_country_siting.py LBN; then
    echo "[SUCCESS] LBN siting analysis completed"
else
    echo "[ERROR] LBN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LBR (T3)..."
if $PY process_country_siting.py LBR; then
    echo "[SUCCESS] LBR siting analysis completed"
else
    echo "[ERROR] LBR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LBY (T3)..."
if $PY process_country_siting.py LBY; then
    echo "[SUCCESS] LBY siting analysis completed"
else
    echo "[ERROR] LBY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LCA (T3)..."
if $PY process_country_siting.py LCA; then
    echo "[SUCCESS] LCA siting analysis completed"
else
    echo "[ERROR] LCA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LKA (T3)..."
if $PY process_country_siting.py LKA; then
    echo "[SUCCESS] LKA siting analysis completed"
else
    echo "[ERROR] LKA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LSO (T3)..."
if $PY process_country_siting.py LSO; then
    echo "[SUCCESS] LSO siting analysis completed"
else
    echo "[ERROR] LSO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LTU (T3)..."
if $PY process_country_siting.py LTU; then
    echo "[SUCCESS] LTU siting analysis completed"
else
    echo "[ERROR] LTU siting analysis failed"
fi

echo "[INFO] Siting batch 16/24 (T3) completed at $(date)"
