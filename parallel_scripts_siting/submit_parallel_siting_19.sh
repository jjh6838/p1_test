#!/bin/bash --login
#SBATCH --job-name=p19s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_19_%j.out
#SBATCH --error=outputs_global/logs/siting_19_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 19/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: NGA, NIC, NLD, NOR, NPL, NZL, OMN, PAK, PAN, PER"
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

echo "[INFO] Processing siting analysis for NGA (T3)..."
if $PY process_country_siting.py NGA; then
    echo "[SUCCESS] NGA siting analysis completed"
else
    echo "[ERROR] NGA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NIC (T3)..."
if $PY process_country_siting.py NIC; then
    echo "[SUCCESS] NIC siting analysis completed"
else
    echo "[ERROR] NIC siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NLD (T3)..."
if $PY process_country_siting.py NLD; then
    echo "[SUCCESS] NLD siting analysis completed"
else
    echo "[ERROR] NLD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NOR (T3)..."
if $PY process_country_siting.py NOR; then
    echo "[SUCCESS] NOR siting analysis completed"
else
    echo "[ERROR] NOR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NPL (T3)..."
if $PY process_country_siting.py NPL; then
    echo "[SUCCESS] NPL siting analysis completed"
else
    echo "[ERROR] NPL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NZL (T3)..."
if $PY process_country_siting.py NZL; then
    echo "[SUCCESS] NZL siting analysis completed"
else
    echo "[ERROR] NZL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for OMN (T3)..."
if $PY process_country_siting.py OMN; then
    echo "[SUCCESS] OMN siting analysis completed"
else
    echo "[ERROR] OMN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PAK (T3)..."
if $PY process_country_siting.py PAK; then
    echo "[SUCCESS] PAK siting analysis completed"
else
    echo "[ERROR] PAK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PAN (T3)..."
if $PY process_country_siting.py PAN; then
    echo "[SUCCESS] PAN siting analysis completed"
else
    echo "[ERROR] PAN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PER (T3)..."
if $PY process_country_siting.py PER; then
    echo "[SUCCESS] PER siting analysis completed"
else
    echo "[ERROR] PER siting analysis failed"
fi

echo "[INFO] Siting batch 19/24 (T3) completed at $(date)"
