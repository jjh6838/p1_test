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

echo "[INFO] Starting siting analysis script 16/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: ISR, ITA, JAM, JOR, JPN, KEN, KGZ, KHM, KIR, KNA"
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

echo "[INFO] Processing siting analysis for ISR (T3)..."
if $PY process_country_siting.py ISR; then
    echo "[SUCCESS] ISR siting analysis completed"
else
    echo "[ERROR] ISR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ITA (T3)..."
if $PY process_country_siting.py ITA; then
    echo "[SUCCESS] ITA siting analysis completed"
else
    echo "[ERROR] ITA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JAM (T3)..."
if $PY process_country_siting.py JAM; then
    echo "[SUCCESS] JAM siting analysis completed"
else
    echo "[ERROR] JAM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JOR (T3)..."
if $PY process_country_siting.py JOR; then
    echo "[SUCCESS] JOR siting analysis completed"
else
    echo "[ERROR] JOR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JPN (T3)..."
if $PY process_country_siting.py JPN; then
    echo "[SUCCESS] JPN siting analysis completed"
else
    echo "[ERROR] JPN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KEN (T3)..."
if $PY process_country_siting.py KEN; then
    echo "[SUCCESS] KEN siting analysis completed"
else
    echo "[ERROR] KEN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KGZ (T3)..."
if $PY process_country_siting.py KGZ; then
    echo "[SUCCESS] KGZ siting analysis completed"
else
    echo "[ERROR] KGZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KHM (T3)..."
if $PY process_country_siting.py KHM; then
    echo "[SUCCESS] KHM siting analysis completed"
else
    echo "[ERROR] KHM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KIR (T3)..."
if $PY process_country_siting.py KIR; then
    echo "[SUCCESS] KIR siting analysis completed"
else
    echo "[ERROR] KIR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KNA (T3)..."
if $PY process_country_siting.py KNA; then
    echo "[SUCCESS] KNA siting analysis completed"
else
    echo "[ERROR] KNA siting analysis failed"
fi

echo "[INFO] Siting batch 16/25 (T3) completed at $(date)"
