#!/bin/bash --login
#SBATCH --job-name=p13s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_13_%j.out
#SBATCH --error=outputs_global/logs/siting_13_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 13/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: DNK, DOM, DZA, ECU, EGY, ERI, ESP, EST, ETH, FIN, FJI"
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

# Check scenario flags (passed via sbatch --export)
SCENARIO_FLAG=""
if [ -n "$SUPPLY_FACTOR" ]; then
    SCENARIO_FLAG="--supply-factor $SUPPLY_FACTOR"
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor)"
elif [ "$RUN_ALL_SCENARIOS" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing siting analysis for DNK (T3)..."
if $PY process_country_siting.py DNK $SCENARIO_FLAG; then
    echo "[SUCCESS] DNK siting analysis completed"
else
    echo "[ERROR] DNK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DOM (T3)..."
if $PY process_country_siting.py DOM $SCENARIO_FLAG; then
    echo "[SUCCESS] DOM siting analysis completed"
else
    echo "[ERROR] DOM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DZA (T3)..."
if $PY process_country_siting.py DZA $SCENARIO_FLAG; then
    echo "[SUCCESS] DZA siting analysis completed"
else
    echo "[ERROR] DZA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ECU (T3)..."
if $PY process_country_siting.py ECU $SCENARIO_FLAG; then
    echo "[SUCCESS] ECU siting analysis completed"
else
    echo "[ERROR] ECU siting analysis failed"
fi

echo "[INFO] Processing siting analysis for EGY (T3)..."
if $PY process_country_siting.py EGY $SCENARIO_FLAG; then
    echo "[SUCCESS] EGY siting analysis completed"
else
    echo "[ERROR] EGY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ERI (T3)..."
if $PY process_country_siting.py ERI $SCENARIO_FLAG; then
    echo "[SUCCESS] ERI siting analysis completed"
else
    echo "[ERROR] ERI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ESP (T3)..."
if $PY process_country_siting.py ESP $SCENARIO_FLAG; then
    echo "[SUCCESS] ESP siting analysis completed"
else
    echo "[ERROR] ESP siting analysis failed"
fi

echo "[INFO] Processing siting analysis for EST (T3)..."
if $PY process_country_siting.py EST $SCENARIO_FLAG; then
    echo "[SUCCESS] EST siting analysis completed"
else
    echo "[ERROR] EST siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ETH (T3)..."
if $PY process_country_siting.py ETH $SCENARIO_FLAG; then
    echo "[SUCCESS] ETH siting analysis completed"
else
    echo "[ERROR] ETH siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FIN (T3)..."
if $PY process_country_siting.py FIN $SCENARIO_FLAG; then
    echo "[SUCCESS] FIN siting analysis completed"
else
    echo "[ERROR] FIN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FJI (T3)..."
if $PY process_country_siting.py FJI $SCENARIO_FLAG; then
    echo "[SUCCESS] FJI siting analysis completed"
else
    echo "[ERROR] FJI siting analysis failed"
fi

echo "[INFO] Siting batch 13/25 (T3) completed at $(date)"
