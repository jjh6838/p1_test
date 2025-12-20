#!/bin/bash --login
#SBATCH --job-name=p14s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_14_%j.out
#SBATCH --error=outputs_global/logs/siting_14_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 14/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: FRA, FRO, GAB, GBR, GEO, GHA, GIN, GMB, GNB, GNQ, GRC"
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

echo "[INFO] Processing siting analysis for FRA (T3)..."
if $PY process_country_siting.py FRA $SCENARIO_FLAG; then
    echo "[SUCCESS] FRA siting analysis completed"
else
    echo "[ERROR] FRA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for FRO (T3)..."
if $PY process_country_siting.py FRO $SCENARIO_FLAG; then
    echo "[SUCCESS] FRO siting analysis completed"
else
    echo "[ERROR] FRO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GAB (T3)..."
if $PY process_country_siting.py GAB $SCENARIO_FLAG; then
    echo "[SUCCESS] GAB siting analysis completed"
else
    echo "[ERROR] GAB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GBR (T3)..."
if $PY process_country_siting.py GBR $SCENARIO_FLAG; then
    echo "[SUCCESS] GBR siting analysis completed"
else
    echo "[ERROR] GBR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GEO (T3)..."
if $PY process_country_siting.py GEO $SCENARIO_FLAG; then
    echo "[SUCCESS] GEO siting analysis completed"
else
    echo "[ERROR] GEO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GHA (T3)..."
if $PY process_country_siting.py GHA $SCENARIO_FLAG; then
    echo "[SUCCESS] GHA siting analysis completed"
else
    echo "[ERROR] GHA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GIN (T3)..."
if $PY process_country_siting.py GIN $SCENARIO_FLAG; then
    echo "[SUCCESS] GIN siting analysis completed"
else
    echo "[ERROR] GIN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GMB (T3)..."
if $PY process_country_siting.py GMB $SCENARIO_FLAG; then
    echo "[SUCCESS] GMB siting analysis completed"
else
    echo "[ERROR] GMB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GNB (T3)..."
if $PY process_country_siting.py GNB $SCENARIO_FLAG; then
    echo "[SUCCESS] GNB siting analysis completed"
else
    echo "[ERROR] GNB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GNQ (T3)..."
if $PY process_country_siting.py GNQ $SCENARIO_FLAG; then
    echo "[SUCCESS] GNQ siting analysis completed"
else
    echo "[ERROR] GNQ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for GRC (T3)..."
if $PY process_country_siting.py GRC $SCENARIO_FLAG; then
    echo "[SUCCESS] GRC siting analysis completed"
else
    echo "[ERROR] GRC siting analysis failed"
fi

echo "[INFO] Siting batch 14/25 (T3) completed at $(date)"
