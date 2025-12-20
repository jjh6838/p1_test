#!/bin/bash --login
#SBATCH --job-name=p10s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_10_%j.out
#SBATCH --error=outputs_per_country/logs/siting_10_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 10/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: BEL, BEN, BFA, BGD, BGR, BHR, BHS, BIH, BLR, BLZ, BMU"
echo "[INFO] Tier: T3 | Memory: 25G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country/logs outputs_global

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
# Use ${VAR:-} syntax to avoid 'unbound variable' errors with set -u
SCENARIO_FLAG=""
if [ -n "${SUPPLY_FACTOR:-}" ]; then
    SCENARIO_FLAG="--supply-factor $SUPPLY_FACTOR"
    echo "[INFO] Running single scenario: ${SUPPLY_FACTOR} (supply factor)"
elif [ "${RUN_ALL_SCENARIOS:-}" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing siting analysis for BEL (T3)..."
if $PY process_country_siting.py BEL $SCENARIO_FLAG; then
    echo "[SUCCESS] BEL siting analysis completed"
else
    echo "[ERROR] BEL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BEN (T3)..."
if $PY process_country_siting.py BEN $SCENARIO_FLAG; then
    echo "[SUCCESS] BEN siting analysis completed"
else
    echo "[ERROR] BEN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BFA (T3)..."
if $PY process_country_siting.py BFA $SCENARIO_FLAG; then
    echo "[SUCCESS] BFA siting analysis completed"
else
    echo "[ERROR] BFA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BGD (T3)..."
if $PY process_country_siting.py BGD $SCENARIO_FLAG; then
    echo "[SUCCESS] BGD siting analysis completed"
else
    echo "[ERROR] BGD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BGR (T3)..."
if $PY process_country_siting.py BGR $SCENARIO_FLAG; then
    echo "[SUCCESS] BGR siting analysis completed"
else
    echo "[ERROR] BGR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BHR (T3)..."
if $PY process_country_siting.py BHR $SCENARIO_FLAG; then
    echo "[SUCCESS] BHR siting analysis completed"
else
    echo "[ERROR] BHR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BHS (T3)..."
if $PY process_country_siting.py BHS $SCENARIO_FLAG; then
    echo "[SUCCESS] BHS siting analysis completed"
else
    echo "[ERROR] BHS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BIH (T3)..."
if $PY process_country_siting.py BIH $SCENARIO_FLAG; then
    echo "[SUCCESS] BIH siting analysis completed"
else
    echo "[ERROR] BIH siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BLR (T3)..."
if $PY process_country_siting.py BLR $SCENARIO_FLAG; then
    echo "[SUCCESS] BLR siting analysis completed"
else
    echo "[ERROR] BLR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BLZ (T3)..."
if $PY process_country_siting.py BLZ $SCENARIO_FLAG; then
    echo "[SUCCESS] BLZ siting analysis completed"
else
    echo "[ERROR] BLZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BMU (T3)..."
if $PY process_country_siting.py BMU $SCENARIO_FLAG; then
    echo "[SUCCESS] BMU siting analysis completed"
else
    echo "[ERROR] BMU siting analysis failed"
fi

echo "[INFO] Siting batch 10/25 (T3) completed at $(date)"
