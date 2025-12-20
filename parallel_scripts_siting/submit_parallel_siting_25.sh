#!/bin/bash --login
#SBATCH --job-name=p25s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_25_%j.out
#SBATCH --error=outputs_per_country/logs/siting_25_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 25/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: URY, UZB, VIR, VNM, VUT, WSM, YEM, ZAF, ZMB, ZWE"
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

echo "[INFO] Processing siting analysis for URY (T3)..."
if $PY process_country_siting.py URY $SCENARIO_FLAG; then
    echo "[SUCCESS] URY siting analysis completed"
else
    echo "[ERROR] URY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for UZB (T3)..."
if $PY process_country_siting.py UZB $SCENARIO_FLAG; then
    echo "[SUCCESS] UZB siting analysis completed"
else
    echo "[ERROR] UZB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VIR (T3)..."
if $PY process_country_siting.py VIR $SCENARIO_FLAG; then
    echo "[SUCCESS] VIR siting analysis completed"
else
    echo "[ERROR] VIR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VNM (T3)..."
if $PY process_country_siting.py VNM $SCENARIO_FLAG; then
    echo "[SUCCESS] VNM siting analysis completed"
else
    echo "[ERROR] VNM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for VUT (T3)..."
if $PY process_country_siting.py VUT $SCENARIO_FLAG; then
    echo "[SUCCESS] VUT siting analysis completed"
else
    echo "[ERROR] VUT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for WSM (T3)..."
if $PY process_country_siting.py WSM $SCENARIO_FLAG; then
    echo "[SUCCESS] WSM siting analysis completed"
else
    echo "[ERROR] WSM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for YEM (T3)..."
if $PY process_country_siting.py YEM $SCENARIO_FLAG; then
    echo "[SUCCESS] YEM siting analysis completed"
else
    echo "[ERROR] YEM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZAF (T3)..."
if $PY process_country_siting.py ZAF $SCENARIO_FLAG; then
    echo "[SUCCESS] ZAF siting analysis completed"
else
    echo "[ERROR] ZAF siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZMB (T3)..."
if $PY process_country_siting.py ZMB $SCENARIO_FLAG; then
    echo "[SUCCESS] ZMB siting analysis completed"
else
    echo "[ERROR] ZMB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ZWE (T3)..."
if $PY process_country_siting.py ZWE $SCENARIO_FLAG; then
    echo "[SUCCESS] ZWE siting analysis completed"
else
    echo "[ERROR] ZWE siting analysis failed"
fi

echo "[INFO] Siting batch 25/25 (T3) completed at $(date)"
