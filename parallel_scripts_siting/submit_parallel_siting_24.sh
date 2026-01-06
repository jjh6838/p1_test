#!/bin/bash --login
#SBATCH --job-name=p24s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_24_%j.out
#SBATCH --error=outputs_per_country/logs/siting_24_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 24/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: TLS, TON, TTO, TUN, TUR, TWN, TZA, UGA, UKR, URY"
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

echo "[INFO] Processing siting analysis for TLS (T3)..."
if $PY process_country_siting.py TLS $SCENARIO_FLAG; then
    echo "[SUCCESS] TLS siting analysis completed"
else
    echo "[ERROR] TLS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TON (T3)..."
if $PY process_country_siting.py TON $SCENARIO_FLAG; then
    echo "[SUCCESS] TON siting analysis completed"
else
    echo "[ERROR] TON siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TTO (T3)..."
if $PY process_country_siting.py TTO $SCENARIO_FLAG; then
    echo "[SUCCESS] TTO siting analysis completed"
else
    echo "[ERROR] TTO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TUN (T3)..."
if $PY process_country_siting.py TUN $SCENARIO_FLAG; then
    echo "[SUCCESS] TUN siting analysis completed"
else
    echo "[ERROR] TUN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TUR (T3)..."
if $PY process_country_siting.py TUR $SCENARIO_FLAG; then
    echo "[SUCCESS] TUR siting analysis completed"
else
    echo "[ERROR] TUR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TWN (T3)..."
if $PY process_country_siting.py TWN $SCENARIO_FLAG; then
    echo "[SUCCESS] TWN siting analysis completed"
else
    echo "[ERROR] TWN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TZA (T3)..."
if $PY process_country_siting.py TZA $SCENARIO_FLAG; then
    echo "[SUCCESS] TZA siting analysis completed"
else
    echo "[ERROR] TZA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for UGA (T3)..."
if $PY process_country_siting.py UGA $SCENARIO_FLAG; then
    echo "[SUCCESS] UGA siting analysis completed"
else
    echo "[ERROR] UGA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for UKR (T3)..."
if $PY process_country_siting.py UKR $SCENARIO_FLAG; then
    echo "[SUCCESS] UKR siting analysis completed"
else
    echo "[ERROR] UKR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for URY (T3)..."
if $PY process_country_siting.py URY $SCENARIO_FLAG; then
    echo "[SUCCESS] URY siting analysis completed"
else
    echo "[ERROR] URY siting analysis failed"
fi

echo "[INFO] Siting batch 24/25 (T3) completed at $(date)"
