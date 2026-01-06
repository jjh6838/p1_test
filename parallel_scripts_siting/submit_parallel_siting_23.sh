#!/bin/bash --login
#SBATCH --job-name=p23s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_23_%j.out
#SBATCH --error=outputs_per_country/logs/siting_23_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 23/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: SVN, SWE, SWZ, SYC, SYR, TCD, TGO, THA, TJK, TKM"
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

echo "[INFO] Processing siting analysis for SVN (T3)..."
if $PY process_country_siting.py SVN $SCENARIO_FLAG; then
    echo "[SUCCESS] SVN siting analysis completed"
else
    echo "[ERROR] SVN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SWE (T3)..."
if $PY process_country_siting.py SWE $SCENARIO_FLAG; then
    echo "[SUCCESS] SWE siting analysis completed"
else
    echo "[ERROR] SWE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SWZ (T3)..."
if $PY process_country_siting.py SWZ $SCENARIO_FLAG; then
    echo "[SUCCESS] SWZ siting analysis completed"
else
    echo "[ERROR] SWZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SYC (T3)..."
if $PY process_country_siting.py SYC $SCENARIO_FLAG; then
    echo "[SUCCESS] SYC siting analysis completed"
else
    echo "[ERROR] SYC siting analysis failed"
fi

echo "[INFO] Processing siting analysis for SYR (T3)..."
if $PY process_country_siting.py SYR $SCENARIO_FLAG; then
    echo "[SUCCESS] SYR siting analysis completed"
else
    echo "[ERROR] SYR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TCD (T3)..."
if $PY process_country_siting.py TCD $SCENARIO_FLAG; then
    echo "[SUCCESS] TCD siting analysis completed"
else
    echo "[ERROR] TCD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TGO (T3)..."
if $PY process_country_siting.py TGO $SCENARIO_FLAG; then
    echo "[SUCCESS] TGO siting analysis completed"
else
    echo "[ERROR] TGO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for THA (T3)..."
if $PY process_country_siting.py THA $SCENARIO_FLAG; then
    echo "[SUCCESS] THA siting analysis completed"
else
    echo "[ERROR] THA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TJK (T3)..."
if $PY process_country_siting.py TJK $SCENARIO_FLAG; then
    echo "[SUCCESS] TJK siting analysis completed"
else
    echo "[ERROR] TJK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for TKM (T3)..."
if $PY process_country_siting.py TKM $SCENARIO_FLAG; then
    echo "[SUCCESS] TKM siting analysis completed"
else
    echo "[ERROR] TKM siting analysis failed"
fi

echo "[INFO] Siting batch 23/25 (T3) completed at $(date)"
