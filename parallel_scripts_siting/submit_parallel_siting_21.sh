#!/bin/bash --login
#SBATCH --job-name=p21s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_21_%j.out
#SBATCH --error=outputs_per_country/logs/siting_21_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 21/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: PNG, POL, PRI, PRK, PRT, PRY, PSE, QAT, ROU, RWA"
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

echo "[INFO] Processing siting analysis for PNG (T3)..."
if $PY process_country_siting.py PNG $SCENARIO_FLAG; then
    echo "[SUCCESS] PNG siting analysis completed"
else
    echo "[ERROR] PNG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for POL (T3)..."
if $PY process_country_siting.py POL $SCENARIO_FLAG; then
    echo "[SUCCESS] POL siting analysis completed"
else
    echo "[ERROR] POL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRI (T3)..."
if $PY process_country_siting.py PRI $SCENARIO_FLAG; then
    echo "[SUCCESS] PRI siting analysis completed"
else
    echo "[ERROR] PRI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRK (T3)..."
if $PY process_country_siting.py PRK $SCENARIO_FLAG; then
    echo "[SUCCESS] PRK siting analysis completed"
else
    echo "[ERROR] PRK siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRT (T3)..."
if $PY process_country_siting.py PRT $SCENARIO_FLAG; then
    echo "[SUCCESS] PRT siting analysis completed"
else
    echo "[ERROR] PRT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PRY (T3)..."
if $PY process_country_siting.py PRY $SCENARIO_FLAG; then
    echo "[SUCCESS] PRY siting analysis completed"
else
    echo "[ERROR] PRY siting analysis failed"
fi

echo "[INFO] Processing siting analysis for PSE (T3)..."
if $PY process_country_siting.py PSE $SCENARIO_FLAG; then
    echo "[SUCCESS] PSE siting analysis completed"
else
    echo "[ERROR] PSE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for QAT (T3)..."
if $PY process_country_siting.py QAT $SCENARIO_FLAG; then
    echo "[SUCCESS] QAT siting analysis completed"
else
    echo "[ERROR] QAT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ROU (T3)..."
if $PY process_country_siting.py ROU $SCENARIO_FLAG; then
    echo "[SUCCESS] ROU siting analysis completed"
else
    echo "[ERROR] ROU siting analysis failed"
fi

echo "[INFO] Processing siting analysis for RWA (T3)..."
if $PY process_country_siting.py RWA $SCENARIO_FLAG; then
    echo "[SUCCESS] RWA siting analysis completed"
else
    echo "[ERROR] RWA siting analysis failed"
fi

echo "[INFO] Siting batch 21/25 (T3) completed at $(date)"
