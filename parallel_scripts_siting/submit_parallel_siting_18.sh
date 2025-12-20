#!/bin/bash --login
#SBATCH --job-name=p18s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_18_%j.out
#SBATCH --error=outputs_global/logs/siting_18_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 18/25 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: LUX, LVA, MAR, MDA, MDG, MDV, MKD, MLI, MLT, MMR"
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

echo "[INFO] Processing siting analysis for LUX (T3)..."
if $PY process_country_siting.py LUX $SCENARIO_FLAG; then
    echo "[SUCCESS] LUX siting analysis completed"
else
    echo "[ERROR] LUX siting analysis failed"
fi

echo "[INFO] Processing siting analysis for LVA (T3)..."
if $PY process_country_siting.py LVA $SCENARIO_FLAG; then
    echo "[SUCCESS] LVA siting analysis completed"
else
    echo "[ERROR] LVA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MAR (T3)..."
if $PY process_country_siting.py MAR $SCENARIO_FLAG; then
    echo "[SUCCESS] MAR siting analysis completed"
else
    echo "[ERROR] MAR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDA (T3)..."
if $PY process_country_siting.py MDA $SCENARIO_FLAG; then
    echo "[SUCCESS] MDA siting analysis completed"
else
    echo "[ERROR] MDA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDG (T3)..."
if $PY process_country_siting.py MDG $SCENARIO_FLAG; then
    echo "[SUCCESS] MDG siting analysis completed"
else
    echo "[ERROR] MDG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MDV (T3)..."
if $PY process_country_siting.py MDV $SCENARIO_FLAG; then
    echo "[SUCCESS] MDV siting analysis completed"
else
    echo "[ERROR] MDV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MKD (T3)..."
if $PY process_country_siting.py MKD $SCENARIO_FLAG; then
    echo "[SUCCESS] MKD siting analysis completed"
else
    echo "[ERROR] MKD siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MLI (T3)..."
if $PY process_country_siting.py MLI $SCENARIO_FLAG; then
    echo "[SUCCESS] MLI siting analysis completed"
else
    echo "[ERROR] MLI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MLT (T3)..."
if $PY process_country_siting.py MLT $SCENARIO_FLAG; then
    echo "[SUCCESS] MLT siting analysis completed"
else
    echo "[ERROR] MLT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MMR (T3)..."
if $PY process_country_siting.py MMR $SCENARIO_FLAG; then
    echo "[SUCCESS] MMR siting analysis completed"
else
    echo "[ERROR] MMR siting analysis failed"
fi

echo "[INFO] Siting batch 18/25 (T3) completed at $(date)"
