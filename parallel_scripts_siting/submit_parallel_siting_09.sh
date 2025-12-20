#!/bin/bash --login
#SBATCH --job-name=p09s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_09_%j.out
#SBATCH --error=outputs_global/logs/siting_09_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 9/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: ABW, AFG, AGO, ALB, ARE, ARM, ASM, ATG, AUT, AZE, BDI"
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

echo "[INFO] Processing siting analysis for ABW (T3)..."
if $PY process_country_siting.py ABW $SCENARIO_FLAG; then
    echo "[SUCCESS] ABW siting analysis completed"
else
    echo "[ERROR] ABW siting analysis failed"
fi

echo "[INFO] Processing siting analysis for AFG (T3)..."
if $PY process_country_siting.py AFG $SCENARIO_FLAG; then
    echo "[SUCCESS] AFG siting analysis completed"
else
    echo "[ERROR] AFG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for AGO (T3)..."
if $PY process_country_siting.py AGO $SCENARIO_FLAG; then
    echo "[SUCCESS] AGO siting analysis completed"
else
    echo "[ERROR] AGO siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ALB (T3)..."
if $PY process_country_siting.py ALB $SCENARIO_FLAG; then
    echo "[SUCCESS] ALB siting analysis completed"
else
    echo "[ERROR] ALB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ARE (T3)..."
if $PY process_country_siting.py ARE $SCENARIO_FLAG; then
    echo "[SUCCESS] ARE siting analysis completed"
else
    echo "[ERROR] ARE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ARM (T3)..."
if $PY process_country_siting.py ARM $SCENARIO_FLAG; then
    echo "[SUCCESS] ARM siting analysis completed"
else
    echo "[ERROR] ARM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ASM (T3)..."
if $PY process_country_siting.py ASM $SCENARIO_FLAG; then
    echo "[SUCCESS] ASM siting analysis completed"
else
    echo "[ERROR] ASM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ATG (T3)..."
if $PY process_country_siting.py ATG $SCENARIO_FLAG; then
    echo "[SUCCESS] ATG siting analysis completed"
else
    echo "[ERROR] ATG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for AUT (T3)..."
if $PY process_country_siting.py AUT $SCENARIO_FLAG; then
    echo "[SUCCESS] AUT siting analysis completed"
else
    echo "[ERROR] AUT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for AZE (T3)..."
if $PY process_country_siting.py AZE $SCENARIO_FLAG; then
    echo "[SUCCESS] AZE siting analysis completed"
else
    echo "[ERROR] AZE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for BDI (T3)..."
if $PY process_country_siting.py BDI $SCENARIO_FLAG; then
    echo "[SUCCESS] BDI siting analysis completed"
else
    echo "[ERROR] BDI siting analysis failed"
fi

echo "[INFO] Siting batch 9/25 (T3) completed at $(date)"
