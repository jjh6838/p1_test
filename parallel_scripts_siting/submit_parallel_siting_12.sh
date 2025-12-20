#!/bin/bash --login
#SBATCH --job-name=p12s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_12_%j.out
#SBATCH --error=outputs_per_country/logs/siting_12_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 12/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: COG, COL, COM, CPV, CRI, CUB, CYM, CYP, CZE, DEU, DJI"
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

echo "[INFO] Processing siting analysis for COG (T3)..."
if $PY process_country_siting.py COG $SCENARIO_FLAG; then
    echo "[SUCCESS] COG siting analysis completed"
else
    echo "[ERROR] COG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for COL (T3)..."
if $PY process_country_siting.py COL $SCENARIO_FLAG; then
    echo "[SUCCESS] COL siting analysis completed"
else
    echo "[ERROR] COL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for COM (T3)..."
if $PY process_country_siting.py COM $SCENARIO_FLAG; then
    echo "[SUCCESS] COM siting analysis completed"
else
    echo "[ERROR] COM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CPV (T3)..."
if $PY process_country_siting.py CPV $SCENARIO_FLAG; then
    echo "[SUCCESS] CPV siting analysis completed"
else
    echo "[ERROR] CPV siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CRI (T3)..."
if $PY process_country_siting.py CRI $SCENARIO_FLAG; then
    echo "[SUCCESS] CRI siting analysis completed"
else
    echo "[ERROR] CRI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CUB (T3)..."
if $PY process_country_siting.py CUB $SCENARIO_FLAG; then
    echo "[SUCCESS] CUB siting analysis completed"
else
    echo "[ERROR] CUB siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CYM (T3)..."
if $PY process_country_siting.py CYM $SCENARIO_FLAG; then
    echo "[SUCCESS] CYM siting analysis completed"
else
    echo "[ERROR] CYM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CYP (T3)..."
if $PY process_country_siting.py CYP $SCENARIO_FLAG; then
    echo "[SUCCESS] CYP siting analysis completed"
else
    echo "[ERROR] CYP siting analysis failed"
fi

echo "[INFO] Processing siting analysis for CZE (T3)..."
if $PY process_country_siting.py CZE $SCENARIO_FLAG; then
    echo "[SUCCESS] CZE siting analysis completed"
else
    echo "[ERROR] CZE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DEU (T3)..."
if $PY process_country_siting.py DEU $SCENARIO_FLAG; then
    echo "[SUCCESS] DEU siting analysis completed"
else
    echo "[ERROR] DEU siting analysis failed"
fi

echo "[INFO] Processing siting analysis for DJI (T3)..."
if $PY process_country_siting.py DJI $SCENARIO_FLAG; then
    echo "[SUCCESS] DJI siting analysis completed"
else
    echo "[ERROR] DJI siting analysis failed"
fi

echo "[INFO] Siting batch 12/25 (T3) completed at $(date)"
