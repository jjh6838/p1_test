#!/bin/bash --login
#SBATCH --job-name=p16s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/siting_16_%j.out
#SBATCH --error=outputs_per_country/logs/siting_16_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 16/25 (T3) at $(date)"
echo "[INFO] Processing 11 countries in this batch: ISR, ITA, JAM, JOR, JPN, KEN, KGZ, KHM, KIR, KNA, KOR"
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

echo "[INFO] Processing siting analysis for ISR (T3)..."
if $PY process_country_siting.py ISR $SCENARIO_FLAG; then
    echo "[SUCCESS] ISR siting analysis completed"
else
    echo "[ERROR] ISR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for ITA (T3)..."
if $PY process_country_siting.py ITA $SCENARIO_FLAG; then
    echo "[SUCCESS] ITA siting analysis completed"
else
    echo "[ERROR] ITA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JAM (T3)..."
if $PY process_country_siting.py JAM $SCENARIO_FLAG; then
    echo "[SUCCESS] JAM siting analysis completed"
else
    echo "[ERROR] JAM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JOR (T3)..."
if $PY process_country_siting.py JOR $SCENARIO_FLAG; then
    echo "[SUCCESS] JOR siting analysis completed"
else
    echo "[ERROR] JOR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for JPN (T3)..."
if $PY process_country_siting.py JPN $SCENARIO_FLAG; then
    echo "[SUCCESS] JPN siting analysis completed"
else
    echo "[ERROR] JPN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KEN (T3)..."
if $PY process_country_siting.py KEN $SCENARIO_FLAG; then
    echo "[SUCCESS] KEN siting analysis completed"
else
    echo "[ERROR] KEN siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KGZ (T3)..."
if $PY process_country_siting.py KGZ $SCENARIO_FLAG; then
    echo "[SUCCESS] KGZ siting analysis completed"
else
    echo "[ERROR] KGZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KHM (T3)..."
if $PY process_country_siting.py KHM $SCENARIO_FLAG; then
    echo "[SUCCESS] KHM siting analysis completed"
else
    echo "[ERROR] KHM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KIR (T3)..."
if $PY process_country_siting.py KIR $SCENARIO_FLAG; then
    echo "[SUCCESS] KIR siting analysis completed"
else
    echo "[ERROR] KIR siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KNA (T3)..."
if $PY process_country_siting.py KNA $SCENARIO_FLAG; then
    echo "[SUCCESS] KNA siting analysis completed"
else
    echo "[ERROR] KNA siting analysis failed"
fi

echo "[INFO] Processing siting analysis for KOR (T3)..."
if $PY process_country_siting.py KOR $SCENARIO_FLAG; then
    echo "[SUCCESS] KOR siting analysis completed"
else
    echo "[ERROR] KOR siting analysis failed"
fi

echo "[INFO] Siting batch 16/25 (T3) completed at $(date)"
