#!/bin/bash --login
#SBATCH --job-name=p36_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_per_country/logs/parallel_36_%j.out
#SBATCH --error=outputs_per_country/logs/parallel_36_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 36/40 (T5) at $(date)"
echo "[INFO] Processing 12 countries in this batch: MOZ, MRT, MUS, MWI, MYS, NAM, NCL, NER, NIC, NLD, NPL, NZL"
echo "[INFO] Tier: T5 | Memory: 25G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country/logs outputs_global

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path to avoid activation issues in batch shells
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

echo "[INFO] Processing MOZ (T5)..."
if $PY process_country_supply.py MOZ $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MOZ completed"
else
    echo "[ERROR] MOZ failed"
fi

echo "[INFO] Processing MRT (T5)..."
if $PY process_country_supply.py MRT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MRT completed"
else
    echo "[ERROR] MRT failed"
fi

echo "[INFO] Processing MUS (T5)..."
if $PY process_country_supply.py MUS $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MUS completed"
else
    echo "[ERROR] MUS failed"
fi

echo "[INFO] Processing MWI (T5)..."
if $PY process_country_supply.py MWI $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MWI completed"
else
    echo "[ERROR] MWI failed"
fi

echo "[INFO] Processing MYS (T5)..."
if $PY process_country_supply.py MYS $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MYS completed"
else
    echo "[ERROR] MYS failed"
fi

echo "[INFO] Processing NAM (T5)..."
if $PY process_country_supply.py NAM $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NAM completed"
else
    echo "[ERROR] NAM failed"
fi

echo "[INFO] Processing NCL (T5)..."
if $PY process_country_supply.py NCL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NCL completed"
else
    echo "[ERROR] NCL failed"
fi

echo "[INFO] Processing NER (T5)..."
if $PY process_country_supply.py NER $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NER completed"
else
    echo "[ERROR] NER failed"
fi

echo "[INFO] Processing NIC (T5)..."
if $PY process_country_supply.py NIC $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NIC completed"
else
    echo "[ERROR] NIC failed"
fi

echo "[INFO] Processing NLD (T5)..."
if $PY process_country_supply.py NLD $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NLD completed"
else
    echo "[ERROR] NLD failed"
fi

echo "[INFO] Processing NPL (T5)..."
if $PY process_country_supply.py NPL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NPL completed"
else
    echo "[ERROR] NPL failed"
fi

echo "[INFO] Processing NZL (T5)..."
if $PY process_country_supply.py NZL $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NZL completed"
else
    echo "[ERROR] NZL failed"
fi

echo "[INFO] Batch 36/40 (T5) completed at $(date)"
