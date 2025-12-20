#!/bin/bash --login
#SBATCH --job-name=p35_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_35_%j.out
#SBATCH --error=outputs_global/logs/parallel_35_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 35/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: MLT, MNE, MNG, MOZ, MRT, MUS, MWI, MYS, NAM, NCL"
echo "[INFO] Tier: T5 | Memory: 30G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path to avoid activation issues in batch shells
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Check if running all scenarios (passed via sbatch --export)
SCENARIO_FLAG=""
if [ "$RUN_ALL_SCENARIOS" = "1" ]; then
    SCENARIO_FLAG="--run-all-scenarios"
    echo "[INFO] Running ALL scenarios (100%, 90%, 80%, 70%, 60%)"
fi

# Process countries in this batch

echo "[INFO] Processing MLT (T5)..."
if $PY process_country_supply.py MLT $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MLT completed"
else
    echo "[ERROR] MLT failed"
fi

echo "[INFO] Processing MNE (T5)..."
if $PY process_country_supply.py MNE $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MNE completed"
else
    echo "[ERROR] MNE failed"
fi

echo "[INFO] Processing MNG (T5)..."
if $PY process_country_supply.py MNG $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] MNG completed"
else
    echo "[ERROR] MNG failed"
fi

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

echo "[INFO] Batch 35/40 (T5) completed at $(date)"
