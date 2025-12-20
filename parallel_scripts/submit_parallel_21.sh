#!/bin/bash --login
#SBATCH --job-name=p21_t4
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=95G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_21_%j.out
#SBATCH --error=outputs_global/logs/parallel_21_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 21/40 (T4) at $(date)"
echo "[INFO] Processing 2 countries in this batch: NGA, NOR"
echo "[INFO] Tier: T4 | Memory: 95G | CPUs: 40 | Time: 12:00:00"

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

echo "[INFO] Processing NGA (T4)..."
if $PY process_country_supply.py NGA $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NGA completed"
else
    echo "[ERROR] NGA failed"
fi

echo "[INFO] Processing NOR (T4)..."
if $PY process_country_supply.py NOR $SCENARIO_FLAG --output-dir outputs_per_country; then
    echo "[SUCCESS] NOR completed"
else
    echo "[ERROR] NOR failed"
fi

echo "[INFO] Batch 21/40 (T4) completed at $(date)"
