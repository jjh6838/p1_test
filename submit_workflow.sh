#!/bin/bash --login
#SBATCH --job-name=combine_global
#SBATCH --partition=Medium
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/test_%j.out
#SBATCH --error=outputs_global/logs/test_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting global results combination at $(date)"
echo "[INFO] Memory: 64GB | CPUs: 40 | Time limit: 12h"

# --- directories ---
mkdir -p outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

echo "[INFO] Combining all country results into global outputs..."
echo "[INFO] Auto-detecting scenarios from outputs_per_country/parquet/"

# Run combination script (auto-detects scenarios)
$PY combine_global_results.py --input-dir outputs_per_country

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Global results combination completed at $(date)"
    echo "[INFO] Output files:"
    ls -lh outputs_global/*.gpkg
else
    echo "[ERROR] Global results combination failed at $(date)"
    exit 1
fi
