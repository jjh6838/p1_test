#!/bin/bash --login
#SBATCH --job-name=test
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/test_%j.out
#SBATCH --error=outputs_global/logs/test_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
# Work from submission dir if present, else the script's directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(cd -- "$(dirname -- "$0")" && pwd)"
fi

COUNTRY=${1:-${COUNTRY:-JAM}}
THREADS=${THREADS:-72}

echo "[INFO] Starting single-country test at $(date)"
echo "[INFO] Country: $COUNTRY | Threads: $THREADS"

# Output directories
mkdir -p outputs_per_country outputs_global outputs_global/logs

# Minimal conda bootstrap; use conda run for simplicity
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh
conda --version

RUN="conda run -n p1_etl python"
echo "[INFO] Runner: $RUN"

echo "[INFO] Running process_country_supply.py..."
$RUN process_country_supply.py "$COUNTRY" --output-dir outputs_per_country --threads "$THREADS"

echo "[INFO] Done at $(date)"