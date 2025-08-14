#!/bin/bash --login
#SBATCH --job-name=combine_global
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
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting global combination at $(date)"
echo "[INFO] This job only combines results from parallel processing"

# ─── directories ──────────────────────────────────────────────
mkdir -p outputs_global outputs_global/logs

# ─── Check completion status ─────────────────────────────────
# Check for Parquet files in country subdirectories (production mode)
completed=0
if [ -d "outputs_per_country" ]; then
    completed=$(find outputs_per_country -name "*.parquet" -type f | wc -l)
fi
total=$(wc -l < countries_list.txt 2>/dev/null || echo "0")
echo "[INFO] Found $completed parquet files from $total countries"

if [ "$completed" -eq 0 ]; then
    echo "[ERROR] No completed countries found!"
    echo "[ERROR] Run parallel processing first (production mode):"
    echo "[ERROR]   python get_countries.py --create-parallel"
    echo "[ERROR]   ./submit_all_parallel.sh"
    exit 1
fi

# ─── Conda bootstrap ─────────────────────────────────────────
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl

# ─── Run global combination ──────────────────────────────────
echo "[INFO] Running global combination at $(date)"

python combine_global_results.py \
    --input-dir outputs_per_country \
    --output outputs_global/global_supply_analysis.gpkg \
    --countries-file countries_list.txt

echo "[INFO] Global combination completed at $(date)"
echo ""
echo "=== COMBINATION COMPLETED ==="
echo "Check outputs_global/ for combined results:"
echo "  - global_supply_analysis.gpkg (main visualization file)"
echo "  - logs/combine_results.log (processing log)"
echo ""
echo "You can now open global_supply_analysis.gpkg for global visualization!"