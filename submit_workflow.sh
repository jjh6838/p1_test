#!/bin/bash --login
#SBATCH --job-name=combine_global
#SBATCH --partition=Short
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs_global/logs/combine_global_%j.out
#SBATCH --error=outputs_global/logs/combine_global_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting global combination at $(date)"
echo "[INFO] This job only combines results from parallel processing"

# ─── directories ──────────────────────────────────────────────
mkdir -p outputs_global outputs_global/logs

# ─── Check completion status ─────────────────────────────────
completed=$(find outputs_per_country -name "supply_analysis_*.gpkg" | wc -l)
total=$(wc -l < countries_list.txt 2>/dev/null || echo "0")
echo "[INFO] Found $completed completed countries out of $total total"

if [ "$completed" -eq 0 ]; then
    echo "[ERROR] No completed countries found!"
    echo "[ERROR] Run parallel processing first:"
    echo "[ERROR]   python get_countries.py --create-parallel"
    echo "[ERROR]   ./submit_all_parallel.sh"
    exit 1
fi

# ─── Conda bootstrap ─────────────────────────────────────────
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl

# ─── Run Snakemake (combination only) ────────────────────────
echo "[INFO] Launching Snakemake for combination at $(date)"

snakemake \
    --cores 12 \
    --rerun-incomplete \
    --keep-going \
    --latency-wait 60 \
    --printshellcmds

echo "[INFO] Global combination completed at $(date)"
echo ""
echo "=== COMBINATION COMPLETED ==="
echo "Check outputs_global/ for combined results:"
echo "  - global_centroids.gpkg/csv"
echo "  - global_grid_lines.gpkg/csv"
echo "  - global_facilities.gpkg/csv"
echo "  - global_supply_analysis_all_layers.gpkg"
echo "  - global_supply_summary.csv"