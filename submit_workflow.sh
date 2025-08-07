#!/bin/bash --login
#SBATCH --job-name=supply_analysis
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/snakemake_%j.out
#SBATCH --error=outputs_global/logs/snakemake_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ─── directories ──────────────────────────────────────────────
mkdir -p outputs_per_country outputs_global outputs_global/logs

# ─── Conda bootstrap ─────────────────────────────────────────
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

# ensure conda is present
conda --version

# ─── Run Snakemake without profile (direct execution) ────────

echo "[INFO] Launching Snakemake at $(date)"

# Activate the existing p1_etl environment for all rules
conda activate p1_etl

snakemake \
    --cores 72 \
    --rerun-incomplete \
    --keep-going \
    --latency-wait 60 \
    --printshellcmds

echo "[INFO] Workflow finished at $(date)"