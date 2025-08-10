# Snakefile for global supply analysis - PARALLEL MODE ONLY
# ────────────────────────────────────────────────────────────
# Step 1: python get_countries.py --create-parallel
# Step 2: ./submit_all_parallel.sh  
# Step 3: sbatch submit_workflow.sh (combination only)

from pathlib import Path

# ──────────────────────── 1. helpers ────────────────────────

def get_countries():
    """Read country list produced by get_countries.py; fallback to demo list."""
    if Path("countries_list.txt").exists():
        return [c.strip() for c in Path("countries_list.txt").read_text().splitlines() if c.strip()]
    return ["USA", "CHN", "IND", "BRA", "RUS"]

# Always in combination mode - countries processed via parallel scripts
COUNTRIES = []

# Remove tier-based resources - no longer needed for individual processing
# Countries are processed via external parallel scripts

# ──────────────────────── 2. I/O paths ────────────────────────
OUTPUT_DIR          = "outputs_per_country"
GLOBAL_DIR          = "outputs_global"
GLOBAL_SUMMARY_CSV  = f"{GLOBAL_DIR}/global_supply_summary.csv"
ENV_FILE            = "environment.yml"   # environment file in root directory

# Layer-based outputs (GPKG only)
GLOBAL_CENTROIDS    = f"{GLOBAL_DIR}/global_centroids.gpkg"
GLOBAL_GRID_LINES   = f"{GLOBAL_DIR}/global_grid_lines.gpkg"
GLOBAL_FACILITIES   = f"{GLOBAL_DIR}/global_facilities.gpkg"
GLOBAL_ALL_LAYERS   = f"{GLOBAL_DIR}/global_supply_analysis_all_layers.gpkg"
GLOBAL_STATISTICS   = f"{GLOBAL_DIR}/global_statistics.csv"

# ──────────────────────── 3. workflow ────────────────────────
# Only combination step - countries processed externally via parallel scripts
rule all:
    input: GLOBAL_ALL_LAYERS, GLOBAL_STATISTICS, GLOBAL_SUMMARY_CSV

rule get_country_list:
    input:
        demand="outputs_processed_data/p1_b_ember_2024_30_50.xlsx",
        gadm="bigdata_gadm/gadm_410-levels.gpkg"
    output: "countries_list.txt"
    shell: "python get_countries.py"

rule combine_results:
    input: 
        countries_list="countries_list.txt"
    output:
        centroids=GLOBAL_CENTROIDS,
        grid_lines=GLOBAL_GRID_LINES,
        facilities=GLOBAL_FACILITIES,
        all_layers=GLOBAL_ALL_LAYERS,
        statistics=GLOBAL_STATISTICS,
        summary=GLOBAL_SUMMARY_CSV
    threads: 12
    resources:
        mem_mb=64_000,
        runtime=240,
        slurm_partition="Short"
    shell:
        """
        echo "[INFO] Starting global combination..."
        echo "[INFO] Assuming countries were processed via parallel scripts"
        python combine_global_results.py --input-dir {OUTPUT_DIR} --output-file outputs_global/global_supply_analysis.parquet --countries-file {input.countries_list}
        """