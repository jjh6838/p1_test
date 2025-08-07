# Snakefile for global supply analysis
# ────────────────────────────────────
# Run with:  sbatch submit_workflow.sh

from pathlib import Path

# ──────────────────────── 1. helpers ────────────────────────

def get_countries():
    """Read country list produced by get_countries.py; fallback to demo list."""
    if Path("countries_list.txt").exists():
        return [c.strip() for c in Path("countries_list.txt").read_text().splitlines() if c.strip()]
    return ["USA", "CHN", "IND", "BRA", "RUS"]

COUNTRIES = get_countries()

TIER_1 = {"USA", "CHN", "IND", "RUS", "BRA", "CAN", "AUS"}
TIER_2 = {"ARG", "KAZ", "DZA", "COD", "SAU", "MEX", "IDN", "SDN", "LBY", "IRN", "MNG"}
TIER_3 = {
    "KOR", "PER", "TCD", "NER", "AGO", "MLI", "ZAF", "COL", "ETH", "BOL", "MRT", "EGY", "TZA", "NGA",
    "VEN", "PAK", "TUR", "CHL", "ZMB", "MMR", "AFG", "SOM", "CAF", "UKR", "MDG", "BWA", "KEN", "FRA",
    "YEM", "THA", "ESP", "TKM", "CMR", "PNG", "SWE", "UZB", "MAR", "IRQ", "PRY", "ZWE", "JPN", "DEU",
    "NOR", "MYS", "CIV", "POL", "OMN", "ITA", "PHL", "ECU", "BFA", "NZL", "GAB", "GIN", "GBR"
}

# Tier‑based resources in one place
TIER_RES = {
    "t1": dict(threads=20, mem_mb=80_000, runtime=1440),
    "t2": dict(threads=12, mem_mb=48_000, runtime=720),
    "t3": dict(threads=8,  mem_mb=32_000, runtime=360),
    "other": dict(threads=4, mem_mb=16_000, runtime=180),
}

def tier(key):
    return TIER_RES[key]

def pick_tier(country):
    if country in TIER_1:
        return "t1"
    if country in TIER_2:
        return "t2"
    if country in TIER_3:
        return "t3"
    return "other"

# ──────────────────────── 2. I/O paths ────────────────────────
OUTPUT_DIR          = "outputs_per_country"
GLOBAL_DIR          = "outputs_global"
GLOBAL_PARQUET      = f"{GLOBAL_DIR}/global_supply_analysis.parquet"
GLOBAL_SUMMARY_CSV  = f"{GLOBAL_DIR}/global_supply_summary.csv"
ENV_FILE            = "environment.yml"   # environment file in root directory

# ──────────────────────── 3. workflow ────────────────────────
rule all:
    input: GLOBAL_PARQUET, GLOBAL_SUMMARY_CSV

rule get_country_list:
    input:
        demand="outputs_processed_data/p1_b_ember_2024_30_50.xlsx",
        gadm="bigdata_gadm/gadm_410-levels.gpkg"
    output: "countries_list.txt"
    shell: "python get_countries.py"

rule process_country:
    input:
        gadm="bigdata_gadm/gadm_410-levels.gpkg",
        grid="bigdata_gridfinder/grid.gpkg",
        population="bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif",
        demand="outputs_processed_data/p1_b_ember_2024_30_50.xlsx"
    output:
        parquet=f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet",
        csv=f"{OUTPUT_DIR}/supply_analysis_{{country}}.csv"
    params:
        outdir=OUTPUT_DIR
    threads: lambda wc: tier(pick_tier(wc.country))["threads"]
    resources:
        mem_mb=lambda wc: tier(pick_tier(wc.country))["mem_mb"],
        runtime=lambda wc: tier(pick_tier(wc.country))["runtime"],
        slurm_partition="Medium"
    shell:
        """
        echo "[{wildcards.country}] Environment: $CONDA_DEFAULT_ENV | Python: $(python --version) | Pandas: $(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "NOT_FOUND")"
        python process_country_supply.py {wildcards.country} --output-dir {params.outdir} --threads {threads}
        """

rule combine_results:
    input: expand(f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet", country=COUNTRIES)
    output:
        parquet=GLOBAL_PARQUET,
        csv=f"{GLOBAL_DIR}/global_supply_analysis.csv",
        summary=GLOBAL_SUMMARY_CSV
    threads: 12
    resources:
        mem_mb=32_000,
        runtime=180,
        slurm_partition="Medium"
    shell:
        "python combine_global_results.py --input-dir {OUTPUT_DIR} --output-file {output.parquet}"