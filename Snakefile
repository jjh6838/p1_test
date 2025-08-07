# Snakemake workflow for global supply analysis
# Run with: snakemake --cores all --use-conda

## import pandas as pd (removed, not needed for workflow logic)
from pathlib import Path

# Get list of countries from GADM data
def get_countries():
    """Get list of countries to process"""
    if Path("countries_list.txt").exists():
        with open("countries_list.txt", 'r') as f:
            countries = [line.strip() for line in f if line.strip()]
    else:
        # Default list - you should run get_countries.py first
        countries = ["USA", "CHN", "IND", "BRA", "RUS"]  # Example countries
    return countries

COUNTRIES = get_countries()

# Define country tiers for resource allocation
TIER_1_COUNTRIES = ["USA", "CHN", "IND", "RUS", "BRA", "CAN", "AUS"]
TIER_2_COUNTRIES = ["ARG", "KAZ", "DZA", "COD", "SAU", "MEX", "IDN", "SDN", "LBY", "IRN", "MNG"]
TIER_3_COUNTRIES = ["KOR", "PER", "TCD", "NER", "AGO", "MLI", "ZAF", "COL", "ETH", "BOL", "MRT", 
                   "EGY", "TZA", "NGA", "VEN", "PAK", "TUR", "CHL", "ZMB", "MMR", "AFG", "SOM", 
                   "CAF", "UKR", "MDG", "BWA", "KEN", "FRA", "YEM", "THA", "ESP", "TKM", "CMR", 
                   "PNG", "SWE", "UZB", "MAR", "IRQ", "PRY", "ZWE", "JPN", "DEU", "NOR", "MYS", 
                   "CIV", "POL", "OMN", "ITA", "PHL", "ECU", "BFA", "NZL", "GAB", "GIN", "GBR"]

# Helper function to get tier resources (Snakemake 9.9.0 compatible)
def get_tier_resources(wildcards):
    country = wildcards.country
    if country in TIER_1_COUNTRIES:
        return {"threads": 20, "mem_mb": 80000, "runtime": 1440, "slurm_partition": "Medium"}
    elif country in TIER_2_COUNTRIES:
        return {"threads": 12, "mem_mb": 48000, "runtime": 720, "slurm_partition": "Medium"}
    elif country in TIER_3_COUNTRIES:
        return {"threads": 8, "mem_mb": 32000, "runtime": 360, "slurm_partition": "Medium"}
    else:
        return {"threads": 4, "mem_mb": 16000, "runtime": 180, "slurm_partition": "Medium"}

# Define output directories
OUTPUT_DIR = "outputs_per_country"
GLOBAL_OUTPUT_DIR = "outputs_global"
GLOBAL_OUTPUT = f"{GLOBAL_OUTPUT_DIR}/global_supply_analysis.parquet"

# Rules
rule all:
    input:
        GLOBAL_OUTPUT,
        f"{GLOBAL_OUTPUT_DIR}/global_supply_summary.csv"

rule get_country_list:
    """Extract list of countries that have both demand data and boundaries"""
    input:
        demand="outputs_processed_data/p1_b_ember_2024_30_50.xlsx",
        gadm="bigdata_gadm/gadm_410-levels.gpkg"
    output:
        "countries_list.txt"
    conda:
        "environment.yml"
    shell:
        "python get_countries.py"

rule process_country:
    """Process supply analysis for a single country with tiered resources"""
    input:
        gadm="bigdata_gadm/gadm_410-levels.gpkg",
        grid="bigdata_gridfinder/grid.gpkg",
        population="bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif",
        demand="outputs_processed_data/p1_b_ember_2024_30_50.xlsx"
    output:
        parquet=f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet",
        csv=f"{OUTPUT_DIR}/supply_analysis_{{country}}.csv"
    params:
        country="{country}",
        output_dir=OUTPUT_DIR
    conda:
        "environment.yml"
    threads: lambda wildcards: 16 if wildcards.country in TIER_1_COUNTRIES else 12 if wildcards.country in TIER_2_COUNTRIES else 8 if wildcards.country in TIER_3_COUNTRIES else 4
    resources:
        mem_mb=lambda wildcards: 64000 if wildcards.country in TIER_1_COUNTRIES else 48000 if wildcards.country in TIER_2_COUNTRIES else 32000 if wildcards.country in TIER_3_COUNTRIES else 16000,
        runtime=lambda wildcards: 1440 if wildcards.country in TIER_1_COUNTRIES else 720 if wildcards.country in TIER_2_COUNTRIES else 360 if wildcards.country in TIER_3_COUNTRIES else 180,
        slurm_partition="Medium",
        total_cores=lambda wildcards: 16 if wildcards.country in TIER_1_COUNTRIES else 12 if wildcards.country in TIER_2_COUNTRIES else 8 if wildcards.country in TIER_3_COUNTRIES else 4,
        total_mem_mb=lambda wildcards: 64000 if wildcards.country in TIER_1_COUNTRIES else 48000 if wildcards.country in TIER_2_COUNTRIES else 32000 if wildcards.country in TIER_3_COUNTRIES else 16000
    shell:
        """
        echo "=== Environment validation for {params.country} ==="
        echo "Python executable: $(which python)"
        echo "Python version: $(python --version)"
        echo "Checking pandas import..."
        python -c "import pandas; print('pandas version:', pandas.__version__)"
        echo "=== Starting analysis ==="
        python process_country_supply.py {params.country} --output-dir {params.output_dir} --threads {threads}
        """

rule combine_results:
    """Combine all country results into global dataset"""
    input:
        expand(f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet", country=COUNTRIES)
    output:
        parquet=GLOBAL_OUTPUT,
        csv=f"{GLOBAL_OUTPUT_DIR}/global_supply_analysis.csv",
        summary=f"{GLOBAL_OUTPUT_DIR}/global_supply_summary.csv"
    params:
        input_dir=OUTPUT_DIR,
        output_file=GLOBAL_OUTPUT
    conda:
        "environment.yml"
    threads: 12
    resources:
        mem_mb=32000,
        runtime=180,
        slurm_partition="Medium",
        total_cores=12,
        total_mem_mb=32000
    shell:
        "python combine_global_results.py --input-dir {params.input_dir} --output-file {params.output_file}"
