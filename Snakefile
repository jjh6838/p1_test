# Snakemake workflow for global supply analysis
# Run with: snakemake --cores all --use-conda

import pandas as pd
from pathlib import Path

# Configuration
configfile: "config.yaml"

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

# Define output directories
OUTPUT_DIR = config.get("output_dir", "outputs_per_country")
GLOBAL_OUTPUT = config.get("global_output", "global_supply_analysis.parquet")

# Rules
rule all:
    input:
        GLOBAL_OUTPUT,
        "global_supply_summary.csv"

rule get_country_list:
    """Extract list of countries that have both demand data and boundaries"""
    input:
        demand="outputs_processed_data/p1_a_ember_2024_30.xlsx",
        gadm="bigdata_gadm/gadm_410.gpkg"
    output:
        "countries_list.txt"
    conda:
        "environment.yml"
    shell:
        "python get_countries.py"

rule process_country:
    """Process supply analysis for a single country"""
    input:
        gadm="bigdata_gadm/gadm_410.gpkg",
        grid="bigdata_gridfinder/grid.gpkg",
        population="bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif",
        demand="outputs_processed_data/p1_a_ember_2024_30.xlsx"
    output:
        parquet=f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet",
        csv=f"{OUTPUT_DIR}/supply_analysis_{{country}}.csv"
    params:
        country="{country}",
        output_dir=OUTPUT_DIR
    conda:
        "environment.yml"
    resources:
        mem_mb=8000,
        runtime=120  # 2 hours
    shell:
        "python process_country_supply.py {params.country} --output-dir {params.output_dir}"

rule combine_results:
    """Combine all country results into global dataset"""
    input:
        expand(f"{OUTPUT_DIR}/supply_analysis_{{country}}.parquet", country=COUNTRIES)
    output:
        parquet=GLOBAL_OUTPUT,
        csv="global_supply_analysis.csv",
        summary="global_supply_summary.csv"
    params:
        input_dir=OUTPUT_DIR,
        output_file=GLOBAL_OUTPUT
    conda:
        "environment.yml"
    resources:
        mem_mb=16000,
        runtime=60  # 1 hour
    shell:
        "python combine_global_results.py --input-dir {params.input_dir} --output-file {params.output_file}"


