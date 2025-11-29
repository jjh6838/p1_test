#!/usr/bin/env python3
"""
# ==================================================================================================
# SCRIPT PURPOSE: Country List Generation and Parallel Script Creation for HPC
# ==================================================================================================
#
# WHAT THIS CODE DOES:
# This script serves two primary functions for a large-scale geospatial data processing workflow:
#
# 1. Get Country List: It generates a definitive list of countries to be processed. It does this by:
#    a) Reading a list of all countries from the main energy projection data.
#    b) Cross-referencing this list with available administrative boundary data (GADM).
#    c) Producing a final `countries_list.txt` containing only countries that exist in BOTH datasets.
#
# 2. Create Parallel Scripts: It automatically generates shell scripts designed to run the main
#    `process_country_supply.py` analysis in parallel on a High-Performance Computing (HPC)
#    cluster that uses the SLURM workload manager.
#
# WHY THIS IS NEEDED:
# - Consistency: Ensures that the workflow only attempts to process countries for which all
#   necessary input data is available.
# - Efficiency: Processing over 150 countries sequentially would take weeks. By generating
#   parallel scripts, the workload can be distributed across dozens or hundreds of CPU cores,
#   reducing the total runtime to hours.
# - Resource Optimization: The script uses a "tiered" approach, grouping countries by their
#   expected computational load (e.g., large, complex countries like China get a dedicated
#   script with maximum resources, while many small countries are batched together). This
#   ensures efficient use of cluster resources.
# ==================================================================================================
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

def get_country_list():
    """
    Generates a list of countries by finding the intersection between countries present
    in the energy demand dataset and those with available administrative boundaries in GADM.
    The final list is saved to 'countries_list.txt' for use in the Snakemake workflow.
    """
    demand_file = "outputs_processed_data/p1_b_ember_2024_30_50.xlsx"
    gadm_file = "bigdata_gadm/gadm_410-levels.gpkg"
    
    # Check if files exist
    if not Path(demand_file).exists():
        print(f"Error: Demand data file not found: {demand_file}")
        return []
    
    if not Path(gadm_file).exists():
        print(f"Error: GADM boundaries file not found: {gadm_file}")
        return []
    
    print(f"Loading country list from energy demand data: {demand_file}")
    
    try:
        # Load demand data
        demand_df = pd.read_excel(demand_file)
        
        if 'ISO3_code' not in demand_df.columns:
            print(f"Error: 'ISO3_code' column not found in {demand_file}")
            print(f"Available columns: {list(demand_df.columns)}")
            return []
        
        # Get countries from demand data
        demand_countries = demand_df['ISO3_code'].dropna().unique()
        demand_countries = set(str(c).strip() for c in demand_countries if pd.notna(c) and str(c).strip())
        
        print(f"Found {len(demand_countries)} countries in demand data")
        
        # Load GADM boundaries to check which countries exist
        print("Checking which countries have GADM boundaries...")
        admin_df = gpd.read_file(gadm_file, layer="ADM_0", columns=['GID_0'])
        gadm_countries = set(admin_df['GID_0'].dropna().unique())
        gadm_countries = {str(c).strip() for c in gadm_countries if pd.notna(c)}
        
        # Only keep countries that exist in both datasets
        valid_countries = demand_countries & gadm_countries
        missing_countries = demand_countries - gadm_countries
        
        print(f"Countries with both demand data AND boundaries: {len(valid_countries)}")
        if missing_countries:
            print(f"Countries with demand data but NO boundaries (will be skipped): {len(missing_countries)}")
            for country in sorted(list(missing_countries)[:5]):  # Show first 5
                print(f"  - {country}")
            if len(missing_countries) > 5:
                print(f"  ... and {len(missing_countries) - 5} more")
        
        # Sort for consistent ordering
        countries = sorted(list(valid_countries))
        
        print(f"\nFinal country list: {len(countries)} countries will be processed")
        for country in countries[:10]:  # Show first 10
            print(f"  {country}")
        if len(countries) > 10:
            print(f"  ... and {len(countries) - 10} more")
        
        # Save to file for Snakemake
        with open('countries_list.txt', 'w') as f:
            for country in countries:
                f.write(f"{country}\n")
        
        print(f"\nCountry list saved to countries_list.txt")
        print(f"Ready to process {len(countries)} countries in parallel!")
        
        return countries
        
    except Exception as e:
        print(f"Error processing country list: {e}")
        return []

def create_parallel_scripts(num_scripts=40, countries=None):
    """
    Creates a set of shell scripts to process countries in parallel on a SLURM-based HPC cluster.

    This function implements a tiered batching strategy to optimize resource usage. Countries are
    categorized by size and complexity, and scripts are generated with appropriate resource
    requests (memory, CPUs, time). This prevents small jobs from wasting large resource allocations
    and ensures large jobs get the resources they need.
    """
    from pathlib import Path
    
    if countries is None:
        countries = get_country_list()
    
    if not countries:
        print("No countries to process!")
        return False
    
    print(f"Creating {num_scripts} parallel scripts for {len(countries)} countries with tiered approach")
    print(f"📊 Cluster optimization: {num_scripts} scripts = {num_scripts} nodes maximum utilization")
    
    # Define tiers based on country size/complexity. This is determined empirically based on
    # the computational intensity (number of centroids, grid lines) of each country.
    TIER_1 = {"CHN", "USA"}  # Largest countries, require maximum resources.
    TIER_2 = {"IND", "CAN", "MEX"}  # Large countries.
    TIER_3 = {"RUS", "BRA", "AUS", "ARG", "KAZ", "SAU", "IDN", "IRN", "ZAF", "EGY"}  # Medium-large countries.
    TIER_4 = {
        "TUR", "NGA", "COL", "PAK", "PER", "DZA", "VEN", "UKR", "ETH", "PHL", "MLI", "TCD", "SDN", "FRA",
        "SWE", "NOR", "DEU", "IRQ", "MMR", "JPN"
    }  # Medium countries.
    
    # Tier-based resource allocation. This configuration is tailored for a specific HPC cluster node specification.
    # The goal is to pack jobs efficiently onto nodes based on computational requirements.
    # Memory allocation: Tier 1 uses 200GB (high-memory nodes), others use 90GB
    # Partition strategy: Interactive (48h) for Tier 1, Medium (48h) for Tier 2, Short (12h) for Tiers 3-5
    # CPU count: Tier 1 uses 56 CPUs, others use 40 CPUs
    # Cluster Spec as of 11/26/2025: Long nodes 40CPUs/100G, Medium nodes 40CPUs/100G, Short nodes 40CPUs/100GB or 56CPUs/480G
    # Check culster spec on cluster: sinfo -N -o "%P %N %t %c %m" | sort

    TIER_CONFIG = {
        "t1": {"max_countries_per_script": 1, "mem": "200G", "cpus": 56, "time": "48:00:00", "partition": "Interactive"},  # CHN, USA - one country per script
        "t2": {"max_countries_per_script": 1, "mem": "90G", "cpus": 40, "time": "48:00:00", "partition": "Medium"},     # IND, CAN, MEX - one country per script
        "t3": {"max_countries_per_script": 1, "mem": "90G", "cpus": 40, "time": "12:00:00", "partition": "Short"},      # RUS, BRA, AUS, etc. - one country per script
        "t4": {"max_countries_per_script": 2, "mem": "90G", "cpus": 40, "time": "12:00:00", "partition": "Short"},      # TUR, NGA, COL, etc. - two countries per script
        "t5": {"max_countries_per_script": 11, "mem": "90G", "cpus": 40, "time": "12:00:00", "partition": "Short"}     # All others - 11 countries per script
    }
    
    

    def get_tier(country):
        if country in TIER_1:
            return "t1"
        elif country in TIER_2:
            return "t2"
        elif country in TIER_3:
            return "t3"
        elif country in TIER_4:
            return "t4"
        else:
            return "t5"
    
    # Sort countries by tier (biggest first)
    countries_by_tier = {
        "t1": [c for c in countries if get_tier(c) == "t1"],
        "t2": [c for c in countries if get_tier(c) == "t2"],
        "t3": [c for c in countries if get_tier(c) == "t3"],
        "t4": [c for c in countries if get_tier(c) == "t4"],
        "t5": [c for c in countries if get_tier(c) == "t5"]
    }
    
    print("Countries by tier:")
    for tier, tier_countries in countries_by_tier.items():
        config = TIER_CONFIG[tier]
        print(f"  {tier.upper()}: {len(tier_countries)} countries (max {config['max_countries_per_script']} per script)")
        if tier_countries:
            print(f"    Examples: {', '.join(tier_countries[:3])}")
    
    # Create directories
    scripts_dir = Path("parallel_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create batches with tiered approach.
    # This logic groups countries into batches that will become individual shell scripts.
    # It starts with the biggest countries (Tier 1) and works its way down, ensuring
    # that the most resource-intensive jobs are scheduled first.
    all_batches = []
    script_counter = 1
    
    # Process each tier - start with largest countries first
    for tier in ["t1", "t2", "t3", "t4", "t5"]:
        tier_countries = countries_by_tier[tier]
        if not tier_countries:
            continue
            
        config = TIER_CONFIG[tier]
        max_per_script = config["max_countries_per_script"]
        
        # For Tier 5, distribute evenly to avoid overloading the last script
        if tier == "t5" and len(tier_countries) > max_per_script:
            # Calculate optimal batch size to distribute evenly
            scripts_available = num_scripts - script_counter + 1
            num_batches = min(scripts_available, (len(tier_countries) + max_per_script - 1) // max_per_script)
            
            if num_batches > 0:
                # Distribute countries as evenly as possible
                countries_per_batch = len(tier_countries) // num_batches
                extra_countries = len(tier_countries) % num_batches
                
                tier_countries_copy = tier_countries.copy()
                for batch_idx in range(num_batches):
                    # First 'extra_countries' batches get one extra country
                    batch_size = countries_per_batch + (1 if batch_idx < extra_countries else 0)
                    batch = tier_countries_copy[:batch_size]
                    tier_countries_copy = tier_countries_copy[batch_size:]
                    
                    batch_info = {
                        "countries": batch,
                        "tier": tier,
                        "config": config,
                        "script_num": script_counter
                    }
                    all_batches.append(batch_info)
                    script_counter += 1
                    
                    if script_counter > num_scripts:
                        break
        else:
            # For other tiers, use standard batching
            for i in range(0, len(tier_countries), max_per_script):
                batch = tier_countries[i:i + max_per_script]
                batch_info = {
                    "countries": batch,
                    "tier": tier,
                    "config": config,
                    "script_num": script_counter
                }
                all_batches.append(batch_info)
                script_counter += 1
                
                # Stop if we reach the max number of scripts
                if script_counter > num_scripts:
                    # If we have too many batches, combine remaining into last script
                    remaining_countries = tier_countries[i + max_per_script:]
                    if remaining_countries:
                        all_batches[-1]["countries"].extend(remaining_countries)
                    break
        
        if script_counter > num_scripts:
            break
    
    # If we have more tiers but reached script limit, add remaining countries to existing scripts
    if script_counter <= num_scripts:
        for tier in ["t1", "t2", "t3", "t4", "t5"]:
            tier_countries = countries_by_tier[tier]
            if tier_countries and not any(batch["tier"] == tier for batch in all_batches):
                # Add remaining countries to the last batch
                if all_batches:
                    all_batches[-1]["countries"].extend(tier_countries)
    
    # Limit to requested number of scripts
    all_batches = all_batches[:num_scripts]
    
    print(f"\nCreated {len(all_batches)} script batches:")
    
    # Create shell scripts
    for i, batch_info in enumerate(all_batches, 1):
        batch = batch_info["countries"]
        tier = batch_info["tier"]
        config = batch_info["config"]
        
        # The script content includes SLURM directives (#SBATCH) which request the necessary
        # resources (time, memory, CPUs) from the cluster scheduler.
        script_content = f"""#!/bin/bash --login
#SBATCH --job-name=p{i:02d}_{tier}
#SBATCH --partition={config["partition"]}
#SBATCH --time={config["time"]}
#SBATCH --mem={config["mem"]}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task={config["cpus"]}
#SBATCH --output=outputs_global/logs/parallel_{i:02d}_%j.out
#SBATCH --error=outputs_global/logs/parallel_{i:02d}_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script {i}/{len(all_batches)} ({tier.upper()}) at $(date)"
echo "[INFO] Processing {len(batch)} countries in this batch: {', '.join(batch)}"
echo "[INFO] Tier: {tier.upper()} | Memory: {config['mem']} | CPUs: {config['cpus']} | Time: {config['time']}"

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

# Process countries in this batch
"""
        
        # Add country processing commands
        # Each script will call `process_country_supply.py` for each country in its batch.
        for country in batch:
            script_content += f"""
echo "[INFO] Processing {country} ({get_tier(country).upper()})..."
$PY process_country_supply.py {country} --output-dir outputs_per_country
if [ $? -eq 0 ]; then
    echo "[SUCCESS] {country} completed"
else
    echo "[ERROR] {country} failed"
fi
"""
        
        script_content += f"""
echo "[INFO] Batch {i}/{len(all_batches)} ({tier.upper()}) completed at $(date)"
"""
        
        # Write script file
        script_file = scripts_dir / f"submit_parallel_{i:02d}.sh"
        script_file.write_text(script_content, encoding='utf-8')
        script_file.chmod(0o755)
        
        print(f"  Script {i:02d}: {len(batch)} countries ({tier.upper()}) - {', '.join(batch)}")
    
    # Create master submission script
    # This script submits all jobs immediately and returns to prompt
    master_script = f"""#!/bin/bash
# Submit all parallel jobs immediately (SLURM will queue them automatically)

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting {len(all_batches)} parallel jobs..."
echo "[INFO] SLURM will automatically queue and manage job execution (max 8 running at once)"
echo ""

# Submit all jobs
for i in {{01..{len(all_batches):02d}}}; do
    echo "[$(date +%H:%M:%S)] Submitting job $i..."
    sbatch parallel_scripts/submit_parallel_${{i}}.sh
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "[INFO] All {len(all_batches)} jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \\$USER"
echo "  watch -n 60 'squeue -u \\$USER'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name '*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (tiered partition strategy):"
echo "  Tier 1 (CHN, USA):              1 country/script  | Interactive partition (48h) | {TIER_CONFIG['t1']['mem']}, {TIER_CONFIG['t1']['cpus']} CPUs"
echo "  Tier 2 (IND, CAN, MEX):         1 country/script  | Medium partition (48h)      | {TIER_CONFIG['t2']['mem']}, {TIER_CONFIG['t2']['cpus']} CPUs"
echo "  Tier 3 (RUS, BRA, AUS, etc.):   1 country/script  | Short partition (12h)       | {TIER_CONFIG['t3']['mem']}, {TIER_CONFIG['t3']['cpus']} CPUs"
echo "  Tier 4 (TUR, NGA, COL, etc.):   2 countries/script | Short partition (12h)       | {TIER_CONFIG['t4']['mem']}, {TIER_CONFIG['t4']['cpus']} CPUs"
echo "  Tier 5 (all others):           11 countries/script | Short partition (12h)       | {TIER_CONFIG['t5']['mem']}, {TIER_CONFIG['t5']['cpus']} CPUs"
"""
    
    master_file = Path("submit_all_parallel.sh")
    master_file.write_text(master_script, encoding='utf-8')
    master_file.chmod(0o755)
    
    print(f"\nCreated {master_file} to submit all parallel jobs")
    
    # Create combination workflow script
    workflow_script = """#!/bin/bash --login
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
"""
    
    workflow_file = Path("submit_workflow.sh")
    workflow_file.write_text(workflow_script, encoding='utf-8')
    workflow_file.chmod(0o755)
    
    print(f"Created {workflow_file} for combining results")
    
    # Create individual script runner for re-running specific scripts
    single_script = """#!/bin/bash
# Run a single parallel script by number (e.g., ./submit_one.sh 06)
# Usage: ./submit_one.sh <script_number>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <script_number>"
    echo "Example: $0 06"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

SCRIPT_NUM=$(printf "%02d" $1)
SCRIPT_FILE="parallel_scripts/submit_parallel_${SCRIPT_NUM}.sh"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script not found: $SCRIPT_FILE"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts/submit_parallel_*.sh | sed 's/.*submit_parallel_/  /' | sed 's/.sh//'
    exit 1
fi

echo "[INFO] Submitting script ${SCRIPT_NUM}..."
sbatch "$SCRIPT_FILE"

echo ""
echo "Monitor with:"
echo "  squeue -u \\$USER"
echo "  watch -n 60 'squeue -u \\$USER'"
"""
    
    single_file = Path("submit_one.sh")
    single_file.write_text(single_script, encoding='utf-8')
    single_file.chmod(0o755)
    
    print(f"Created {single_file} for submitting individual scripts")
    
    print(f"\nTotal resource allocation:")
    
    # Calculate total resources
    total_mem = sum(int(batch_info["config"]["mem"].replace("G", "")) for batch_info in all_batches)
    total_cpus = sum(batch_info["config"]["cpus"] for batch_info in all_batches)
    
    print(f"  Total Memory: {total_mem}GB")
    print(f"  Total CPUs: {total_cpus}")
    print(f"  Scripts: {len(all_batches)}")
    
    return True

if __name__ == "__main__":
    import sys
    # Allows running from the command line with '--create-parallel' to generate the scripts.
    if len(sys.argv) > 1 and sys.argv[1] == "--create-parallel":
        # Create parallel scripts
        countries = get_country_list()
        if countries:
            create_parallel_scripts(countries=countries)
        else:
            print("No countries found!")
    else:
        # Default behavior - just get countries
        countries = get_country_list()
