# Essential Files for Global Supply Analysis

## CORE FILES (Required - 7 files)

### 1. Main Processing Scripts (3 files)
- `get_countries.py` - Gets list of valid countries from demand data
- `process_country_supply.py` - Processes one country at a time  
- `combine_global_results.py` - Combines all country results

### 2. Workflow Files (4 files)
- `Snakefile` - Defines the workflow (rule conflicts fixed)
- `config.yaml` - Configuration settings
- `environment.yml` - Conda environment with all dependencies
- `test_workflow.py` - Simple test with 2 countries

## CLUSTER FILES (Optional - 2 files)

- `cluster_config.yaml` - Cluster resource settings
- `submit_workflow.sh` - Submit to SLURM cluster

## DATA FILES (Required - must exist)

- `bigdata_gadm/gadm_410.gpkg` - Country boundaries
- `bigdata_gridfinder/grid.gpkg` - Electrical grid  
- `bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif` - Population data
- `outputs_processed_data/p1_a_ember_2024_30.xlsx` - Energy demand data

## USAGE

```bash
# 1. Setup environment (first time only)
conda env update -f environment.yml

# 2. Test with 2 countries first
python test_workflow.py

# 3. If test passes, run full Snakemake workflow
snakemake --cores 4 --use-conda

# 4. For cluster
sbatch submit_workflow.sh
```

## STATUS: ✅ WORKING

- **Environment**: Fixed dependency conflicts, Snakemake installs properly
- **Workflow**: Rule conflicts resolved, DAG builds successfully  
- **Testing**: `test_workflow.py` provides safe way to test
- **Dependencies**: All required packages included in `environment.yml`

## OPTIONAL FILES

- `countries_list.txt` - Auto-generated list of valid countries
