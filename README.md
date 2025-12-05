# Parallel Supply Analysis — Cluster & Local Execution

Process all countries via 40 parallel SLURM jobs (cluster) or locally with automatic CPU detection.

## Status: ✅ Ready for Execution
- **Cluster**: 40 parallel SLURM scripts with tiered batching (1/2/4/8 countries per job)
- **Local**: Automatic core detection (8/16/72 cores), per-country execution
- **Internal parallelization**: Grid processing, facility distance calculations utilize all available cores
- **NEW**: Siting analysis for remote settlement electrification
- **NEW**: ADD_V2 workflow for merging siting results with supply analysis

---

## Recent Major Updates (November 2025)

### 🆕 **Siting Analysis Module** (`process_country_siting.py`)
**⚠️ MUST run AFTER supply analysis completes** - requires existing facility/grid/centroid data

Identifies underserved remote settlements and designs optimal electrification solutions:

#### **Geographic Component Detection** (50km threshold)
- Uses DBSCAN clustering with 50km threshold to identify isolated island groups/territories
- Each geographic component represents settlements that cannot easily share grid infrastructure
- Components with ≥5 settlements are considered viable for facility placement
- Components with <5 settlements are left as off-grid (too small for infrastructure investment)

#### **Synthetic Facility Creation Strategy**
**Goal**: Country-level energy diversity (all 5 types if non-zero demand), but not all types in every component

**Algorithm**:
1. **Sort energy types by total capacity** (smallest to largest): Other Renewables → Wind → Solar → Hydro → Fossil
2. **One facility per component**: Assign smallest unassigned energy type to each viable component
3. **Facility capacity = component's total demand** (not proportional to energy mix)
4. **Fallback**: Remaining energy types go to largest component

**Example** (TLS with 2 components):
- Component 0 (79 settlements, 282,582 MWh) gets: Other Renewables (2,145 MWh capacity set to 282,582 MWh)
- Component 1 (9 settlements, 39,037 MWh) gets: Wind (13,397 MWh capacity set to 39,037 MWh)
- Remaining types (Solar, Hydro, Fossil) → Component 0 (largest, as fallback)

**Rationale**:
- 50km threshold creates truly isolated components (e.g., Timor-Leste mainland vs Oecusse enclave)
- Each component gets dedicated facility with capacity matching its demand
- Smallest energy types distributed first to maximize component coverage
- Geographic suitability can be refined later (e.g., wind for coastal, hydro for mountains)

1. **Settlement Filtering**: 
   - Loads centroids with 'Partially Filled' or 'Not Filled' supply_status
   - Removes settlements with population < 100
   
2. **Capacity-Driven Clustering**: 
   - Calculates number of clusters based on remaining facility capacity by energy type
   - Uses weighted K-means clustering (weight = demand_gap_mwh)
   - Ensures cluster count doesn't exceed available settlements

3. **Facility Matching**: 
   - Matches clusters to facilities based on energy type and remaining capacity
   - Component-aware matching: facilities only serve settlements in same geographic component
   - Dynamic target allocation: prevents monopolization by early facilities
   - Preserves `geo_component` in cluster output for verification
   
4. **Grid Distance Calculation**: 
   - Computes distance from cluster centers to nearest grid lines
   - Identifies remote clusters (>50km from grid) vs grid-accessible clusters

5. **Network Design**: 
   - For remote clusters: Creates minimum spanning tree networks connecting settlements
   - Clips networks to country boundaries (GADM + EEZ)
   - Recalculates distances after clipping

6. **Output Generation**:
   - `siting_clusters_{ISO3}.parquet` - Cluster centers with demand, facility assignments, and geo_component
   - `siting_networks_{ISO3}.parquet` - Network geometries for remote connections
   - `siting_summary_{ISO3}.xlsx` - Summary statistics

```bash
# Step 1: REQUIRED - Run supply analysis first
python process_country_supply.py KEN

# Step 2: Run siting analysis (uses outputs from step 1)
python process_country_siting.py KEN

# Optional: Run all supply scenarios (100%, 90%, 80%, 70%, 60%)
python process_country_siting.py KEN --run-all-scenarios
```

### 🆕 **ADD_V2 Workflow** (Siting Integration)
After siting analysis completes, **re-run supply analysis** to automatically merge siting results:

**How it works:**
1. **First supply run**: `process_country_supply.py` generates base facilities/grid/centroids
2. **Siting run**: `process_country_siting.py` generates cluster centers and networks for remote settlements
3. **Second supply run**: `process_country_supply.py` detects siting outputs and creates integrated `_add_v2` files

**Automatic detection:**
- Supply script checks for `siting_summary_{ISO3}.xlsx` to trigger ADD_V2 mode
- If found, loads existing `grid_lines_{ISO3}.parquet` and `facilities_{ISO3}.parquet` from first run
- Appends siting clusters as synthetic facilities with merged capacity
- Appends siting networks to grid infrastructure
- Stitches networks to existing grid (10km MST threshold)
- Saves integrated results with `_add_v2` suffix

**Output structure:**
```
outputs_per_country/parquet/
  2030_supply_100%/              # First run outputs
    centroids_KEN.parquet
    facilities_KEN.parquet
    grid_lines_KEN.parquet
    siting_clusters_KEN.parquet  # Added by siting run
    siting_networks_KEN.parquet  # Added by siting run
    siting_summary_KEN.xlsx      # Added by siting run
  
  2030_supply_100%_add_v2/       # Second run outputs (integrated)
    centroids_KEN_add_v2.parquet
    facilities_KEN_add_v2.parquet    # Includes synthetic facilities from clusters
    grid_lines_KEN_add_v2.parquet    # Includes siting networks + stitches
```

**Key features:**
- **Data preservation**: Uses first-run facilities/grid as base, avoiding re-computation
- **Network integration**: Siting networks tagged as `line_type='siting_networks'`
- **Grid stitching**: Connects remote networks to existing infrastructure
- **Column consistency**: Ensures `line_type` and `line_id` populated for all grid segments

```bash
# Complete workflow example:
# 1) First supply run
python process_country_supply.py KEN

# 2) Siting analysis (generates siting_clusters, siting_networks, siting_summary)
python process_country_siting.py KEN

# 3) Second supply run (auto-detects siting outputs, generates _add_v2 files)
python process_country_supply.py KEN  # Creates 2030_supply_100%_add_v2/
```

### 🔧 **Facility Data Processing** (`p1_a_ember_gem_2024.py`)
Enhanced spatial clustering and boundary validation:
- **300 arcsec Grid Merging**: Facilities within ~10km × 10km cells are spatially clustered by type
- **Boundary Validation**: Filters out facilities with invalid coordinates using GADM + EEZ data
- **Maritime Support**: Includes offshore facilities in territorial waters (uses Marine Regions EEZ v12)
- **Capacity Aggregation**: Sums capacity for co-located facilities, preserving spatial distribution

### 🔧 **Network Improvements**
- **Geometry Handling**: Fixed WKB deserialization for siting network geometries
- **Line Type Preservation**: Maintains `line_type` attributes (grid_infrastructure, siting_networks, component_stitch)
- **Consistent Stitching**: Uses `line_type` instead of `segment_source` for all edge types
- **Boundary Clipping**: Networks clipped to country boundaries to prevent cross-border extensions

---

## Required Data Files

### **Core Datasets**
All large datasets should be placed in the project root directory:

1. **GADM Boundaries** (Land territories)
   - Path: `bigdata_gadm/gadm_410-levels.gpkg`
   - Source: [GADM v4.1.0](https://gadm.org/download_world.html)
   - Usage: Country land boundaries for spatial validation and clipping

2. **EEZ Maritime Boundaries** (Ocean territories)
   - Path: `bigdata_eez/eez_v12.gpkg`
   - Source: [Marine Regions EEZ v12](https://marineregions.org/downloads.php)
   - Usage: Maritime territorial waters for offshore facility validation
   - Note: Combined with GADM land boundaries for comprehensive coverage

3. **GridFinder Infrastructure**
   - Path: `bigdata_gridfinder/grid.gpkg`
   - Source: [GridFinder](https://gridfinder.rdrn.me/)
   - Usage: Existing grid infrastructure networks

4. **JRC Population**
   - Path: `bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif`
   - Source: [JRC Global Human Settlement](https://ghsl.jrc.ec.europa.eu/)
   - Usage: Population distribution for settlement electrification

### **Input Energy Data**
- `ember_energy_data/`: Ember electricity generation data
- `iea_energy_projections/`: IEA World Energy Outlook projections
- `re_data/`: Global Energy Monitor facility database
- `un_pop/`: UN population projections
- `wb_country_class/`: World Bank country classifications

---

## Quick Start: Cluster Execution (Recommended for All Countries)

### **Supply Analysis (Primary Step)**
```bash
# 1) Generate 40 parallel scripts for supply analysis
python get_countries.py --create-parallel

# 2) Fix line endings & make executable (Linux/cluster)
sed -i 's/\r$//' submit_all_parallel.sh submit_one.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh submit_one.sh parallel_scripts/*.sh

# 3) Submit all 40 parallel jobs (immediate submission, returns to prompt)
./submit_all_parallel.sh

# 4) Monitor progress
squeue -u $USER
tail -f outputs_global/logs/parallel_*.out

# 5) Combine results when jobs complete
sbatch submit_workflow.sh
```

### **Siting Analysis (Secondary Step - Run After Supply)**
⚠️ **Prerequisites**: Supply analysis must complete first

**What siting generates**: Siting analysis creates **additional** parquet files for remote settlements, not the same files as supply analysis:
- `siting_clusters_{ISO3}.parquet` - Remote settlement cluster centers
- `siting_networks_{ISO3}.parquet` - Network lines for remote areas
- `siting_summary_{ISO3}.xlsx` - Summary statistics

**Note**: Siting does NOT regenerate facilities/grid/centroids - those remain from supply analysis.

```bash
# 1) Generate 24 parallel siting scripts (3-tier system)
python get_countries.py --create-parallel-siting

# 2) Fix line endings & make executable (Linux/cluster)
sed -i 's/\r$//' submit_all_parallel_siting.sh submit_one_siting.sh parallel_scripts_siting/*.sh
chmod +x submit_all_parallel_siting.sh submit_one_siting.sh parallel_scripts_siting/*.sh

# 3) Submit all 24 siting jobs
./submit_all_parallel_siting.sh

# 4) Monitor siting progress
squeue -u $USER
tail -f outputs_global/logs/siting_*.out

# 5) Check completion (should be ~189 countries × 2 files = ~378 siting parquets)
# ⚠️ Note: Only countries with unfilled settlements generate siting outputs
#    Countries with 100% supply coverage skip siting (expected behavior)
find outputs_per_country/parquet -name 'siting_clusters_*.parquet' | wc -l
find outputs_per_country/parquet -name 'siting_networks_*.parquet' | wc -l
find outputs_per_country -name 'siting_summary_*.xlsx' | wc -l
```

**What siting generates**: Only for countries with unfilled/partially filled settlements:
- `siting_clusters_{ISO3}.parquet` - Remote settlement cluster centers
- `siting_networks_{ISO3}.parquet` - Network lines for remote areas  
- `siting_summary_{ISO3}.xlsx` - Summary statistics

**Expected behavior**: Siting outputs are **conditional** - only generated when settlements have `supply_status` of "Partially Filled" or "Not Filled". Countries where supply analysis shows 100% coverage (all settlements "Filled") correctly skip siting analysis with no output files generated. Typical range: 20-80 countries need siting depending on scenario assumptions.

### **ADD_V2 Integration (Tertiary Step - Merges Siting into Supply)**
⚠️ **Run AFTER both supply AND siting complete** to create integrated `_add_v2` outputs

**Purpose**: The second supply run generates new `_add_v2` parquet files that merge siting clusters and networks back into the supply analysis layers. This creates a complete integrated dataset.

**Why re-run supply analysis?**
- Reuses existing facility and grid data from first run (no re-computation)
- Loads siting outputs (clusters, networks) and integrates them
- Performs grid stitching to connect siting networks to existing infrastructure
- Ensures consistent column schemas across all layers

**What gets updated in _add_v2:**
- `facilities_{ISO3}_add_v2.parquet`: Original facilities + synthetic facilities from siting clusters
- `grid_lines_{ISO3}_add_v2.parquet`: Original grid + siting networks + connection stitches
- `centroids_{ISO3}_add_v2.parquet`: Same as original (no changes needed)

```bash
# Generate supply parallel scripts again (same as step 1)
python get_countries.py --create-parallel

# Fix line endings if needed
sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh parallel_scripts/*.sh

# Re-submit supply jobs (will auto-detect siting outputs and create _add_v2)
./submit_all_parallel.sh

# Monitor ADD_V2 generation
tail -f outputs_global/logs/parallel_*.out | grep -E "Siting data detected|add_v2"

# Verify completion (should match number of countries with siting outputs)
find outputs_per_country/parquet -type d -name '*_add_v2' | wc -l
ls outputs_per_country/parquet/2030_supply_100%_add_v2/*_add_v2.parquet | wc -l
```

**Expected outputs** (per country with siting data):
- `2030_supply_100%_add_v2/centroids_{ISO3}_add_v2.parquet`
- `2030_supply_100%_add_v2/facilities_{ISO3}_add_v2.parquet` ← Includes synthetic facilities
- `2030_supply_100%_add_v2/grid_lines_{ISO3}_add_v2.parquet` ← Includes siting networks


**Key Files:**
- **Supply Analysis (40 scripts)** - Step 1 & 3: Generates base layers, then _add_v2 layers:
  - `submit_all_parallel.sh` - Submits all 40 supply jobs
  - `submit_one.sh` - Submit single supply script (e.g., `./submit_one.sh 06`)
  - `parallel_scripts/submit_parallel_01.sh` through `submit_parallel_40.sh`
  
- **Siting Analysis (24 scripts)** - Step 2: Generates siting_clusters, siting_networks:
  - `submit_all_parallel_siting.sh` - Submits all 24 siting jobs
  - `submit_one_siting.sh` - Submit single siting script (e.g., `./submit_one_siting.sh 03`)
  - `parallel_scripts_siting/submit_parallel_siting_01.sh` through `submit_parallel_siting_24.sh`

**Complete workflow summary:**
1. **First supply run**: Creates `2030_supply_100%/` with facilities, grid, centroids
2. **Siting run**: Creates `siting_clusters`, `siting_networks`, `siting_summary` in same directory
3. **Second supply run**: Creates `2030_supply_100%_add_v2/` with integrated facilities + networks

**Tips:**
- **Three-step workflow**: 
  1. Supply (first run) → generates base facilities/grid/centroids
  2. Siting → generates clusters/networks for remote settlements  
  3. Supply (second run) → merges siting into _add_v2 integrated layers
- The second supply run is **not** a full recomputation - it reuses existing data
- Siting creates 2 new parquet types per country (clusters, networks)
- ADD_V2 creates 3 integrated parquets per country (all with _add_v2 suffix)
- `line_type` column distinguishes grid infrastructure, siting networks, and stitches
- Use `./script.sh` on Linux/macOS when executing from current directory
- Run `sed -i 's/\r$//'` on cluster if files prepared on Windows
- Each job automatically uses allocated CPUs via internal parallelization

---

## Alternative: Local Execution (Small Datasets or Testing)

```bash
# Single country (auto-detects cores: 8/16/72)
conda activate p1_etl
python process_country_supply.py KOR

# Multiple countries (serial)
python process_country_supply.py USA CHN IND

# Combine all results into global GPKG
python combine_global_results.py --input-dir outputs_per_country

# Or combine single country into GPKG for visualization
python combine_one_results.py KEN
python combine_one_results.py USA --scenario 2050_supply_100%
```

**Performance (automatic scaling):**
- **Korea**: 20-25 mins (16-core laptop) | 35-45 mins (8-core laptop) | 15 mins (72-core cluster)
- **Japan**: 1-1.5 hours (16-core) | 2-3 hours (8-core) | <30 mins (72-core)
- **China**: 2-3 hours (16-core) | 4-6 hours (8-core) | <30 mins (72-core)

---

## Shell Scripts Overview

### **Supply Analysis Scripts (40 scripts)**
- **`submit_all_parallel.sh`** - Submits all 40 supply jobs immediately
  - Submits all jobs at once (returns to prompt in ~1 minute)
  - SLURM automatically manages queue (max 8 jobs run simultaneously based on cluster limits)
  - No active monitoring or wave management needed
  - Script exits after submission, jobs continue running on cluster

- **`submit_one.sh`** - Submit a single supply script by number
  - Usage: `./submit_one.sh 06` (submits supply script 06)
  - Useful for re-running failed jobs or testing specific country batches
  - Lists available scripts if run without arguments
  - Example: `./submit_one.sh 12` submits `parallel_scripts/submit_parallel_12.sh`

### **Siting Analysis Scripts (24 scripts)**
⚠️ **Run after supply analysis completes**

- **`submit_all_parallel_siting.sh`** - Submits all 24 siting jobs immediately
  - Lighter-weight than supply analysis (3-tier system)
  - SLURM automatically manages queue
  - Processes remote settlement electrification for all countries

- **`submit_one_siting.sh`** - Submit a single siting script by number
  - Usage: `./submit_one_siting.sh 03` (submits siting script 03)
  - Useful for re-running failed siting jobs
  - Lists available scripts if run without arguments
  - Example: `./submit_one_siting.sh 08` submits `parallel_scripts_siting/submit_parallel_siting_08.sh`

### **Individual Job Scripts**

**Supply Analysis** (`parallel_scripts/submit_parallel_*.sh`)
40 scripts with tiered country allocation (smallest first):

| Script Range | Countries/Job | Country Tier | Execution Order | Examples |
|--------------|---------------|--------------|-----------------|----------|
| `01-07` | 1 | Tier 1 (Largest) | 1st | USA, CHN, IND, RUS, BRA, CAN, AUS |
| `08-18` | 2 | Tier 2 (Large) | 2nd | ARG+KAZ, DZA+MEX, IDN+SDN, etc. |
| `19-XX` | 4 | Tier 3 (Medium) | 3rd | KOR+FRA+DEU+JPN, GBR+ESP+THA+VNM, etc. |
| `XX-40` | 8 | Other (Smallest) | Last | Island nations, small countries |

**Siting Analysis** (`parallel_scripts_siting/submit_parallel_siting_*.sh`)
24 scripts with simplified 3-tier allocation:

| Script Range | Countries/Job | Country Tier | Examples |
|--------------|---------------|--------------|----------|
| `01-02` | 1 | Tier 1 (Largest) | CHN, USA |
| `03-12` | 2 | Tier 2 (Large) | IND+CAN, MEX+RUS, BRA+AUS, etc. |
| `13-24` | 11 | Tier 3 (Others) | Small/medium countries |

**Each supply script:**
- CPUs: **Tier 1** uses 56 CPUs (high-memory Short nodes); **Tier 2/3/Other** use 40 CPUs
- Memory: 100GB for all tiers
- Partition assignment by tier:
  - **Tier 1** (~17% of scripts): Short partition, 12h time limit, 56 CPUs on high-memory nodes
  - **Tier 2** (~28% of scripts): Medium partition, 48h time limit, 40 CPUs
  - **Tier 3 + Other** (~55% of scripts): Short partition, 12h time limit, 40 CPUs
- Activates conda environment: `conda activate p1_etl`
- Processes assigned countries: `python process_country_supply.py {ISO3}`
- Outputs to: `outputs_per_country/parquet/{scenario}/{layer}_{ISO3}.parquet`
  - Example: `outputs_per_country/parquet/2050_supply_100%/centroids_KOR.parquet`

**Each siting script:**
- CPUs: **Tier 1** uses 56 CPUs; **Tier 2/3** use 40 CPUs
- Memory: **Tier 1** 200GB, **Tier 2** 98GB, **Tier 3** 28GB
- Partition: **Tier 1** Interactive (168h), **Tier 2** Short (12h), **Tier 3** Short (12h)
- Activates conda environment: `conda activate p1_etl`
- Processes assigned countries: `python process_country_siting.py {ISO3}`
- Outputs to: `outputs_per_country/parquet/{scenario}/siting_{layer}_{ISO3}.parquet`
  - Example: `outputs_per_country/parquet/2030_supply_100%/siting_clusters_KEN.parquet`

### **Combination Script**
- **`submit_workflow.sh`** - Combines all country results into global outputs
  - **When to run**: After all 40 parallel jobs complete
  - **Resources**: 40 CPUs, 100GB RAM
  - **Partition**: Medium (12h sufficient for combination)
  - **What it does**: Runs `combine_global_results.py` to merge parquet files
  - **Auto-detects scenarios**: Scans `outputs_per_country/parquet/` for scenario subfolders
  - **Outputs**: `outputs_global/{scenario}_global.gpkg` (one per scenario)
    - Example: `outputs_global/2050_supply_100%_global.gpkg`


---

## Internal Parallelization (Automatic)

**Key Enhancement**: All operations now utilize available cores automatically

### **Automatic Core Detection**
```python
# On cluster: Uses SLURM allocation (respects --cpus-per-task=40)
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if slurm_cpus:
    MAX_WORKERS = int(slurm_cpus)  # Result: 40 on cluster
else:
    MAX_WORKERS = min(40, max(1, os.cpu_count() or 1))  # Result: 16 on laptop, 8 on older laptop
```

### **Parallelized Operations**
1. **Grid line splitting** (>1,000 lines)
   - Divides into batches: `max(50, num_lines // (MAX_WORKERS * 2))`
   - Processes batches in parallel with all cores
   
2. **Grid chunked processing** (>10,000 lines)
   - Creates chunks for spatial processing
   - Parallel execution with ThreadPoolExecutor
   
3. **Facility distance calculations** (>20 facilities, >1,000 centroids)
   - Divides facilities into batches: `len(facilities) // MAX_WORKERS`
   - Each batch runs Dijkstra pathfinding independently
   - Progress reporting every 10%

### **Smart Thresholds**
- Small datasets use serial processing (avoids threading overhead)
- Medium datasets use moderate parallelization
- Large datasets (Japan, China, USA) use full parallelization

**Performance Impact:**
- Korea: 8h+ → 15-25 mins (40-core cluster allocation)
- Japan: 2.5h+ → 25-40 mins target (40-core cluster allocation)
- China: Expected 30-50 mins (40-core cluster allocation)

---

## Workflow Details

### **Supply Analysis** (`process_country_supply.py`)
Main electricity supply chain analysis pipeline:

1. **Facility Processing** (`p1_a_ember_gem_2024.py`)
   - Loads Global Energy Monitor (GEM) facility data
   - **Spatial Clustering**: Groups facilities within 300 arcsecond (~10km) grid cells by energy type
   - **Boundary Validation**: Filters out erroneous coordinates using GADM (land) + EEZ (maritime)
   - **Capacity Aggregation**: Sums capacity for clustered facilities
   - Merges with Ember data for generation statistics

2. **Grid Infrastructure Loading**
   - Loads GridFinder infrastructure or existing processed grid (ADD_V2 mode)
   - Preserves line attributes: `line_type`, `line_id`, `distance_km`
   - Supports multiple line types: grid_infrastructure, siting_networks, component_stitch

3. **Demand Allocation**
   - Projects population and energy demand (IEA scenarios)
   - Allocates demand to population centroids
   - Creates demand layer with unfilled_demand tracking

4. **Network Analysis**
   - Builds graph from grid lines
   - Calculates shortest paths from facilities to centroids
   - Considers grid connectivity constraints
   - **Component Stitching**: Connects isolated grid components using MST (max 10km threshold)

5. **Output Generation**
   - Facilities layer: Point geometries with capacity and generation data
   - Grid lines layer: LineString geometries with distances and line types
   - Centroid demand layer: Population centers with supply/demand matching
   - Summary statistics: Excel report with supply-demand balance

**Outputs:**
- GeoPackage: `outputs_per_country/p1_{ISO3}.gpkg`
- Parquet (per layer): `outputs_per_country/parquet/{scenario}/{layer}_{ISO3}.parquet`

### **Siting Analysis** (`process_country_siting.py`)
**Prerequisites**: Supply analysis must be completed first - siting loads existing facilities/grid/centroids

Remote settlement electrification planning workflow:

1. **Settlement Filtering**
   - Loads centroids from supply analysis output parquet files
   - Filters for 'Partially Filled' or 'Not Filled' supply_status
   - Removes settlements with population < 100
   - Reports breakdown by status

2. **Capacity-Driven Clustering**
   - Calculates remaining facility capacity by energy type (Total - Matched demand)
   - Determines number of clusters = Total remaining capacity / Average cluster demand
   - Uses weighted K-means clustering (weight = demand_gap_mwh per settlement)
   - Ensures cluster count ≤ number of settlements

3. **Facility-Cluster Matching**
   - Assigns each cluster to a specific facility based on:
     - Energy type (Solar, Wind, Hydro, etc.)
     - Remaining facility capacity (remaining_mwh)
     - Cluster demand gap (demand_gap_mwh)
   - Updates facility capacities after matching

4. **Grid Distance Analysis**
   - Computes distance from cluster centers to nearest existing grid lines
   - Classifies clusters:
     - Grid-accessible: ≤50km from grid
     - Remote: >50km from grid (requires new network)

5. **Network Design for Remote Clusters**
   - Creates minimum spanning tree (MST) networks within each remote cluster
   - Connects all settlements in cluster with minimum total line length
   - Clips network edges to country boundaries (GADM + EEZ)
   - Recalculates distances after boundary clipping

6. **Output Generation**
   - Siting clusters: Aggregated settlement groups with capacity requirements
   - Siting networks: LineString geometries for remote connections
   - Summary: Excel report with cluster assignments and network statistics

**Outputs:**
- Parquet: `outputs_per_country/parquet/{scenario}/siting_clusters_{ISO3}.parquet`
- Parquet: `outputs_per_country/parquet/{scenario}/siting_networks_{ISO3}.parquet`
- Summary: `outputs_per_country/parquet/{scenario}/siting_summary_{ISO3}.xlsx`

**Command-line Usage:**
```bash
# Single scenario (100% supply)
python process_country_siting.py KEN

# All scenarios (100%, 90%, 80%, 70%, 60%)
python process_country_siting.py KEN --run-all-scenarios

# Custom output directory
python process_country_siting.py KEN --output-dir custom_outputs
```

### **ADD_V2 Workflow** (Automatic Siting Integration)
When siting analysis completes, the supply analysis can automatically merge results:

**Detection:**
- Checks for `siting_summary_{ISO3}.xlsx` in parquet directory
- If found, triggers ADD_V2 mode with modified processing

**Modified Processing:**
1. Loads existing facilities/grid from base supply analysis parquets (instead of raw data)
2. Loads siting clusters and appends as new facilities with summed capacity
3. Loads siting networks and appends to grid_lines layer
4. Runs component stitching to connect siting networks to main grid
5. Outputs to `{scenario}_add_v2/` directory with `_add_v2` suffix

**Output Naming:**
- Test mode: `p1_{ISO3}_add_v2.gpkg`
- Production mode: `{scenario}_add_v2/` directory for parquet files
- Layers: `facilities_{ISO3}_add_v2.parquet`, `grid_lines_{ISO3}_add_v2.parquet`, etc.

**Example Workflow:**
```bash
# Step 1: Base supply analysis (REQUIRED FIRST)
python process_country_supply.py KEN
# Outputs: 2030_supply_100%/facilities_KEN.parquet, grid_lines_KEN.parquet, centroids_KEN.parquet

# Step 2: Siting analysis (loads outputs from Step 1)
python process_country_siting.py KEN
# Outputs: 2030_supply_100%/siting_clusters_KEN.parquet, siting_networks_KEN.parquet

# Step 3: Integrated analysis (automatic detection, optional)
python process_country_supply.py KEN
# Outputs: 2030_supply_100%_add_v2/facilities_KEN_add_v2.parquet, grid_lines_KEN_add_v2.parquet
```

---

## Multi-threading Configuration

### **Environment Variables** (set automatically)
```bash
OMP_NUM_THREADS = allocated_cpus
MKL_NUM_THREADS = allocated_cpus  
OPENBLAS_NUM_THREADS = allocated_cpus
NUMEXPR_NUM_THREADS = allocated_cpus
```

### **No Manual Configuration Needed**
- Cluster: Automatically uses SLURM-allocated cores (40 CPUs via `SLURM_CPUS_PER_TASK`)
- Laptop: Automatically detects 8/16 cores
- Threading overhead avoided for small datasets

---

## Cluster Resources & Performance

### **Tiered Resource Allocation**

Resource allocation based on actual cluster specs (as of 11/26/2025):
- **Long partition**: 40 CPUs/100GB nodes
- **Medium partition**: 40 CPUs/100GB nodes  
- **Short partition**: 40 CPUs/100GB nodes OR 56 CPUs/480GB nodes (high-memory)

| Country Tier | CPUs/Job | Memory | Partition | Time Limit | Countries/Job | Execution Order | Examples |
|--------------|----------|--------|-----------|------------|---------------|-----------------|----------|
| **Tier 1** | 56 | 100GB | Short | 12h | 1 | 1st (scripts 01-07) | USA, CHN, IND, RUS, BRA, CAN, AUS |
| **Tier 2** | 40 | 100GB | Medium | 48h (2 days) | 2 | 2nd (scripts 08-18) | ARG+KAZ, DZA+MEX, IDN+SDN, LBY+SAU |
| **Tier 3** | 40 | 100GB | Short | 12h | 4 | 3rd (scripts 19-XX) | KOR+FRA+DEU+JPN, GBR+ESP+THA+VNM |
| **Other** | 40 | 100GB | Short | 12h | 8 | Last (scripts XX-40) | Small island nations |

**Execution Strategy:**
- **Largest countries first**: Tier 1 uses high-memory Short nodes (56 CPUs/480GB) for fastest processing
- **Progressive scaling**: Moves from largest to smallest countries
- **Resource optimization**: 100GB memory standard across all tiers; Tier 1 leverages 56 CPUs on Short partition

**Partition Strategy:**
- **Short** (many slots): Used for Tier 1 (largest, 12h on 56 CPUs/480GB nodes), Tier 3, and Other countries (~65% of jobs)
- **Medium** (some slots): Used for Tier 2 (large countries needing up to 2 days, 100GB) (~35% of jobs)

**Combination Step:**
- Memory: 100GB
- Partition: Medium (12h is sufficient for combination)
- CPUs: 40

### **Expected Performance**
- **Tier 1 countries** (Short partition, 56 CPUs): 15-30 minutes, 12h available, fastest processing on high-memory nodes
- **Tier 2 countries** (Medium partition, 40 CPUs): 15-25 minutes, 2 days available
- **Tier 3 countries** (Short partition, 40 CPUs): 5-15 minutes, completes well within 12h
- **Small countries** (Short partition, 40 CPUs): <5 minutes, completes within 12h

**Laptop Performance (16 cores):**
- Small: <10 minutes
- Medium: 20-30 minutes
- Large: 1-2 hours
- Very large: 2-3 hours

### **Performance Monitoring**
```bash
# Check CPU utilization during job
sstat -j <JOB_ID> --format=JobID,MaxRSS,AveCPU

# Check detailed resource usage
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,CPUTime,CPUTimeRAW

# Monitor all parallel jobs
squeue -u $USER --format="%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R"

# Check logs for parallelization messages
tail -f outputs_global/logs/parallel_01.out | grep "Using parallel processing"
```

---

## Troubleshooting

### **Spatial Validation Issues**
**Facilities outside country boundaries:**
- Cause: Coordinate errors in GEM database
- Solution: Automatic filtering using GADM + EEZ boundaries
- Check: `filter_facilities_to_boundaries()` logs show removed facilities per country

**Networks extending beyond borders:**
- Cause: MST algorithm creates straight-line connections
- Solution: Automatic clipping with `gpd.clip()`, distance recalculation
- Check: Verify `line_type` includes `siting_networks` after clipping

**Clusters in ocean or wrong territory:**
- Cause: Weighted centroid calculation can place centers outside polygons
- Solution: Boundary validation moves clusters to nearest valid point
- Check: Siting summary shows cluster repositioning statistics

**Missing offshore facilities:**
- Cause: Using only GADM land boundaries
- Solution: EEZ maritime boundaries included for offshore platforms
- Check: Verify `bigdata_eez/eez_v12.gpkg` exists and loads correctly

### **ADD_V2 Workflow Issues**
**Siting data not detected:**
- Check: `siting_summary_{ISO3}.xlsx` exists in parquet directory
- Verify: File naming matches exactly (case-sensitive)
- Debug: Look for "Siting data detected" message in logs

**Line types not preserved:**
- Check: `line_type` column exists in input parquets
- Verify: Values include 'siting_networks', 'grid_infrastructure', 'component_stitch'
- Debug: Use `gpd.read_parquet().columns` to inspect available attributes

**Duplicate facilities after merging:**
- Cause: Facilities at grid cell boundaries may cluster incorrectly
- Solution: 300 arcsecond grid ensures consistent snapping
- Check: Verify `Num_of_Merged_Units` in facilities output

### **Fixed Issues ✅**
- Environment conflicts resolved in `environment.yml`
- Conda activation fixed in all 40 parallel scripts
- Unicode encoding fixed for Windows/Linux compatibility
- Geographic CRS warnings fixed with proper UTM projections
- WKB geometry deserialization fixed for siting networks
- Spatial clipping implemented for all geometric features
- Internal parallelization implemented (grid processing, facility distances)
- Automatic core detection for cluster/laptop execution

### **Common Issues**
1. **"No completed countries found"** (when running `submit_workflow.sh`)
   - Ensure all parallel jobs completed: `squeue -u $USER`
   - Check for parquet files: `find outputs_per_country -name "*.parquet"`
   - Review job logs: `tail outputs_global/logs/parallel_*.out`

2. **Re-running jobs overwrites existing data**
   - Existing parquet files are **overwritten** without backup
   - To preserve previous results: `mv outputs_per_country outputs_per_country_backup`
   - Or rename scenario subfolder: `mv outputs_per_country/parquet/2050_supply_100% outputs_per_country/parquet/2050_supply_100%_v1`
   - Partial re-runs create mixed datasets (old + new data)

3. **Line ending errors** (`'\r': command not found`)
   - Run on cluster: `sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh`
   - Or use Git: `git config core.autocrlf true`

4. **Permission denied**
   - Run: `chmod +x submit_all_parallel.sh parallel_scripts/*.sh`

5. **Memory errors** (rare with 1024GB allocation)
   - Check job output: `sacct -j <JOB_ID> --format=JobID,MaxRSS`

6. **Time limit exceeded**
   - Tier 1 (Long partition): 168h should be sufficient for all countries
   - Tier 2/3 (Medium/Short): If job exceeds time, consider moving to higher tier
   - Check logs to identify which country took longest

7. **Missing countries in output**
   - Workflow skips countries without GADM boundaries
   - Check `countries_list.txt` for expected countries
   - Some islands may have no grid data (handled gracefully)

### **Performance Issues**
- **Slow execution on laptop**: Expected for large countries, consider cluster
- **Underutilized cores**: Check logs for "Using parallel processing" messages
- **Long runtimes**: Ensure latest code with internal parallelization

---

## Monitoring & Logs

### **Active Job Monitoring**
```bash
# View running jobs
squeue -u $USER

# Monitor specific job output in real-time
tail -f outputs_global/logs/parallel_01.out

# Check all parallel job logs
ls outputs_global/logs/parallel_*.out

# Search for errors across all logs
grep -i error outputs_global/logs/parallel_*.out
```

### **Log Files Generated**
- `outputs_global/logs/parallel_{01-40}.out` - Individual parallel job outputs
- `outputs_global/logs/parallel_{01-40}.err` - Individual parallel job errors
- `outputs_global/logs/test_{JOBID}.out` - Combination step output (from `submit_workflow.sh`)
- `outputs_global/logs/test_{JOBID}.err` - Combination step errors

### **Key Log Messages**
Look for these indicators of successful parallelization:
```
[CONFIG] Using MAX_WORKERS=72 for parallel operations
Using parallel processing for 1234 facilities with 72 workers...
Using parallel processing for 56789 grid lines...
Processed 500/1234 facilities...
```

---
## Cluster Execution Guide

### **Step-by-Step Workflow**

```bash
# ═══════════════════════════════════════════════════════════
# STEP 1: Prepare environment (one-time setup)
# ═══════════════════════════════════════════════════════════
conda activate p1_etl

# ═══════════════════════════════════════════════════════════
# STEP 2: Generate parallel scripts
# ═══════════════════════════════════════════════════════════
# Supply analysis (40 scripts)
python get_countries.py --create-parallel

# Siting analysis (24 scripts)
python get_countries.py --create-parallel-siting

# ═══════════════════════════════════════════════════════════
# STEP 3: Transfer to cluster & fix line endings
# ═══════════════════════════════════════════════════════════
# On cluster - Supply scripts:
sed -i 's/\r$//' submit_all_parallel.sh submit_one.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh submit_one.sh parallel_scripts/*.sh

# On cluster - Siting scripts:
sed -i 's/\r$//' submit_all_parallel_siting.sh submit_one_siting.sh parallel_scripts_siting/*.sh
chmod +x submit_all_parallel_siting.sh submit_one_siting.sh parallel_scripts_siting/*.sh

# ═══════════════════════════════════════════════════════════
# STEP 4: Submit supply analysis jobs (40 jobs, ~8-12 hours)
# ═══════════════════════════════════════════════════════════
./submit_all_parallel.sh
# Or test single script:
./submit_one.sh 06

# Monitor:
squeue -u $USER
watch -n 60 'squeue -u $USER | wc -l'  # Count running jobs

# ═══════════════════════════════════════════════════════════
# STEP 5: Check supply completion
# ═══════════════════════════════════════════════════════════
# Count completed countries (should be ~196):
find outputs_per_country/parquet -name "facilities_*.parquet" -type f | wc -l
find outputs_per_country/parquet -name "centroids_*.parquet" -type f | wc -l

# Check for errors:
grep -i "error\|failed" outputs_global/logs/parallel_*.out

# ═══════════════════════════════════════════════════════════
# STEP 6: Submit siting analysis jobs (24 jobs, ~4-6 hours)
# ═══════════════════════════════════════════════════════════
# ⚠️ Only after supply analysis completes!
./submit_all_parallel_siting.sh
# Or test single script:
./submit_one_siting.sh 03

# Monitor siting:
squeue -u $USER
tail -f outputs_global/logs/siting_*.out

# ═══════════════════════════════════════════════════════════
# STEP 7: Check siting completion
# ═══════════════════════════════════════════════════════════
# Count completed siting files (should be ~189 × 2 = 378):
find outputs_per_country/parquet -name 'siting_clusters_*.parquet' | wc -l
find outputs_per_country/parquet -name 'siting_networks_*.parquet' | wc -l

# Check for siting errors:
grep -i "error\|failed" outputs_global/logs/siting_*.out

# ═══════════════════════════════════════════════════════════
# STEP 8: Generate ADD_V2 integrated parquets (OPTIONAL)
# ═══════════════════════════════════════════════════════════
# ⚠️ Only after BOTH supply AND siting complete!
# Re-run supply analysis to merge siting results
python get_countries.py --create-parallel  # Regenerate scripts
./submit_all_parallel.sh  # Re-submit supply jobs

# Monitor ADD_V2 generation:
squeue -u $USER
tail -f outputs_global/logs/parallel_*.out | grep "Siting data detected"

# ═══════════════════════════════════════════════════════════
# STEP 9: Check ADD_V2 completion (if running ADD_V2)
# ═══════════════════════════════════════════════════════════
# Count _add_v2 parquets (should be ~189 countries × multiple layers):
find outputs_per_country/parquet -name '*_add_v2.parquet' | wc -l
find outputs_per_country/parquet -name 'facilities_*_add_v2.parquet' | wc -l

# Check for ADD_V2 errors:
grep -i "error\|failed" outputs_global/logs/parallel_*.out

# ═══════════════════════════════════════════════════════════
# STEP 10: Combine results (after desired analysis completes)
# ═══════════════════════════════════════════════════════════
sbatch submit_workflow.sh

# Monitor combination:
squeue -u $USER
tail -f outputs_global/logs/test_*.out

# ═══════════════════════════════════════════════════════════
# STEP 11: Verify outputs
# ═══════════════════════════════════════════════════════════
ls -lh outputs_global/*_global.gpkg
# Should show one GPKG per scenario, e.g.:
#   2030_supply_100%_global.gpkg
#   2050_supply_100%_global.gpkg
#   2030_supply_100%_add_v2_global.gpkg (if ADD_V2 run)
```

### **Expected Timeline**
- **Supply analysis**: 8-12 hours (40 simultaneous jobs) - ~189 countries
- **Siting analysis**: 4-6 hours (24 simultaneous jobs, lighter workload) - ~189 countries
- **ADD_V2 integration** (optional): 8-12 hours (40 simultaneous jobs, re-run supply) - ~189 countries
- **Combination**: 1-2 hours
- **Total without ADD_V2**: ~13-20 hours
- **Total with ADD_V2**: ~21-32 hours

---

## Development & Testing

### **Test Single Country Locally**
```bash
conda activate p1_etl
python process_country_supply.py KOR  # Should complete in 20-45 mins on laptop
```

### **Test on Cluster**
```bash
# Edit submit_parallel_01.sh to test single country
sbatch parallel_scripts/submit_parallel_01.sh

# Monitor
squeue -u $USER
tail -f outputs_global/logs/parallel_01.out
```

### **Performance Profiling**
```bash
# Check log files for timing information:
grep "Completed in" outputs_global/logs/parallel_*.out

# Extract performance metrics:
grep "Using parallel processing" outputs_global/logs/parallel_*.out
```

---

## Data Combination Scripts

### **`combine_global_results.py` - Merge All Countries**
Combines parquet files from all countries into single global GeoPackage per scenario.

**Usage:**
```bash
# Auto-detect scenarios in outputs_per_country/parquet/
python combine_global_results.py --input-dir outputs_per_country

# Specify scenario explicitly
python combine_global_results.py --input-dir outputs_per_country --scenario 2030_supply_100%

# Combine only specific countries
python combine_global_results.py --input-dir outputs_per_country --countries USA CHN IND
```

**Features:**
- Scans `outputs_per_country/parquet/` for scenario subfolders
- Combines all 4 layers: facilities, grid_lines, centroids, polylines
- Outputs to: `outputs_global/{scenario}_global.gpkg`
- Logs to: `outputs_global/logs/combine_results.log`
- Shows progress bar for each layer

**When to use:**
- After all parallel supply jobs complete
- To create global analysis dataset
- Called automatically by `submit_workflow.sh` on cluster

### **`combine_one_results.py` - Single Country GPKG**
Converts parquet files for a single country into a GeoPackage for local analysis/visualization.

**Usage:**
```bash
# Basic usage (default scenario: 2030_supply_100%)
python combine_one_results.py KEN

# Specify scenario
python combine_one_results.py KEN --scenario 2050_supply_100%

# Custom base directory
python combine_one_results.py KEN --base-dir outputs_per_country --scenario 2030_supply_100%
```

**Output files:**
- Basic supply: `outputs_per_country/{scenario}_{ISO3}.gpkg`
  - 4 layers: centroids, polylines, grid_lines, facilities

- With siting: `outputs_per_country/{scenario}_{ISO3}_add.gpkg`
  - 7 layers: core 4 + siting_clusters, siting_settlements, siting_networks

- Integrated (ADD_V2): `outputs_per_country/{scenario}_{ISO3}_add_v2.gpkg`
  - 4 layers: integrated centroids, polylines, grid_lines, facilities with siting merged

**Features:**
- Automatically detects siting outputs and creates appropriate GPKG variant
- Removes old GPKG before writing (prevents mixed layers)
- Handles `_add_v2` suffix in scenario names correctly

**When to use:**
- Quick visualization of single country results in QGIS
- Local testing/validation without cluster
- Creating shareable country-specific datasets

---

## Data Schema

Each country output contains:
- `geometry` - Point geometry for population centroid (WGS84)
- `Population_centroid` - Population at this centroid
- `Total_Demand_2030_centroid` - Energy demand for 2030 (MWh)  
- `Total_Demand_2050_centroid` - Energy demand for 2050 (MWh)
- `GID_0` - ISO3 country code
- `distance_km` - Network distance to nearest facility
- `facility_type` - Type of power plant (coal, gas, solar, etc.)
- `facility_capacity` - Capacity in MW

---

## File Structure Summary

```
├── process_country_supply.py          # Main processing script (internal parallelization)
├── process_country_siting.py          # Siting analysis for remote settlements
├── get_countries.py                   # Generates parallel scripts
├── combine_global_results.py          # Combines all countries into global GPKG
├── combine_one_results.py             # Converts single country parquets to GPKG
├── submit_all_parallel.sh             # Master SLURM submission (supply)
├── submit_all_parallel_siting.sh      # Master SLURM submission (siting)
├── submit_one.sh                      # Submit single supply script (e.g., ./submit_one.sh 06)
├── submit_one_siting.sh               # Submit single siting script (e.g., ./submit_one_siting.sh 03)
├── submit_workflow.sh                 # Combination step submission
├── Snakefile                          # Optional Snakemake workflow (combination only)
├── parallel_scripts/
│   ├── submit_parallel_01.sh          # Job 1: USA (Tier 1)
│   ├── submit_parallel_02.sh          # Job 2: CHN (Tier 1)
│   └── ...submit_parallel_40.sh       # Job 40: Small countries
├── outputs_per_country/
│   └── parquet/
│       ├── 2030_supply_100%/          # Scenario-specific outputs
│       │   ├── centroids_KOR.parquet
│       │   ├── facilities_KOR.parquet
│       │   └── ...
│       └── 2050_supply_100%/
│           └── ...
└── outputs_global/
    ├── logs/                          # Job logs
    ├── 2030_supply_100%_global.gpkg   # Combined outputs per scenario
    └── 2050_supply_100%_global.gpkg
```
