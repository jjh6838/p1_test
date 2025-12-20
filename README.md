# Global Electricity Supply-Demand Analysis Framework

A comprehensive geospatial analysis pipeline for modeling electricity supply networks, projecting future energy demand, and identifying optimal locations for renewable energy infrastructure across 189+ countries.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Requirements](#data-requirements)
- [Scripts Reference](#scripts-reference)
- [Workflow Guide](#workflow-guide)
- [Output Formats](#output-formats)
- [High-Performance Computing (HPC)](#high-performance-computing-hpc)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This project performs country-level analysis of electricity supply and demand networks by:

1. **Integrating global datasets** — Power plant locations (Global Energy Monitor), electricity statistics (Ember), population distributions (JRC GHS-POP), and grid infrastructure (GridFinder)
2. **Projecting future scenarios** — 2030 and 2050 energy demand based on IEA World Energy Outlook and UN population projections
3. **Modeling supply networks** — Network graph analysis to match power generation facilities with population demand centers
4. **Identifying underserved areas** — Siting analysis for remote settlements requiring new infrastructure
5. **Assessing climate impacts** — CMIP6-based projections of solar PV output, wind power density, and hydropower runoff changes

---

## Features

- **Multi-scale analysis**: From global aggregation to individual settlement resolution (~9km grid cells)
- **Multiple scenarios**: Configurable supply factors (60%–100%) and target years (2030, 2050)
- **Energy type differentiation**: Solar, Wind, Hydro, Other Renewables, Nuclear, Fossil
- **Parallel processing**: Automatic CPU detection with HPC cluster support (SLURM)
- **Maritime support**: Includes offshore facilities using EEZ boundaries
- **Climate projections**: CMIP6 ensemble-mean projections for solar, wind, and hydro with uncertainty quantification
- **Publication-ready outputs**: GeoPackage and Parquet formats for GIS visualization and analysis

---

## Project Structure

```
├── config.py                          # Central configuration parameters
│
├── # ═══ Data Preparation Scripts ═══
├── p1_a_ember_gem_2024.py             # Harmonize Ember + Global Energy Monitor data
├── p1_b_ember_2024_30_50.py           # Project 2030/2050 energy scenarios
├── p1_c_cmip6_solar.py                # CMIP6 solar radiation projections
├── p1_d_cmip6_wind.py                 # CMIP6 wind power density projections
├── p1_e_cmip6_hydro.py                # ERA5-Land + CMIP6 runoff projections
├── p1_f_hydroatlas.py                 # HydroATLAS river reach projections
│
├── # ═══ Core Analysis Scripts ═══
├── process_country_supply.py          # Main supply-demand network analysis
├── process_country_siting.py          # Remote settlement siting analysis
├── generate_hpc_scripts.py            # Generate country list and HPC scripts
│
├── # ═══ Results Processing ═══
├── combine_one_results.py             # Combine single country to GeoPackage
├── combine_global_results.py          # Combine all countries to global GeoPackage
├── p1_y_results_data_etl.py           # Exposure analysis ETL pipeline
│
├── # ═══ Figure Generation ═══
├── p1_z_fig12.py                      # Figures 1-2: Global energy exposure
├── p1_z_fig34.py                      # Figures 3-4: Exposure by type/year
├── p1_z_fig56.py                      # Figures 5-6: Detailed exposure
├── p1_z_fig7.py                       # Figure 7: Sensitivity analysis
├── p1_z_fig8.py                       # Figure 8: Hazard-specific breakdown
│
├── # ═══ HPC Execution Scripts ═══
├── submit_all_parallel.sh             # Submit all supply analysis jobs
├── submit_all_parallel_siting.sh      # Submit all siting analysis jobs
├── submit_one.sh                      # Submit single supply script
├── submit_one_siting.sh               # Submit single siting script
├── submit_workflow.sh                 # Submit results combination job
├── parallel_scripts/                  # 40 supply analysis SLURM scripts
├── parallel_scripts_siting/           # 24 siting analysis SLURM scripts
│
├── # ═══ Data Directories ═══
├── bigdata_gadm/                      # GADM administrative boundaries
├── bigdata_eez/                       # Marine Regions EEZ boundaries
├── bigdata_gridfinder/                # GridFinder electrical grid data
├── bigdata_settlements_jrc/           # JRC GHS-POP population raster
├── bigdata_solar_pvout/               # Global Solar Atlas baseline
├── bigdata_wind_atlas/                # Global Wind Atlas baseline
├── bigdata_solar_cmip6/               # CMIP6 solar projections
├── bigdata_wind_cmip6/                # CMIP6 wind projections
├── bigdata_hydro_cmip6/               # CMIP6 runoff projections
├── bigdata_hydro_era5_land/           # ERA5-Land runoff data
├── bigdata_hydro_atlas/               # HydroATLAS river datasets
├── data_energy_ember/                 # Ember electricity statistics
├── data_energy_projections_iea/       # IEA World Energy Outlook data
├── data_pop_un/                       # UN population projections
├── data_country_class_wb/             # World Bank country classifications
│
├── # ═══ Output Directories ═══
├── outputs_per_country/               # Country-level Parquet outputs
├── outputs_global/                    # Combined global GeoPackage outputs
├── outputs_processed_data/            # Processed analysis results
└── outputs_processed_fig/             # Generated figures
```

---

## Installation

### Prerequisites

- Python 3.11+
- Conda or Mamba package manager
- ~50GB disk space for datasets
- 16GB+ RAM (32GB+ recommended for large countries)

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd p1_test

# Create conda environment
conda env create -f environment.yml
conda activate p1_etl

# Verify installation
python -c "import geopandas; import networkx; print('Ready!')"
```

### Required Packages

Key dependencies (full list in `environment.yml`):
- `geopandas` — Geospatial data handling
- `networkx` — Graph-based network analysis
- `rasterio` — Raster data processing
- `scikit-learn` — K-means clustering
- `scipy` — Minimum spanning tree algorithms
- `pandas`, `numpy` — Data manipulation
- `pyarrow` — Parquet I/O

---

## Quick Start

### Single Country Analysis (Local)

```bash
# Activate environment
conda activate p1_etl

# Run supply analysis for Kenya
python process_country_supply.py KEN

# Run siting analysis (after supply completes)
python process_country_siting.py KEN

# Combine results to GeoPackage for visualization
python combine_one_results.py KEN
```

### Multiple Countries (Local)

```bash
# Process multiple countries sequentially
python process_country_supply.py USA CHN IND

# Combine all completed countries to global dataset
python combine_global_results.py --input-dir outputs_per_country
```

### All Countries (HPC Cluster)

```bash
# Generate parallel SLURM scripts
python generate_hpc_scripts.py --create-parallel

# Fix line endings (if prepared on Windows)
sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh parallel_scripts/*.sh

# Submit all 40 parallel jobs (single scenario: 100%)
./submit_all_parallel.sh

# OR: Submit with ALL scenarios (100%, 90%, 80%, 70%, 60%)
./submit_all_parallel.sh --run-all-scenarios

# OR: Submit with a specific supply factor (e.g., 90% only)
./submit_all_parallel.sh --supply-factor 0.9

# Monitor progress
squeue -u $USER
tail -f outputs_global/logs/parallel_*.out
```

### Single Country (HPC Cluster)

Use `submit_one.sh` and `submit_one_siting.sh` to submit individual parallel scripts.
Each script contains one or more countries grouped by computational tier.

```bash
# List available scripts and see which countries are in each
cat parallel_scripts/submit_parallel_01.sh | grep "Processing"
# Output: Processing 1 countries in this batch: CHN

# Submit a specific supply script by number (single scenario: 100%)
./submit_one.sh 01              # Submit script 01 (CHN)
./submit_one.sh 5               # Leading zero optional

# Submit with all 5 scenarios (100%, 90%, 80%, 70%, 60%)
./submit_one.sh 01 --run-all-scenarios

# Submit with a specific supply factor (e.g., 90% only)
./submit_one.sh 01 --supply-factor 0.9

# Same for siting analysis
./submit_one_siting.sh 03
./submit_one_siting.sh 03 --run-all-scenarios
./submit_one_siting.sh 03 --supply-factor 0.9

# Check which script contains a specific country
grep -l "USA" parallel_scripts/*.sh
# Output: parallel_scripts/submit_parallel_05.sh
```

**Script-to-Country Mapping (Tier 1-2):**
| Script | Countries | Tier | Notes |
|--------|-----------|------|-------|
| 01 | CHN | T1 | Largest, 168h Interactive |
| 02 | IND | T2 | Large, 168h Long |
| 03 | BRA | T2 | Large, 168h Long |
| 04 | DEU | T2 | Large, 168h Long |
| 05 | USA | T2 | Large, 168h Long |
| 06+ | Multiple | T3-T5 | Medium/Small countries |

---

## Configuration

All configurable parameters are centralized in `config.py`:

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ANALYSIS_YEAR` | 2030 | Target year: 2024, 2030, or 2050 |
| `SUPPLY_FACTOR` | 1.0 | Sensitivity multiplier (0.6–1.0) |
| `COMMON_CRS` | EPSG:4326 | Coordinate reference system |
| `DEMAND_TYPES` | Solar, Wind, Hydro, Other Renewables, Nuclear, Fossil | Energy categories |

### Grid Resolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POP_AGGREGATION_FACTOR` | 10 | Aggregation factor for population grid |
| `TARGET_RESOLUTION_ARCSEC` | 300 | Final resolution (~9km at equator) |

> **Note**: After changing `POP_AGGREGATION_FACTOR`, regenerate CMIP6 outputs:
> ```bash
> python p1_c_cmip6_solar.py --process-only
> python p1_d_cmip6_wind.py --process-only
> python p1_e_cmip6_hydro.py --process-only
> ```

### Network Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_STITCH_DISTANCE_KM` | 30 | Threshold for stitching grid segments |
| `NODE_SNAP_TOLERANCE_M` | 100 | Snap tolerance for grid nodes |
| `MAX_CONNECTION_DISTANCE_M` | 50,000 | Max facility-to-grid distance |
| `FACILITY_SEARCH_RADIUS_KM` | 300 | Max facility search radius |

### Siting Analysis Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLUSTER_RADIUS_KM` | 50 | K-means clustering radius |
| `GRID_DISTANCE_THRESHOLD_KM` | 50 | Remote vs near-grid classification |
| `DROP_PERCENTAGE` | 0.01 | Filter bottom X% settlements by demand |

---

## Data Requirements

### Core Datasets

| Dataset | Path | Source | Description |
|---------|------|--------|-------------|
| **GADM Boundaries** | `bigdata_gadm/gadm_410-levels.gpkg` | [GADM v4.1](https://gadm.org/) | Country land boundaries |
| **EEZ Boundaries** | `bigdata_eez/eez_v12.gpkg` | [Marine Regions v12](https://marineregions.org/) | Maritime territorial waters |
| **GridFinder** | `bigdata_gridfinder/grid.gpkg` | [GridFinder](https://gridfinder.rdrn.me/) | Global grid infrastructure |
| **JRC Population** | `bigdata_settlements_jrc/GHS_POP_E2025_*.tif` | [JRC GHSL](https://ghsl.jrc.ec.europa.eu/) | Population distribution |
| **Solar Baseline** | `bigdata_solar_pvout/PVOUT.tif` | [Global Solar Atlas](https://globalsolaratlas.info/) | PVOUT baseline |
| **Wind Baseline** | `bigdata_wind_atlas/gasp_*.tif` | [Global Wind Atlas](https://globalwindatlas.info/) | Wind power density |
| **HydroATLAS** | `bigdata_hydro_atlas/RiverATLAS_Data_v10.gdb` | [HydroATLAS](https://www.hydrosheds.org/hydroatlas) | River reach attributes |

### Energy Statistics

| Dataset | Path | Source |
|---------|------|--------|
| **Ember Data** | `data_energy_ember/yearly_full_release_*.csv` | [Ember](https://ember-climate.org/) |
| **IEA Projections** | `data_energy_projections_iea/WEO*.csv` | [IEA WEO 2024](https://www.iea.org/) |
| **UN Population** | `data_pop_un/` | [UN WPP 2024](https://population.un.org/) |
| **World Bank** | `data_country_class_wb/` | [World Bank](https://datahelpdesk.worldbank.org/) |

---

## Scripts Reference

### Data Preparation

#### `p1_a_ember_gem_2024.py`
Harmonizes Ember country-level statistics with Global Energy Monitor facility data.

**Features:**
- Integrates country totals with facility locations
- Spatially clusters facilities within 300 arcsec (~10km) grid cells
- Validates coordinates against GADM + EEZ boundaries
- Filters out offshore facilities beyond territorial waters

**Output:** `data_facilities_gem/p1_a_ember_2024_30.xlsx`

#### `p1_b_ember_2024_30_50.py`
Projects 2030 and 2050 electricity generation scenarios.

**Features:**
- Incorporates UN population growth factors
- Processes National Determined Contributions (NDCs)
- Applies IEA growth rates for fossil/nuclear
- Disaggregates broad renewable targets

**Output:** `data_facilities_gem/p1_b_ember_2024_30_50.xlsx`

#### `p1_c_cmip6_solar.py` / `p1_d_cmip6_wind.py`
Generate CMIP6-based climate projections for solar and wind resources.

**Method:**
1. Download CMIP6 ensemble data (CESM2, EC-Earth3-veg-lr, MPI-ESM1-2-lr)
2. Calculate delta: Δ = Future_period / Historical_period
3. Apply to baseline: Future = Baseline × Δ
4. Compute uncertainty (interquartile range)

**Outputs:**
- `bigdata_solar_cmip6/outputs/PVOUT_{year}_300arcsec.tif`
- `bigdata_wind_cmip6/outputs/WPD100_{year}_300arcsec.tif`

#### `p1_e_cmip6_hydro.py`
Generate ERA5-Land + CMIP6 based runoff projections for hydropower potential.

**Data Sources:**
- **Baseline**: ERA5-Land monthly runoff (reanalysis, 0.1° resolution)
- **Climate projections**: CMIP6 `total_runoff` (SSP2-4.5 scenario)

**Method:**
1. Download ERA5-Land runoff via CDS API (1995-2014 baseline)
2. Download CMIP6 total_runoff for historical and future periods
3. Calculate annual mean baseline from ERA5-Land
4. Compute delta ratios: Δ = CMIP6_future / CMIP6_historical
5. Apply deltas to ERA5 baseline: Future = Baseline × Δ
6. Clip extreme deltas [0.2, 3.0] for robustness

**CMIP6 Models:**
- CESM2, EC-Earth3-veg-lr, MPI-ESM1-2-lr (ensemble mean)

**Outputs:**
- `bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_baseline_300arcsec.tif`
- `bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_{2030,2050}_300arcsec.tif`
- `bigdata_hydro_cmip6/outputs/HYDRO_DELTA_{2030,2050}_300arcsec.tif`

#### `p1_f_hydroatlas.py`
Apply CMIP6 runoff change factors to HydroATLAS river reach data.

**Prerequisites:**
- RiverATLAS dataset from [HydroATLAS](https://www.hydrosheds.org/hydroatlas)
- Delta rasters from `p1_e_cmip6_hydro.py`

**Method:**
1. Load RiverATLAS river reach geometries and attributes
2. Extract CMIP6 delta values at each river reach centroid
3. Apply deltas to baseline discharge: Future_Q = Baseline_Q × Δ
4. Compute hydropower potential indicator: P ∝ Q × gradient × elevation
5. Filter by minimum discharge and stream order thresholds

**Key Attributes Used:**
- `dis_m3_pyr`: Mean annual discharge (m³/year)
- `run_mm_cyr`: Land surface runoff (mm/year)
- `sgr_dk_rav`: Stream gradient (‰)
- `ele_mt_uav`: Upstream mean elevation (m)
- `ord_stra`: Strahler stream order

**Outputs:**
- `bigdata_hydro_atlas/outputs/RiverATLAS_projected_{2030,2050}.parquet`
- `bigdata_hydro_atlas/outputs/RiverATLAS_baseline.parquet`
- `bigdata_hydro_atlas/outputs/RiverATLAS_projected.gpkg`

---

### Core Analysis

#### `process_country_supply.py`
Main supply-demand network analysis pipeline.

```bash
# Basic usage (single scenario: 100%)
python process_country_supply.py <ISO3>

# All supply scenarios (100%, 90%, 80%, 70%, 60%)
python process_country_supply.py <ISO3> --run-all-scenarios

# Single specific supply factor (e.g., 90% only)
python process_country_supply.py <ISO3> --supply-factor 0.9

# Multiple countries
python process_country_supply.py USA CHN IND

# Custom scenario
python process_country_supply.py KEN --scenario 2050_supply_100%

# Test mode (outputs GeoPackage)
python process_country_supply.py KEN --test
```

**Pipeline Steps:**
1. **Load boundaries** — GADM (land) + EEZ (maritime)
2. **Process facilities** — Filter, cluster, validate locations
3. **Build grid network** — Load GridFinder, create NetworkX graph
4. **Allocate demand** — Distribute national demand to population centroids
5. **Network analysis** — Calculate shortest paths, match supply to demand
6. **Output generation** — Parquet files per layer

**Outputs:**
- `outputs_per_country/parquet/{scenario}/centroids_{ISO3}.parquet`
- `outputs_per_country/parquet/{scenario}/facilities_{ISO3}.parquet`
- `outputs_per_country/parquet/{scenario}/grid_lines_{ISO3}.parquet`
- `outputs_per_country/parquet/{scenario}/polylines_{ISO3}.parquet`

#### `process_country_siting.py`
Siting analysis for underserved remote settlements.

> **⚠️ Prerequisite:** Must run AFTER `process_country_supply.py` completes.

```bash
# Single scenario (100%)
python process_country_siting.py KEN

# All supply scenarios (100%, 90%, 80%, 70%, 60%)
python process_country_siting.py KEN --run-all-scenarios

# Single specific supply factor (e.g., 90% only)
python process_country_siting.py KEN --supply-factor 0.9
```

**Pipeline Steps:**
1. **Filter settlements** — Select "Partially Filled" or "Not Filled" status
2. **Geographic clustering** — DBSCAN with 50km threshold for isolated regions
3. **Capacity-driven K-means** — Cluster by remaining facility capacity
4. **Grid distance analysis** — Classify remote (>50km) vs near-grid
5. **Network design** — Minimum spanning tree for remote clusters
6. **Boundary clipping** — Ensure networks stay within country bounds

**Outputs:**
- `siting_clusters_{ISO3}.parquet` — Cluster centers with assignments
- `siting_networks_{ISO3}.parquet` — Network geometries
- `siting_summary_{ISO3}.xlsx` — Summary statistics

#### `generate_hpc_scripts.py`
Generate country list and SLURM batch scripts for HPC cluster execution.

```bash
# Generate country list only
python generate_hpc_scripts.py

# Generate 40 parallel supply analysis scripts
python generate_hpc_scripts.py --create-parallel

# Generate 25 parallel siting analysis scripts
python generate_hpc_scripts.py --create-parallel-siting
```

**Features:**
- Reads country list from energy demand data (`p1_b_ember_2024_30_50.xlsx`)
- Validates countries against GADM boundaries (excludes HKG, MAC, XKX)
- Groups countries into computational tiers (T1-T5) based on size/complexity
- Generates optimized SLURM scripts with appropriate resource allocation

**Generated Scripts:**
| Script | Description |
|--------|-------------|
| `submit_all_parallel.sh` | Submit all 40 supply analysis jobs |
| `submit_one.sh` | Submit individual supply script by number |
| `submit_all_parallel_siting.sh` | Submit all 25 siting analysis jobs |
| `submit_one_siting.sh` | Submit individual siting script by number |
| `submit_workflow.sh` | Combine results after all jobs complete |
| `parallel_scripts/*.sh` | 40 individual supply SLURM scripts |
| `parallel_scripts_siting/*.sh` | 25 individual siting SLURM scripts |

**Scenario Flags (all wrapper scripts support these):**
| Flag | Description |
|------|-------------|
| (none) | Run single scenario (100% supply factor) |
| `--run-all-scenarios` | Run all 5 scenarios (100%, 90%, 80%, 70%, 60%) |
| `--supply-factor 0.9` | Run single specific supply factor (e.g., 90%) |

---

### Results Processing

#### `combine_one_results.py`
Convert country Parquet files to GeoPackage for visualization.

```bash
# Basic (4 layers)
python combine_one_results.py KEN

# With siting layers (7 layers)
python combine_one_results.py KEN  # Auto-detects siting outputs

# Custom scenario
python combine_one_results.py KEN --scenario 2050_supply_100%
```

**Output:** `outputs_per_country/{scenario}_{ISO3}.gpkg`

**CMIP6 Climate Layers (auto-included if available):**
- Wind: `wpd`, `wpd_uncertainty`, `wpd_baseline`
- Solar: `pvout`, `pvout_uncertainty`, `pvout_baseline`
- Hydro: `runoff`, `runoff_uncertainty`, `runoff_baseline`
- Rivers: `riveratlas`, `riveratlas_baseline`

#### `combine_global_results.py`
Merge all country outputs into global GeoPackage.

```bash
# Auto-detect scenarios
python combine_global_results.py --input-dir outputs_per_country

# Specific scenario
python combine_global_results.py --scenario 2030_supply_100%

# Subset of countries
python combine_global_results.py --countries USA CHN IND
```

**Output:** `outputs_global/{scenario}_global.gpkg`

#### `p1_y_results_data_etl.py`
Generate exposure analysis dataset across scenarios.

**Dimensions:**
- Years: 2030, 2050
- Supply factors: 100%, 90%, 80%, 70%, 60%
- Buffer distances: 0km, 10km, 20km, 30km, 40km
- Energy types: All 6 categories

**Output:** `outputs_processed_data/exposure_analysis.parquet`

---

### Figure Generation

| Script | Output | Description |
|--------|--------|-------------|
| `p1_z_fig12.py` | Figures 1-2 | Global energy exposure stacked bars |
| `p1_z_fig34.py` | Figures 3-4 | Exposure by type and year |
| `p1_z_fig56.py` | Figures 5-6 | Detailed exposure analysis |
| `p1_z_fig7.py` | Figure 7 | Sensitivity heatmaps (3×6 grid) |
| `p1_z_fig8.py` | Figure 8 | Hazard-specific breakdown |

**Output directory:** `outputs_processed_fig/`

---

## Workflow Guide

### Complete Three-Step Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Supply Analysis                                        │
│  ─────────────────────────                                      │
│  process_country_supply.py                                      │
│                                                                 │
│  Outputs:                                                       │
│  └── 2030_supply_100%/                                          │
│      ├── centroids_{ISO3}.parquet                               │
│      ├── facilities_{ISO3}.parquet                              │
│      ├── grid_lines_{ISO3}.parquet                              │
│      └── polylines_{ISO3}.parquet                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Siting Analysis (Optional)                             │
│  ──────────────────────────────────                             │
│  process_country_siting.py                                      │
│                                                                 │
│  Outputs (same directory):                                      │
│  └── 2030_supply_100%/                                          │
│      ├── siting_clusters_{ISO3}.parquet    ← NEW                │
│      ├── siting_networks_{ISO3}.parquet    ← NEW                │
│      └── siting_summary_{ISO3}.xlsx        ← NEW                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ADD_V2 Integration (Optional)                          │
│  ─────────────────────────────────────                          │
│  Re-run process_country_supply.py (auto-detects siting)         │
│                                                                 │
│  Outputs:                                                       │
│  └── 2030_supply_100%_add_v2/                                   │
│      ├── centroids_{ISO3}_add_v2.parquet                        │
│      ├── facilities_{ISO3}_add_v2.parquet  ← Includes synthetic │
│      └── grid_lines_{ISO3}_add_v2.parquet  ← Includes networks  │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Kenya Analysis

```bash
# Step 1: Supply analysis
python process_country_supply.py KEN
# → Creates: outputs_per_country/parquet/2030_supply_100%/facilities_KEN.parquet

# Step 2: Siting analysis
python process_country_siting.py KEN
# → Creates: outputs_per_country/parquet/2030_supply_100%/siting_clusters_KEN.parquet

# Step 3: Integrated analysis (optional)
python process_country_supply.py KEN
# → Detects siting outputs, creates _add_v2 files

# Combine to GeoPackage
python combine_one_results.py KEN
# → Creates: outputs_per_country/2030_supply_100%_KEN_add_v2.gpkg
```

---

## Output Formats

### Parquet Files (Primary)

Efficient columnar storage for analysis pipelines.

**Location:** `outputs_per_country/parquet/{scenario}/`

**Layers:**
| File | Geometry | Key Attributes |
|------|----------|----------------|
| `centroids_{ISO3}.parquet` | Point | population, demand_mwh, supply_status, matched_facility |
| `facilities_{ISO3}.parquet` | Point | capacity_mw, generation_mwh, facility_type, num_merged |
| `grid_lines_{ISO3}.parquet` | LineString | distance_km, line_type, line_id |
| `polylines_{ISO3}.parquet` | LineString | centroid_id, facility_id, network_distance_km |

### GeoPackage Files (Visualization)

Multi-layer spatial database for GIS software (QGIS, ArcGIS).

**Per-country:** `outputs_per_country/{scenario}_{ISO3}.gpkg`
**Global:** `outputs_global/{scenario}_global.gpkg`

---

## High-Performance Computing (HPC)

### Resource Allocation

| Tier | Countries | CPUs | Memory | Time | Partition | Examples |
|------|-----------|------|--------|------|-----------|----------|
| **1** | 7 largest | 56 | 100GB | 12h | Short | USA, CHN, IND, RUS, BRA, CAN, AUS |
| **2** | 22 large | 40 | 100GB | 48h | Medium | ARG, KAZ, DZA, MEX, IDN, SDN |
| **3** | ~50 medium | 40 | 100GB | 12h | Short | KOR, FRA, DEU, JPN, GBR, ESP |
| **4** | ~110 small | 40 | 100GB | 12h | Short | Island nations, small countries |

### Complete HPC Workflow

```bash
# ═══════════════════════════════════════════════════════════════
# PREPARATION
# ═══════════════════════════════════════════════════════════════

# Generate parallel scripts
python generate_hpc_scripts.py --create-parallel         # 40 supply scripts
python generate_hpc_scripts.py --create-parallel-siting  # 25 siting scripts

# Fix line endings (if prepared on Windows)
sed -i 's/\r$//' submit_all_parallel.sh submit_one.sh parallel_scripts/*.sh
sed -i 's/\r$//' submit_all_parallel_siting.sh submit_one_siting.sh parallel_scripts_siting/*.sh
chmod +x submit_*.sh parallel_scripts/*.sh parallel_scripts_siting/*.sh

# ═══════════════════════════════════════════════════════════════
# STEP 1: SUPPLY ANALYSIS (~8-12 hours for single scenario)
# ═══════════════════════════════════════════════════════════════

# Single scenario (100% only - faster)
./submit_all_parallel.sh

# OR: All 5 scenarios (100%, 90%, 80%, 70%, 60%) - takes ~5x longer
./submit_all_parallel.sh --run-all-scenarios

# OR: Single specific scenario (e.g., 90% only)
./submit_all_parallel.sh --supply-factor 0.9

# Monitor
squeue -u $USER
tail -f outputs_global/logs/parallel_*.out

# Verify completion (~189 countries)
find outputs_per_country/parquet -name "facilities_*.parquet" | wc -l

# ═══════════════════════════════════════════════════════════════
# STEP 2: SITING ANALYSIS (~4-6 hours for single scenario)
# ═══════════════════════════════════════════════════════════════

# Single scenario
./submit_all_parallel_siting.sh

# OR: All 5 scenarios
./submit_all_parallel_siting.sh --run-all-scenarios

# OR: Single specific scenario (e.g., 90% only)
./submit_all_parallel_siting.sh --supply-factor 0.9

# Monitor
tail -f outputs_global/logs/siting_*.out

# Verify completion
find outputs_per_country/parquet -name "siting_clusters_*.parquet" | wc -l

# ═══════════════════════════════════════════════════════════════
# STEP 3: ADD_V2 INTEGRATION (Optional, ~8-12 hours)
# ═══════════════════════════════════════════════════════════════

./submit_all_parallel.sh  # Re-run supply to merge siting

# Verify _add_v2 files
find outputs_per_country/parquet -name "*_add_v2.parquet" | wc -l

# ═══════════════════════════════════════════════════════════════
# STEP 4: COMBINE RESULTS (~1-2 hours)
# ═══════════════════════════════════════════════════════════════

sbatch submit_workflow.sh

# Verify outputs
ls -lh outputs_global/*_global.gpkg
```

### Expected Timeline

| Phase | Single Scenario | All 5 Scenarios | Output |
|-------|-----------------|-----------------|--------|
| Supply Analysis | 8-12 hours | 40-60 hours | ~189 country parquets |
| Siting Analysis | 4-6 hours | 20-30 hours | ~150 siting parquets |
| ADD_V2 Integration | 8-12 hours | 40-60 hours | ~150 integrated parquets |
| Results Combination | 1-2 hours | 1-2 hours | Global GeoPackages |
| **Total (full)** | **21-32 hours** | **101-152 hours** | |
| **Total (no ADD_V2)** | **13-20 hours** | **61-92 hours** | |

> **Note**: Running `--run-all-scenarios` processes 5 supply factors (100%, 90%, 80%, 70%, 60%) sequentially per country, taking ~5x longer than single scenario.

### Single Job Submission

```bash
# Submit specific supply script (single scenario)
./submit_one.sh 06

# Submit specific supply script (all 5 scenarios)
./submit_one.sh 06 --run-all-scenarios

# Submit specific siting script (single scenario)
./submit_one_siting.sh 03

# Submit specific siting script (all 5 scenarios)
./submit_one_siting.sh 03 --run-all-scenarios
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `'\r': command not found` | Run `sed -i 's/\r$//' *.sh` on Linux |
| `Permission denied` | Run `chmod +x *.sh` |
| Memory errors | Check with `sacct -j <JOB_ID> --format=MaxRSS` |
| Missing country outputs | Check `countries_list.txt` and job logs |

### Siting Data Not Detected

```bash
# Verify siting outputs exist
ls outputs_per_country/parquet/2030_supply_100%/siting_summary_*.xlsx

# Check exact filename (case-sensitive)
# Must be: siting_summary_{ISO3}.xlsx
```

### Line Types Not Preserved

```python
# Verify columns in Parquet
import geopandas as gpd
gdf = gpd.read_parquet("grid_lines_KEN.parquet")
print(gdf.columns)
print(gdf['line_type'].unique())
# Expected: ['grid_infrastructure', 'siting_networks', 'component_stitch']
```

### Performance Issues

```bash
# Check parallelization in logs
grep "Using parallel processing" outputs_global/logs/parallel_*.out

# Verify CPU allocation
grep "MAX_WORKERS" outputs_global/logs/parallel_*.out
```

### Log Files

| Log | Location | Content |
|-----|----------|---------|
| Supply jobs | `outputs_global/logs/parallel_*.out` | Processing output |
| Siting jobs | `outputs_global/logs/siting_*.out` | Siting output |
| Combination | `outputs_global/logs/test_*.out` | Merge output |
| Errors | `*.err` files | Error messages |

---

## Data Schema

### Centroids Layer

| Column | Type | Description |
|--------|------|-------------|
| `geometry` | Point | Population centroid location |
| `Population_centroid` | int | Population at centroid |
| `Total_Demand_{year}_centroid` | float | Energy demand (MWh) |
| `supply_status` | str | "Filled", "Partially Filled", "Not Filled" |
| `matched_facility_id` | str | Assigned facility ID |
| `network_distance_km` | float | Distance to matched facility |
| `GID_0` | str | ISO3 country code |

### Facilities Layer

| Column | Type | Description |
|--------|------|-------------|
| `geometry` | Point | Facility location |
| `capacity_mw` | float | Generation capacity (MW) |
| `generation_mwh` | float | Annual generation (MWh) |
| `facility_type` | str | Solar, Wind, Hydro, etc. |
| `num_merged_units` | int | Number of clustered facilities |
| `remaining_capacity_mwh` | float | Unmatched capacity |
| `GID_0` | str | ISO3 country code |

### Grid Lines Layer

| Column | Type | Description |
|--------|------|-------------|
| `geometry` | LineString | Grid line geometry |
| `distance_km` | float | Line segment length |
| `line_type` | str | "grid_infrastructure", "siting_networks", "component_stitch" |
| `line_id` | str | Unique line identifier |
| `GID_0` | str | ISO3 country code |

---

## Performance Benchmarks

### Local Execution (16-core laptop)

| Country | Time | Memory |
|---------|------|--------|
| Small (TLS) | <5 min | <4GB |
| Medium (KEN) | 10-15 min | <8GB |
| Large (KOR) | 20-30 min | <16GB |
| Very Large (USA) | 1-2 hours | <32GB |

### Cluster Execution (40/56 CPUs)

| Country | Time | Memory |
|---------|------|--------|
| Large (CHN, USA) | 15-30 min | <100GB |
| Medium | 5-15 min | <50GB |
| Small | <5 min | <20GB |

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@software{electricity_supply_analysis,
  title = {Global Electricity Supply-Demand Analysis Framework},
  author = {[Author Names]},
  year = {2025},
  url = {[Repository URL]}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Data Sources**: GADM, Marine Regions, GridFinder, JRC GHSL, Global Solar Atlas, Global Wind Atlas, Ember, IEA, UN DESA
- **Computing**: [HPC Cluster Name] for computational resources
- **Funding**: [Funding Sources]
