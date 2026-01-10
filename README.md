# Global Electricity Supply-Demand Analysis Framework

A comprehensive geospatial analysis pipeline for modeling electricity supply networks, projecting future energy demand, and identifying optimal locations for renewable energy infrastructure across 190 countries.

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
├── p1_c_prep_landcover.py             # Download ESA CCI Land Cover 2022 from CDS
├── p1_d_viable_solar.py               # CMIP6 solar projections + viability filter
├── p1_e_viable_wind.py                # CMIP6 wind projections + viability filter
├── p1_f_utils_hydro.py                # Shared utilities for hydro processing
├── p1_f_viable_hydro.py               # ERA5-Land/CMIP6 runoff + RiverATLAS
│
├── # ═══ Core Analysis Scripts ═══
├── process_country_supply.py          # Main supply-demand network analysis
├── process_country_siting.py          # Remote settlement siting analysis
├── generate_hpc_scripts.py            # Generate country list and HPC scripts
│
├── # ═══ Results Processing ═══
├── combine_one_results.py             # Combine single country to GeoPackage + clip TIFs
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
├── submit_one_direct.sh               # Submit any single country directly
├── submit_one_direct_siting.sh        # Submit any single country siting directly
├── submit_workflow.sh                 # Submit results combination job
├── parallel_scripts/                  # 40 supply analysis SLURM scripts
├── parallel_scripts_siting/           # 24 siting analysis SLURM scripts
│
├── # ═══ Data Directories ═══
├── bigdata_gadm/                      # GADM administrative boundaries
├── bigdata_eez/                       # Marine Regions EEZ boundaries
├── bigdata_gridfinder/                # GridFinder electrical grid data
├── bigdata_settlements_jrc/           # JRC GHS-POP population raster
├── bigdata_landcover/                 # ESA CCI Land Cover 2022
├── bigdata_solar_pvout/               # Global Solar Atlas baseline
├── bigdata_wind_atlas/                # Global Wind Atlas baseline
├── bigdata_solar_wind_ms/             # Microsoft renewable energy sites
├── bigdata_landcover_cds/             # ESA CCI Land Cover 2022 (downloads/extracted/outputs)
├── bigdata_solar_cmip6/               # CMIP6 solar projections + outputs
│   └── outputs/                       # Solar TIFs + viable centroids
├── bigdata_wind_cmip6/                # CMIP6 wind projections + outputs
│   └── outputs/                       # Wind TIFs + viable centroids
├── bigdata_hydro_cmip6/               # CMIP6 runoff projections + outputs
│   └── outputs/                       # Hydro TIFs + river projections + viable centroids
├── bigdata_hydro_era5_land/           # ERA5-Land runoff data
├── bigdata_hydro_atlas/               # HydroATLAS river datasets
├── data_energy_ember/                 # Ember electricity statistics
├── data_energy_projections_iea/       # IEA World Energy Outlook data
├── data_pop_un/                       # UN population projections
├── data_country_class_wb/             # World Bank country classifications
│
├── # ═══ Output Directories ═══
├── outputs_per_country/               # Country-level Parquet + GeoPackage outputs
│   └── parquet/{scenario}/            # Parquet files per scenario
├── outputs_global/                    # Combined global GeoPackage outputs
├── outputs_processed_data/            # Processed analysis results
└── outputs_processed_fig/             # Generated figures
```---

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
tail -f outputs_per_country/logs/parallel_*.out
```

### Single Country (HPC Cluster)

Use `submit_one_direct.sh` for supply analysis or `submit_one_direct_siting.sh` for siting analysis.

```bash
# Submit any single country supply analysis (auto-detects tier and resources)
./submit_one_direct.sh KEN                      # Auto-detect tier, 100% scenario
./submit_one_direct.sh KEN --run-all-scenarios  # All 5 scenarios
./submit_one_direct.sh KEN --supply-factor 0.9  # Specific supply factor
./submit_one_direct.sh CHN --tier 1             # Override tier (use T1 resources)

# Submit any single country siting analysis
./submit_one_direct_siting.sh KEN
./submit_one_direct_siting.sh KEN --run-all-scenarios
./submit_one_direct_siting.sh KEN --supply-factor 0.9

# Check which batch script contains a specific country
grep -l "USA" parallel_scripts/*.sh
# Output: parallel_scripts/submit_parallel_06.sh
```

**Script-to-Country Mapping (Tier 1-2):**
| Script | Countries | Tier | Notes |
|--------|-----------|------|-------|
| 01 | CHN | T1 | 168h Long, ouce-cn64 (450GB dedicated) |
| 02 | USA | T2 | 168h Long |
| 03 | IND | T2 | 168h Long |
| 04 | BRA | T2 | 168h Long |
| 05 | DEU | T2 | 168h Long |
| 06 | FRA | T2 | 168h Long |
| 07+ | Multiple | T3-T5 | Medium/Small countries |

---

## Configuration

All configurable parameters are centralized in `config.py`:

### Data Regeneration Guide

When you modify configuration parameters, certain outputs need to be regenerated:

| Change | Scripts to Re-run |
|--------|-------------------|
| `POP_AGGREGATION_FACTOR` | `p1_d_viable_solar.py`, `p1_e_viable_wind.py`, `p1_f_viable_hydro.py`, then all country supply/siting |
| `SOLAR_PVOUT_THRESHOLD` | `p1_d_viable_solar.py --process-only` |
| `WIND_WPD_THRESHOLD` | `p1_e_viable_wind.py --process-only` |
| `HYDRO_MIN_DISCHARGE_VIABLE_M3S` | `p1_f_viable_hydro.py --process-only` |
| `LANDCOVER_VALID_*` | Respective viable script with `--process-only` |
| Network settings | Country supply analysis only (`process_country_supply.py`) |
| Siting settings | Country siting analysis only (`process_country_siting.py`) |

**Typical regeneration workflow:**
```bash
# After modifying viability thresholds:
python p1_d_viable_solar.py --process-only
python p1_e_viable_wind.py --process-only
python p1_f_viable_hydro.py --process-only

# Then re-run country analysis and combine:
python process_country_supply.py KEN
python combine_one_results.py KEN
```

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

> **Note**: After changing `POP_AGGREGATION_FACTOR`, regenerate resource outputs:
> ```bash
> python p1_d_viable_solar.py --process-only
> python p1_e_viable_wind.py --process-only
> python p1_f_viable_hydro.py --process-only
> ```

### Viability Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SOLAR_PVOUT_THRESHOLD` | 3.0 | Min PVOUT (kWh/kWp/day) for viable solar |
| `WIND_WPD_THRESHOLD` | 25 | Min WPD at 100m (W/m²) for viable wind |
| `HYDRO_MIN_DISCHARGE_VIABLE_M3S` | 1.0 | Min projected discharge (m³/s) for viable hydro |

### Land Cover Valid Classes (ESA CCI)

| Parameter | Classes | Description |
|-----------|---------|-------------|
| `LANDCOVER_VALID_SOLAR` | 10, 20, 30, 40, 130, 150, 200 | Cropland, grassland, sparse veg, bare |
| `LANDCOVER_VALID_WIND` | 10, 20, 30, 40, 130, 150, 200 | Same as solar (open terrain) |
| `LANDCOVER_VALID_HYDRO` | 160, 170, 180, 210 | Flooded areas, water bodies |

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
| **MS Solar Sites** | `bigdata_solar_wind_ms/solar_all_2024q2_v1.gpkg` | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) | Existing solar installations |
| **MS Wind Sites** | `bigdata_solar_wind_ms/wind_all_2024q2_v1.gpkg` | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) | Existing wind installations |
| **ESA Land Cover** | `bigdata_landcover_cds/outputs/landcover_2022_300arcsec.tif` | [CDS ERA5-Land](https://cds.climate.copernicus.eu/) | ESA CCI Land Cover 2022 (upscaled, GHS-POP aligned) |

### Energy Statistics

| Dataset | Path | Source |
|---------|------|--------|
| **Ember Data** | `data_energy_ember/yearly_full_release_*.csv` | [Ember](https://ember-climate.org/) |
| **IEA Projections** | `data_energy_projections_iea/WEO*.csv` | [IEA WEO 2024](https://www.iea.org/) |
| **UN Population** | `data_pop_un/` | [UN WPP 2024](https://population.un.org/) |
| **World Bank** | `data_country_class_wb/` | [World Bank](https://datahelpdesk.worldbank.org/) |

### Generated CMIP6 Outputs

These files are generated by the data preparation scripts and used by combine scripts:

**Solar (`bigdata_solar_cmip6/outputs/`):**
| File | Description |
|------|-------------|
| `PVOUT_2030_300arcsec.tif` | Projected PVOUT for 2030 (raw, no viability filter) |
| `PVOUT_2050_300arcsec.tif` | Projected PVOUT for 2050 (raw, no viability filter) |
| `PVOUT_baseline_300arcsec.tif` | Baseline PVOUT (Global Solar Atlas) |
| `PVOUT_UNCERTAINTY_2030_300arcsec.tif` | Ensemble range uncertainty for 2030 |
| `PVOUT_UNCERTAINTY_2050_300arcsec.tif` | Ensemble range uncertainty for 2050 |
| `SOLAR_VIABLE_CENTROIDS_2030.tif` | Viable solar cells raster for 2030 |
| `SOLAR_VIABLE_CENTROIDS_2050.tif` | Viable solar cells raster for 2050 |
| `SOLAR_VIABLE_CENTROIDS_2030.parquet` | Viable solar centroids (is_viable=True only) |
| `SOLAR_VIABLE_CENTROIDS_2050.parquet` | Viable solar centroids (is_viable=True only) |

**Wind (`bigdata_wind_cmip6/outputs/`):**
| File | Description |
|------|-------------|
| `WPD100_2030_300arcsec.tif` | Projected WPD at 100m for 2030 (raw, no viability filter) |
| `WPD100_2050_300arcsec.tif` | Projected WPD at 100m for 2050 (raw, no viability filter) |
| `WPD100_baseline_300arcsec.tif` | Baseline WPD (Global Wind Atlas) |
| `WPD100_UNCERTAINTY_2030_300arcsec.tif` | Ensemble range uncertainty for 2030 |
| `WPD100_UNCERTAINTY_2050_300arcsec.tif` | Ensemble range uncertainty for 2050 |
| `WIND_VIABLE_CENTROIDS_2030.tif` | Viable wind cells raster for 2030 |
| `WIND_VIABLE_CENTROIDS_2050.tif` | Viable wind cells raster for 2050 |
| `WIND_VIABLE_CENTROIDS_2030.parquet` | Viable wind centroids (is_viable=True only) |
| `WIND_VIABLE_CENTROIDS_2050.parquet` | Viable wind centroids (is_viable=True only) |

**Hydro (`bigdata_hydro_cmip6/outputs/`):**
| File | Description |
|------|-------------|
| `HYDRO_RUNOFF_baseline_300arcsec.tif` | Baseline runoff (ERA5-Land, mm/year) |
| `HYDRO_RUNOFF_DELTA_2030_300arcsec.tif` | Climate delta ratio for 2030 |
| `HYDRO_RUNOFF_DELTA_2050_300arcsec.tif` | Climate delta ratio for 2050 |
| `HYDRO_RUNOFF_UNCERTAINTY_2030_300arcsec.tif` | Ensemble range uncertainty for 2030 |
| `HYDRO_RUNOFF_UNCERTAINTY_2050_300arcsec.tif` | Ensemble range uncertainty for 2050 |
| `RiverATLAS_baseline_polyline.parquet` | River reaches with baseline discharge |
| `RiverATLAS_2030_polyline.parquet` | River reaches with projected 2030 discharge |
| `RiverATLAS_2050_polyline.parquet` | River reaches with projected 2050 discharge |
| `HYDRO_VIABLE_CENTROIDS_2030.parquet` | Viable hydro centroids for 2030 |
| `HYDRO_VIABLE_CENTROIDS_2050.parquet` | Viable hydro centroids for 2050 |

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

**Outputs:**
- `outputs_processed_data/p1_a_ember_gem_2024.xlsx` — Country-level aggregates (Granular_cur, Grouped_cur sheets)
- `outputs_processed_data/p1_a_ember_gem_2024_fac_lvl.xlsx` — Facility-level data (2024, 2030, 2050 sheets)

#### `p1_b_ember_2024_30_50.py`
Projects 2030 and 2050 electricity generation scenarios.

**Features:**
- Incorporates UN population growth factors
- Processes National Determined Contributions (NDCs)
- Applies IEA growth rates for fossil/nuclear
- Disaggregates broad renewable targets

**Output:** `data_facilities_gem/p1_b_ember_2024_30_50.xlsx`

#### `p1_c_prep_landcover.py`
Download ESA CCI Land Cover 2022 from Copernicus Climate Data Store.

**Features:**
- Downloads global land cover at ~300m resolution (10 arcsec native)
- Converts NetCDF to GeoTIFF format
- Upscales to 300 arcsec with GHS-POP grid alignment (mode resampling)
- Used for viability filtering in solar/wind/hydro scripts

**Output:**
- `bigdata_landcover_cds/extracted/C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.nc` (raw NetCDF)
- `bigdata_landcover_cds/outputs/landcover_2022_10arcsec.tif` (native resolution)
- `bigdata_landcover_cds/outputs/landcover_2022_300arcsec.tif` (upscaled, GHS-POP aligned)

#### `p1_d_viable_solar.py` / `p1_e_viable_wind.py`
Generate CMIP6-based climate projections for solar and wind resources with viability filtering.

**Viability Filter Logic:**
A cell (300 arcsec) is considered viable if:
1. **MS site present** — Microsoft renewable energy dataset shows existing installation, OR
2. **Land cover valid AND resource >= threshold** — ESA CCI land cover is suitable AND resource value meets minimum threshold

**Thresholds (configurable in `config.py`):**
- Solar: `SOLAR_PVOUT_THRESHOLD = 3.0` kWh/kWp/day
- Wind: `WIND_WPD_THRESHOLD = 25` W/m²

**Data Sources:**
- **Microsoft Viable Sites**: `bigdata_solar_wind_ms/solar_all_2024q2_v1.gpkg` (polygons), `wind_all_2024q2_v1.gpkg` (points)
- **ESA CCI Land Cover**: Classes 10-40 (cropland), 130 (grassland), 150 (sparse vegetation), 200 (bare areas)
- **CMIP6 Models**: CESM2, EC-Earth3-veg-lr, MPI-ESM1-2-lr (ensemble mean + IQR uncertainty)

**Method:**
1. Download CMIP6 ensemble data for historical + SSP245
2. Calculate delta: Δ = Future_period / Historical_period
3. Apply to baseline: Future = Baseline × Δ
4. Apply viability filter: MS_present OR (landcover_valid AND resource >= threshold)
5. Compute uncertainty (interquartile range)

**Usage:**
```bash
# Download only
python p1_d_viable_solar.py --download-only

# Process only (assumes downloads exist)
python p1_d_viable_solar.py --process-only

# Full pipeline
python p1_d_viable_solar.py
```

**Solar Outputs (`bigdata_solar_cmip6/outputs/`):**
- GeoTIFF rasters (raw resource, no viability filter):
  - `PVOUT_{2030,2050}_300arcsec.tif` — Projected PVOUT (climate delta applied)
  - `PVOUT_baseline_300arcsec.tif` — Baseline PVOUT (Global Solar Atlas)
  - `PVOUT_UNCERTAINTY_{2030,2050}_300arcsec.tif` — Ensemble range uncertainty
- GeoTIFF rasters (viability-filtered):
  - `SOLAR_VIABLE_CENTROIDS_{2030,2050}.tif` — Viable cells only (0 = not viable)
- Parquet centroids (raw resource, no viability filter):
  - `PVOUT_{2030,2050}_300arcsec.parquet` — All cells with resource value > 0
- Parquet centroids (viability-filtered, matches TIF):
  - `SOLAR_VIABLE_CENTROIDS_{2030,2050}.parquet` — Only viable cells

**Viable Centroids Parquet Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `geometry` | Point | Pixel center coordinate (WGS84) |
| `source` | string | Resource type ("solar", "wind", "hydro") |
| `value_{year}` | float | Projected resource value for target year |
| `value_baseline` | float | Baseline resource value |
| `delta` | float | Climate change ratio (projected / baseline) |
| `uncertainty` | float | Ensemble range (max - min) |
| `is_ms_viable` | bool | True if MS renewable site present |
| `is_lc_valid` | bool | True if land cover class is valid |
| `meets_threshold` | bool | True if resource ≥ threshold |
| `is_viable` | bool | True (always, by construction: filtered) |

**Wind Outputs (`bigdata_wind_cmip6/outputs/`):**
- GeoTIFF rasters (raw resource, no viability filter):
  - `WPD100_{2030,2050}_300arcsec.tif` — Projected WPD at 100m
  - `WPD100_baseline_300arcsec.tif` — Baseline WPD (Global Wind Atlas)
  - `WPD100_UNCERTAINTY_{2030,2050}_300arcsec.tif` — Ensemble range uncertainty
- GeoTIFF rasters (viability-filtered):
  - `WIND_VIABLE_CENTROIDS_{2030,2050}.tif` — Viable cells only (0 = not viable)
- Parquet centroids (raw resource, no viability filter):
  - `WPD100_{2030,2050}_300arcsec.parquet` — All cells with resource value > 0
- Parquet centroids (viability-filtered, matches TIF):
  - `WIND_VIABLE_CENTROIDS_{2030,2050}.parquet` — Only viable cells (same schema as solar)

#### `p1_f_viable_hydro.py`
Unified hydro processing: ERA5-Land/CMIP6 runoff delta calculation + RiverATLAS river discharge projections.

**Data Sources:**
- **Runoff Baseline**: ERA5-Land monthly runoff (reanalysis, 0.1° resolution)
- **Climate Projections**: CMIP6 `total_runoff` (SSP2-4.5 scenario)
- **River Network**: HydroATLAS RiverATLAS river reach dataset
- **Land Cover**: ESA CCI Land Cover 2022 (water/wetland classes)

**Processing Parts:**

1. **Part 1: Runoff Delta Calculation**
   - Download ERA5-Land runoff baseline (1995-2014)
   - Download CMIP6 total_runoff for historical + SSP245
   - Compute delta: Δ = CMIP6_future / CMIP6_historical
   - Compute uncertainty (model range)
   - Regrid to 300 arcsec (aligned with GHS-POP)
   - Output: Delta + uncertainty rasters (TIF + Parquet)

2. **Part 2: RiverATLAS Projections**
   - Load RiverATLAS river reaches (filtered by min discharge + stream order)
   - Sample delta raster at polyline centroids
   - Apply delta to baseline discharge: `dis_m3_pyr_2030 = dis_m3_pyr × delta`
   - Output: Projected river polylines (Parquet)

3. **Part 3: Viable Hydro Centroids**
   - Create point centroids from RiverATLAS polylines
   - Filter by minimum discharge threshold (≥ 1.0 m³/s)
   - Sample land cover at centroid coordinates
   - Filter by valid hydro land cover classes (160, 170, 180, 210)
   - Output: Viable centroids (Parquet with point geometry)

**Viability Filter Logic:**
A river reach centroid is considered viable if:
1. **Projected discharge ≥ threshold** — `dis_m3_pyr_projected >= HYDRO_MIN_DISCHARGE_VIABLE_M3S`, AND
2. **Land cover valid** — ESA CCI class in [160, 170, 180, 210] (flooded/water)

**CMIP6 Models:**
- CESM2, EC-Earth3-veg-lr, MPI-ESM1-2-lr (ensemble mean + range uncertainty)

**Usage:**
```bash
# Download all data
python p1_f_viable_hydro.py --download-only

# Process only (assumes downloads exist)
python p1_f_viable_hydro.py --process-only

# Full pipeline
python p1_f_viable_hydro.py

# With RiverATLAS filters
python p1_f_viable_hydro.py --min-discharge 1.0 --min-order 4
```

**Hydro Outputs (`bigdata_hydro_cmip6/outputs/`):**

*Part 1 - Delta Rasters (GeoTIFF + Parquet):*
| File | Description |
|------|-------------|
| `HYDRO_RUNOFF_baseline_300arcsec.*` | ERA5-Land baseline runoff (mm/year) |
| `HYDRO_RUNOFF_DELTA_{2030,2050}_300arcsec.*` | Climate delta ratio (future/historical) |
| `HYDRO_RUNOFF_UNCERTAINTY_{2030,2050}_300arcsec.*` | Ensemble range (max - min) |

*Part 2 - River Polylines (Parquet with LineString geometry):*
| File | Description |
|------|-------------|
| `RiverATLAS_baseline_polyline.parquet` | River reaches with baseline discharge |
| `RiverATLAS_{2030,2050}_polyline.parquet` | River reaches with projected discharge |

*Part 3 - Viable Centroids (Parquet with Point geometry):*
| File | Description |
|------|-------------|
| `HYDRO_VIABLE_CENTROIDS_{2030,2050}.parquet` | Viable hydro site centroids |

**RiverATLAS Polyline Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `HYRIV_ID` | int | Unique river reach identifier |
| `geometry` | LineString | River reach polyline (WGS84) |
| `dis_m3_pyr` | float | Baseline annual discharge (m³/s) |
| `delta_{year}` | float | Climate change ratio |
| `dis_m3_pyr_{year}` | float | Projected discharge (m³/s) |
| `dis_change_pct_{year}` | float | Percent change from baseline |
| `ORD_STRA` | int | Strahler stream order |

**Viable Centroids Parquet Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `HYRIV_ID` | int | River reach identifier |
| `geometry` | Point | Polyline centroid (WGS84) |
| `dis_m3_pyr` | float | Baseline discharge (m³/s) |
| `delta` | float | Climate change ratio |
| `dis_m3_pyr_projected` | float | Projected discharge (m³/s) |
| `ORD_STRA` | int | Stream order |
| `landcover_class` | int | ESA CCI land cover class |

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
- `{YEAR}_siting_{FACTOR}%_{ISO3}.xlsx` — Summary statistics (e.g., `2030_siting_100%_KEN.xlsx`)

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
| `submit_one_direct.sh` | Submit any single country supply analysis |
| `submit_all_parallel_siting.sh` | Submit all 25 siting analysis jobs |
| `submit_one_direct_siting.sh` | Submit any single country siting analysis |
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
# Basic (4 layers) - from parquet/2030_supply_100%/
python combine_one_results.py KEN

# With siting layers (7 layers) - auto-detects siting_*.parquet files
python combine_one_results.py KEN  # Creates {scenario}_{ISO3}_add.gpkg

# With _add_v2 (after 2nd supply run) - auto-detects _add_v2 folder
python combine_one_results.py KEN  # Creates {scenario}_{ISO3}_add_v2.gpkg

# Custom scenario (both work the same - auto-detects _add_v2 folder)
python combine_one_results.py KEN --scenario 2030_supply_100%
python combine_one_results.py KEN --scenario 2030_supply_100%_add_v2
```

**Auto-detection logic:**
1. Checks `parquet/{scenario}_add_v2/` folder first for `*_add_v2.parquet` files
2. Falls back to `parquet/{scenario}/` folder
3. Output filename: `{scenario}_{ISO3}.gpkg`, `_add.gpkg`, or `_add_v2.gpkg` based on available files

**Output:** `outputs_per_country/{scenario}_{ISO3}[_add|_add_v2].gpkg`

**Layers included:**
- Core supply analysis: `centroids`, `facilities`, `grid_lines`, `polylines`
- Siting analysis (if available): `siting_clusters`, `siting_networks`
- Viable centroids (CMIP6-based, if available): `SOLAR_VIABLE_CENTROIDS_{year}`, `WIND_VIABLE_CENTROIDS_{year}`, `HYDRO_VIABLE_CENTROIDS_{year}`

**CMIP6 Climate TIF Layers (auto-clipped if global TIFs exist):**

The combine script automatically clips 12 CMIP6 TIF layers to the country extent for each target year:

| Layer | Description | Source |
|-------|-------------|--------|
| `PVOUT_{year}` | Projected solar PVOUT (kWh/kWp/day) | p1_d_viable_solar.py |
| `PVOUT_{year}_uncertainty` | IQR uncertainty from CMIP6 ensemble | p1_d_viable_solar.py |
| `PVOUT_baseline` | Baseline PVOUT from Global Solar Atlas | p1_d_viable_solar.py |
| `SOLAR_VIABLE_CENTROIDS_{year}` | Viable solar cells raster | p1_d_viable_solar.py |
| `WPD100_{year}` | Projected wind power density (W/m²) | p1_e_viable_wind.py |
| `WPD100_{year}_uncertainty` | IQR uncertainty from CMIP6 ensemble | p1_e_viable_wind.py |
| `WPD100_baseline` | Baseline WPD from Global Wind Atlas | p1_e_viable_wind.py |
| `WIND_VIABLE_CENTROIDS_{year}` | Viable wind cells raster | p1_e_viable_wind.py |
| `HYDRO_RUNOFF_baseline` | Baseline runoff from ERA5-Land | p1_f_viable_hydro.py |
| `HYDRO_RUNOFF_DELTA_{year}` | Climate delta ratio for rivers | p1_f_viable_hydro.py |
| `HYDRO_RUNOFF_UNCERTAINTY_{year}` | Ensemble range uncertainty | p1_f_viable_hydro.py |

> **Note**: GPKG raster layers are visible in QGIS but may not display in ArcGIS. For ArcGIS users, the global TIF files in `bigdata_*/outputs/` directories can be used directly.

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
│      └── {YEAR}_siting_{FACTOR}%_{ISO3}.xlsx  ← NEW             │
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
│      ├── grid_lines_{ISO3}_add_v2.parquet  ← Includes networks  │
│      └── polylines_{ISO3}_add_v2.parquet   ← Updated routes     │
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
# → Detects siting outputs, creates _add_v2 files in separate folder:
#   outputs_per_country/parquet/2030_supply_100%_add_v2/

# Combine to GeoPackage (auto-detects _add_v2 folder)
python combine_one_results.py KEN
# → Auto-detects parquet/2030_supply_100%_add_v2/ folder
# → Creates: outputs_per_country/2030_supply_100%_KEN_add_v2.gpkg

# Or explicitly specify scenario (same result)
python combine_one_results.py KEN --scenario 2030_supply_100%_add_v2
# → Creates: outputs_per_country/2030_supply_100%_KEN_add_v2.gpkg
```

> **Note:** The `_add_v2` parquet files are saved to a separate folder (`2030_supply_100%_add_v2/`).
> The combine script auto-detects this folder, so you can use either `--scenario 2030_supply_100%`
> or `--scenario 2030_supply_100%_add_v2` - both will find the _add_v2 files.

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

| Tier | Countries | CPUs | Memory | Time | Partition | Node | Examples |
|------|-----------|------|--------|------|-----------|------|----------|
| **1** | CHN | 40 | 450GB | 168h | Long | ouce-cn64 | China (dedicated node) |
| **2** | 5 large | 40 | 95GB | 168h | Long | - | USA, IND, BRA, DEU, FRA |
| **3** | 11 medium-large | 40 | 95GB | 48h | Medium | - | CAN, MEX, RUS, AUS, ARG, etc. |
| **4** | 20 medium | 40 | 95GB | 12h | Short | - | TUR, NGA, COL, PAK, VEN, etc. (2/script) |
| **5** | ~156 small | 40 | 25GB | 12h | Short | - | All others (12/script) |

### Complete HPC Workflow

```bash
# ═══════════════════════════════════════════════════════════════
# PREPARATION
# ═══════════════════════════════════════════════════════════════

# Generate parallel scripts
python generate_hpc_scripts.py --create-parallel         # 40 supply scripts
python generate_hpc_scripts.py --create-parallel-siting  # 25 siting scripts

# Fix line endings (if prepared on Windows)
sed -i 's/\r$//' submit_all_parallel.sh submit_one_direct.sh parallel_scripts/*.sh
sed -i 's/\r$//' submit_all_parallel_siting.sh submit_one_direct_siting.sh parallel_scripts_siting/*.sh
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
tail -f outputs_per_country/logs/parallel_*.out

# Verify completion (~190 countries)
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
tail -f outputs_per_country/logs/siting_*.out

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
| Supply Analysis | 8-12 hours | 40-60 hours | ~190 country parquets |
| Siting Analysis | 4-6 hours | 20-30 hours | ~190 siting parquets |
| ADD_V2 Integration | 8-12 hours | 40-60 hours | ~190 integrated parquets |
| Results Combination | 1-2 hours | 1-2 hours | Global GeoPackages |
| **Total (full)** | **21-32 hours** | **101-152 hours** | |
| **Total (no ADD_V2)** | **13-20 hours** | **61-92 hours** | |

> **Note**: Running `--run-all-scenarios` processes 5 supply factors (100%, 90%, 80%, 70%, 60%) sequentially per country, taking ~5x longer than single scenario.

### Single Job Submission

```bash
# Submit any single country supply analysis (auto-detects tier)
./submit_one_direct.sh KEN

# Submit single country with all 5 scenarios
./submit_one_direct.sh KEN --run-all-scenarios

# Submit any single country siting analysis
./submit_one_direct_siting.sh KEN

# Submit siting with all 5 scenarios
./submit_one_direct_siting.sh KEN --run-all-scenarios
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
ls outputs_per_country/parquet/2030_supply_100%/2030_siting_100%_*.xlsx

# Check exact filename (case-sensitive)
# Must be: {YEAR}_siting_{FACTOR}%_{ISO3}.xlsx (e.g., 2030_siting_100%_KEN.xlsx)
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
grep "Using parallel processing" outputs_per_country/logs/parallel_*.out

# Verify CPU allocation
grep "MAX_WORKERS" outputs_per_country/logs/parallel_*.out
```

### Log Files

| Log | Location | Content |
|-----|----------|---------|  
| Supply jobs | `outputs_per_country/logs/parallel_*.out` | Processing output |
| Siting jobs | `outputs_per_country/logs/siting_*.out` | Siting output |
| Combination | `outputs_per_country/logs/workflow_*.out` | Merge output |
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
