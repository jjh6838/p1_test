"""
p1_f_viable_hydro.py
====================
Unified Hydro Processing: ERA5-Land/CMIP6 Runoff + RiverATLAS Projections

This script calculates runoff deltas from ERA5-Land and CMIP6 climate projections,
then applies those deltas to RiverATLAS river reach discharge values.

Methodology (3 Parts):
----------------------
PART 1: ERA5-Land + CMIP6 Runoff Delta Calculation
  - Download ERA5-Land runoff baseline (1995-2014)
  - Download CMIP6 total_runoff for historical + SSP245
  - Compute delta: Δ = CMIP6_future / CMIP6_historical
  - Regrid delta to 300 arcsec (aligned with GHS-POP)
  - Output: Delta rasters (GeoTIFF + Parquet)

PART 2: RiverATLAS Projections
  - Load RiverATLAS river reaches with discharge (dis_m3_pyr)
  - Sample delta raster at polyline centroids
  - Apply delta to baseline discharge: dis_m3_pyr_2030 = dis_m3_pyr * delta
  - Output: Projected polylines (Parquet) - keeps polyline geometry

PART 3: Viable Hydro Centroids
  - Create point centroids from projected RiverATLAS
  - Sample land cover raster at centroid coordinates
  - Filter by valid hydro land cover classes (water/wetland: 160, 170, 180, 210)
  - Output: Viable centroids (Parquet) - point geometry

Output Files:
-------------
Part 1 (Delta Rasters):
  - HYDRO_RUNOFF_baseline_300arcsec.tif / .parquet
  - HYDRO_RUNOFF_DELTA_2030_300arcsec.tif / .parquet
  - HYDRO_RUNOFF_DELTA_2050_300arcsec.tif / .parquet
  - HYDRO_RUNOFF_UNCERTAINTY_2030_300arcsec.tif / .parquet
  - HYDRO_RUNOFF_UNCERTAINTY_2050_300arcsec.tif / .parquet

Part 2 (River Polylines):
  - RiverATLAS_baseline_polyline.parquet
  - RiverATLAS_2030_polyline.parquet
  - RiverATLAS_2050_polyline.parquet

Part 3 (Viable Centroids):
  - HYDRO_VIABLE_CENTROIDS_2030.parquet
  - HYDRO_VIABLE_CENTROIDS_2050.parquet

Dependencies:
  - p1_c_cds_landcover.py (needs landcover_2022.tif for filtering)
  - RiverATLAS_Data_v10.gdb in bigdata_hydro_atlas/
"""

import argparse
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point

from config import (
    TARGET_RESOLUTION_ARCSEC,
    LANDCOVER_VALID_HYDRO,
    HYDRO_MIN_DISCHARGE_VIABLE_M3S,
)
from p1_f_utils_hydro import (
    get_bigdata_path,
    CMIP6_MODELS, HIST_PERIOD, P2030, P2050, SECONDS_PER_YEAR,
    RIVERATLAS_COLUMNS, MIN_DISCHARGE_M3S, MIN_STREAM_ORDER,
    download_era5_land, download_cmip6_historical, download_cmip6_ssp245, extract_zip,
    load_era5_land_runoff, load_cmip6_runoff, load_riveratlas,
    era5_to_mm_per_year, compute_temporal_mean, compute_delta_ratio, apply_delta_to_era5,
    regrid_to_target, save_geotiff, save_as_parquet, get_ghs_pop_grid_params,
    extract_delta_at_points,
)

# Paths
LANDCOVER_PATH = Path(get_bigdata_path("bigdata_landcover_cds")) / "outputs" / "landcover_2022_300arcsec.tif"


# =============================================================================
# PART 0: DOWNLOAD DATA
# =============================================================================

def run_part0_download(cmip6_dir: Path, era5_dir: Path):
    """Download all required runoff datasets."""
    print("\n" + "="*70)
    print("DOWNLOADING RUNOFF DATA")
    print("="*70)
    
    download_era5_land(era5_dir)
    
    for model in CMIP6_MODELS:
        print(f"\n--- {model} ---")
        download_cmip6_historical(model, cmip6_dir)
        download_cmip6_ssp245(model, cmip6_dir)
    
    print("\n[done] All downloads complete!")


# =============================================================================
# PART 1: RUNOFF DELTA CALCULATION (ERA5-Land + CMIP6)
# =============================================================================

def run_part1_delta(cmip6_dir: Path, era5_dir: Path, out_dir: Path):
    """
    Process runoff data using delta method.
    
    Outputs:
      - HYDRO_RUNOFF_baseline_*.tif / .parquet (ERA5-Land baseline)
      - HYDRO_RUNOFF_DELTA_2030_*.tif / .parquet (CMIP6/ERA5 ratio)
      - HYDRO_RUNOFF_DELTA_2050_*.tif / .parquet (CMIP6/ERA5 ratio)
    """
    print("\n" + "="*70)
    print("PART 1: RUNOFF DELTA CALCULATION (ERA5-Land + CMIP6)")
    print("="*70)
    
    cmip6_ex_dir = cmip6_dir.parent / "extracted"
    
    # Load ERA5-Land baseline
    print("\n--- Loading ERA5-Land baseline ---")
    era5_file = era5_dir / "era5_land_runoff_1995-2014.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5-Land not found: {era5_file}. Run with --download first.")
    
    era5_runoff = load_era5_land_runoff(era5_file)
    era5_annual = era5_to_mm_per_year(era5_runoff)
    print(f"  ERA5-Land baseline: {float(era5_annual.mean()):.1f} mm/year")
    
    # Process each CMIP6 model
    delta_2030_list, delta_2050_list = [], []
    successful_models = []
    
    for model in CMIP6_MODELS:
        print(f"\n--- Processing: {model} ---")
        try:
            # Historical
            hist_zip = cmip6_dir / f"{model}_historical_runoff_1995-2014.zip"
            if not hist_zip.exists():
                print(f"  [skip] Historical not found")
                continue
            
            hist_nc = extract_zip(hist_zip, cmip6_ex_dir / "historical" / model)
            cmip6_hist = load_cmip6_runoff(hist_nc)
            cmip6_hist_mean = compute_temporal_mean(cmip6_hist, *HIST_PERIOD)
            
            # SSP245
            ssp_zip = cmip6_dir / f"{model}_ssp245_runoff_2021-2060.zip"
            if not ssp_zip.exists():
                print(f"  [skip] SSP245 not found")
                continue
            
            ssp_nc = extract_zip(ssp_zip, cmip6_ex_dir / "ssp245" / model)
            cmip6_ssp = load_cmip6_runoff(ssp_nc)
            
            cmip6_2030_mean = compute_temporal_mean(cmip6_ssp, *P2030)
            cmip6_2050_mean = compute_temporal_mean(cmip6_ssp, *P2050)
            
            # Compute deltas (CMIP6 future / CMIP6 historical)
            delta_2030 = compute_delta_ratio(cmip6_2030_mean, cmip6_hist_mean)
            delta_2050 = compute_delta_ratio(cmip6_2050_mean, cmip6_hist_mean)
            
            # Interpolate deltas to ERA5 grid for consistent output
            delta_2030_interp = delta_2030.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            delta_2050_interp = delta_2050.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            
            delta_2030_list.append(delta_2030_interp)
            delta_2050_list.append(delta_2050_interp)
            successful_models.append(model)
            print(f"  [done] {model}")
            
        except Exception as e:
            print(f"  [ERROR] {model}: {e}")
            traceback.print_exc()
    
    if not delta_2030_list:
        raise RuntimeError("No CMIP6 models processed!")
    
    print(f"\n--- Processed {len(successful_models)} models: {successful_models} ---")
    
    # Ensemble statistics
    print("\n--- Computing ensemble statistics ---")
    delta_2030_stack = xr.concat(delta_2030_list, dim="model")
    delta_2050_stack = xr.concat(delta_2050_list, dim="model")
    
    delta_2030_ens = delta_2030_stack.mean("model", skipna=True)
    delta_2050_ens = delta_2050_stack.mean("model", skipna=True)
    
    # Uncertainty: range across models (max - min)
    delta_unc_2030 = delta_2030_stack.max(dim="model") - delta_2030_stack.min(dim="model")
    delta_unc_2050 = delta_2050_stack.max(dim="model") - delta_2050_stack.min(dim="model")
    
    print(f"  Delta 2030 mean: {float(delta_2030_ens.mean()):.3f} (uncertainty: {float(delta_unc_2030.mean()):.3f})")
    print(f"  Delta 2050 mean: {float(delta_2050_ens.mean()):.3f} (uncertainty: {float(delta_unc_2050.mean()):.3f})")
    
    # Regrid to target resolution
    print(f"\n--- Regridding to {TARGET_RESOLUTION_ARCSEC} arcsec ---")
    baseline_data, lons, lats = regrid_to_target(era5_annual, TARGET_RESOLUTION_ARCSEC)
    delta_2030_data, _, _ = regrid_to_target(delta_2030_ens, TARGET_RESOLUTION_ARCSEC)
    delta_2050_data, _, _ = regrid_to_target(delta_2050_ens, TARGET_RESOLUTION_ARCSEC)
    unc_2030_data, _, _ = regrid_to_target(delta_unc_2030, TARGET_RESOLUTION_ARCSEC)
    unc_2050_data, _, _ = regrid_to_target(delta_unc_2050, TARGET_RESOLUTION_ARCSEC)
    
    print(f"  Output shape: {delta_2030_data.shape}")
    
    # Save outputs (GeoTIFF)
    print("\n--- Saving Part 1 outputs (GeoTIFF) ---")
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    
    save_geotiff(baseline_data, lons, lats, out_dir / f"HYDRO_RUNOFF_baseline_{suffix}.tif")
    save_geotiff(delta_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_DELTA_2030_{suffix}.tif", nodata=1.0)
    save_geotiff(delta_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_DELTA_2050_{suffix}.tif", nodata=1.0)
    save_geotiff(unc_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.tif", nodata=0)
    save_geotiff(unc_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.tif", nodata=0)
    
    # Save outputs (Parquet)
    print("\n--- Saving Part 1 outputs (Parquet) ---")
    save_as_parquet(baseline_data, lons, lats, out_dir / f"HYDRO_RUNOFF_baseline_{suffix}.parquet", "runoff_mm")
    # Delta values can be < 1.0 (decreasing runoff), so don't filter positive only
    save_as_parquet(delta_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_DELTA_2030_{suffix}.parquet", "delta", nodata=1.0, filter_positive=False)
    save_as_parquet(delta_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_DELTA_2050_{suffix}.parquet", "delta", nodata=1.0, filter_positive=False)
    save_as_parquet(unc_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.parquet", "uncertainty")
    save_as_parquet(unc_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.parquet", "uncertainty")
    
    return delta_2030_data, delta_2050_data, lons, lats


# =============================================================================
# PART 2: RIVERATLAS PROJECTIONS
# =============================================================================

def run_part2_riveratlas_projections(riveratlas_path: Path, out_dir: Path,
                                      min_discharge: float, min_order: int,
                                      bbox: tuple = None, save_gpkg: bool = False):
    """
    Load RiverATLAS, sample delta at centroids, project discharge.
    
    Outputs:
      - RiverATLAS_baseline.parquet (polyline geometry)
      - RiverATLAS_projected_2030.parquet (polyline geometry)
      - RiverATLAS_projected_2050.parquet (polyline geometry)
    """
    print("\n" + "="*70)
    print("PART 2: RIVERATLAS PROJECTIONS")
    print("="*70)
    
    # Load RiverATLAS
    print("\n--- Loading RiverATLAS ---")
    if not riveratlas_path.exists():
        print(f"[ERROR] RiverATLAS not found: {riveratlas_path}")
        return None
    
    gdf = load_riveratlas(riveratlas_path, RIVERATLAS_COLUMNS, min_discharge, min_order, bbox)
    if len(gdf) == 0:
        print("[ERROR] No river reaches loaded!")
        return None
    
    print(f"  Loaded {len(gdf):,} river reaches")
    
    # Load delta rasters
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    delta_2030_path = out_dir / f"HYDRO_RUNOFF_DELTA_2030_{suffix}.tif"
    delta_2050_path = out_dir / f"HYDRO_RUNOFF_DELTA_2050_{suffix}.tif"
    
    if not delta_2030_path.exists() or not delta_2050_path.exists():
        print("[ERROR] Delta rasters not found - run Part 1 first!")
        print(f"  Expected: {delta_2030_path.name}")
        print(f"  Expected: {delta_2050_path.name}")
        return None
    
    print("\n--- Loading delta rasters ---")
    with rasterio.open(delta_2030_path) as src:
        delta_2030_data = src.read(1)
        delta_2030_transform = src.transform
    with rasterio.open(delta_2050_path) as src:
        delta_2050_data = src.read(1)
        delta_2050_transform = src.transform
    
    # Extract deltas at river reach centroids
    print("\n--- Extracting delta at river reach centroids ---")
    gdf = extract_delta_at_points(gdf, delta_2030_data, delta_2030_transform, 'delta_2030')
    gdf = extract_delta_at_points(gdf, delta_2050_data, delta_2050_transform, 'delta_2050')
    
    # Compute projected discharge
    print("\n--- Computing projected discharge ---")
    gdf['dis_m3_pyr_2030'] = gdf['dis_m3_pyr'] * gdf['delta_2030']
    gdf['dis_m3_pyr_2050'] = gdf['dis_m3_pyr'] * gdf['delta_2050']
    gdf['dis_change_pct_2030'] = (gdf['delta_2030'] - 1.0) * 100
    gdf['dis_change_pct_2050'] = (gdf['delta_2050'] - 1.0) * 100
    
    print(f"  Baseline mean discharge: {gdf['dis_m3_pyr'].mean():,.2f} m³/s")
    print(f"  2030 mean discharge: {gdf['dis_m3_pyr_2030'].mean():,.2f} m³/s ({gdf['dis_change_pct_2030'].mean():+.1f}%)")
    print(f"  2050 mean discharge: {gdf['dis_m3_pyr_2050'].mean():,.2f} m³/s ({gdf['dis_change_pct_2050'].mean():+.1f}%)")
    
    # Save outputs (polyline geometry preserved)
    print("\n--- Saving Part 2 outputs (Parquet with polyline geometry) ---")
    
    cols_baseline = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx',
                     'run_mm_cyr', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav',
                     'ele_mt_cav', 'ele_mt_uav', 'LENGTH_KM']
    cols_baseline = [c for c in cols_baseline if c in gdf.columns]
    gdf[cols_baseline].to_parquet(out_dir / "RiverATLAS_baseline_polyline.parquet")
    print(f"  Saved: RiverATLAS_baseline_polyline.parquet ({len(gdf):,} reaches)")
    
    cols_2030 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2030', 'dis_m3_pyr_2030', 
                 'dis_change_pct_2030', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']
    cols_2030 = [c for c in cols_2030 if c in gdf.columns]
    gdf[cols_2030].to_parquet(out_dir / "RiverATLAS_2030_polyline.parquet")
    print(f"  Saved: RiverATLAS_2030_polyline.parquet ({len(gdf):,} reaches)")
    
    cols_2050 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2050', 'dis_m3_pyr_2050',
                 'dis_change_pct_2050', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']
    cols_2050 = [c for c in cols_2050 if c in gdf.columns]
    gdf[cols_2050].to_parquet(out_dir / "RiverATLAS_2050_polyline.parquet")
    print(f"  Saved: RiverATLAS_2050_polyline.parquet ({len(gdf):,} reaches)")
    
    if save_gpkg:
        gpkg_cols = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2030', 'delta_2050',
                     'dis_m3_pyr_2030', 'dis_m3_pyr_2050']
        gpkg_cols = [c for c in gpkg_cols if c in gdf.columns]
        gdf[gpkg_cols].to_file(out_dir / "RiverATLAS_polyline.gpkg", driver="GPKG")
        print(f"  Saved: RiverATLAS_polyline.gpkg")
    
    return gdf


# =============================================================================
# PART 3: VIABLE HYDRO CENTROIDS
# =============================================================================

def run_part3_viable_centroids(out_dir: Path, landcover_path: Path = None,
                                min_discharge_viable: float = None):
    """
    Create viable hydro centroids from projected RiverATLAS.
    
    Process:
      1. Load RiverATLAS_*_polyline.parquet
      2. Create point centroids from polylines
      3. Filter by minimum projected discharge threshold
      4. Sample land cover at centroid coordinates
      5. Filter by valid hydro land cover classes (160, 170, 180, 210)
    
    Outputs:
      - HYDRO_VIABLE_CENTROIDS_2030.parquet (point geometry)
      - HYDRO_VIABLE_CENTROIDS_2050.parquet (point geometry)
    """
    print("\n" + "="*70)
    print("PART 3: VIABLE HYDRO CENTROIDS")
    print("="*70)
    
    if landcover_path is None:
        landcover_path = LANDCOVER_PATH
    if min_discharge_viable is None:
        min_discharge_viable = HYDRO_MIN_DISCHARGE_VIABLE_M3S
    
    # Load projected RiverATLAS
    river_2030_path = out_dir / "RiverATLAS_2030_polyline.parquet"
    river_2050_path = out_dir / "RiverATLAS_2050_polyline.parquet"
    
    if not river_2030_path.exists() or not river_2050_path.exists():
        print("[ERROR] RiverATLAS polyline files not found - run Part 2 first!")
        return None, None
    
    print("\n--- Loading projected RiverATLAS ---")
    river_2030 = gpd.read_parquet(river_2030_path)
    river_2050 = gpd.read_parquet(river_2050_path)
    print(f"  2030: {len(river_2030):,} reaches")
    print(f"  2050: {len(river_2050):,} reaches")
    
    # Create centroids from polylines
    print("\n--- Creating centroids from polylines ---")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*geographic CRS.*centroid.*')
        centroids_2030 = river_2030.geometry.centroid
        centroids_2050 = river_2050.geometry.centroid
    
    # Create centroid GeoDataFrames
    viable_2030 = gpd.GeoDataFrame({
        'HYRIV_ID': river_2030['HYRIV_ID'].values,
        'geometry': centroids_2030,
        'dis_m3_pyr': river_2030['dis_m3_pyr'].values,
        'delta': river_2030['delta_2030'].values,
        'dis_m3_pyr_projected': river_2030['dis_m3_pyr_2030'].values,
        'ORD_STRA': river_2030['ORD_STRA'].values if 'ORD_STRA' in river_2030.columns else np.nan,
    }, crs="EPSG:4326")
    
    viable_2050 = gpd.GeoDataFrame({
        'HYRIV_ID': river_2050['HYRIV_ID'].values,
        'geometry': centroids_2050,
        'dis_m3_pyr': river_2050['dis_m3_pyr'].values,
        'delta': river_2050['delta_2050'].values,
        'dis_m3_pyr_projected': river_2050['dis_m3_pyr_2050'].values,
        'ORD_STRA': river_2050['ORD_STRA'].values if 'ORD_STRA' in river_2050.columns else np.nan,
    }, crs="EPSG:4326")
    
    print(f"  Created {len(viable_2030):,} centroids for 2030")
    print(f"  Created {len(viable_2050):,} centroids for 2050")
    
    # Filter by minimum discharge threshold
    print(f"\n--- Filtering by discharge threshold ({min_discharge_viable} m³/s) ---")
    
    discharge_mask_2030 = viable_2030['dis_m3_pyr_projected'] >= min_discharge_viable
    discharge_mask_2050 = viable_2050['dis_m3_pyr_projected'] >= min_discharge_viable
    
    print(f"  2030: {discharge_mask_2030.sum():,} / {len(viable_2030):,} centroids meet discharge threshold")
    print(f"  2050: {discharge_mask_2050.sum():,} / {len(viable_2050):,} centroids meet discharge threshold")
    
    viable_2030 = viable_2030[discharge_mask_2030].reset_index(drop=True)
    viable_2050 = viable_2050[discharge_mask_2050].reset_index(drop=True)
    
    # Sample land cover at centroids
    if landcover_path.exists():
        print(f"\n--- Sampling land cover at centroids ---")
        print(f"  Valid hydro classes: {LANDCOVER_VALID_HYDRO}")
        
        with rasterio.open(landcover_path) as src:
            lc_data = src.read(1)
            lc_transform = src.transform
            
            # Sample for 2030
            coords_2030 = np.array([(p.x, p.y) for p in viable_2030.geometry])
            rows_2030 = ((coords_2030[:, 1] - lc_transform.f) / lc_transform.e).astype(int)
            cols_2030 = ((coords_2030[:, 0] - lc_transform.c) / lc_transform.a).astype(int)
            rows_2030 = np.clip(rows_2030, 0, lc_data.shape[0] - 1)
            cols_2030 = np.clip(cols_2030, 0, lc_data.shape[1] - 1)
            lc_values_2030 = lc_data[rows_2030, cols_2030]
            
            # Sample for 2050
            coords_2050 = np.array([(p.x, p.y) for p in viable_2050.geometry])
            rows_2050 = ((coords_2050[:, 1] - lc_transform.f) / lc_transform.e).astype(int)
            cols_2050 = ((coords_2050[:, 0] - lc_transform.c) / lc_transform.a).astype(int)
            rows_2050 = np.clip(rows_2050, 0, lc_data.shape[0] - 1)
            cols_2050 = np.clip(cols_2050, 0, lc_data.shape[1] - 1)
            lc_values_2050 = lc_data[rows_2050, cols_2050]
        
        # Add land cover class to dataframes
        viable_2030['landcover_class'] = lc_values_2030
        viable_2050['landcover_class'] = lc_values_2050
        
        # Filter by valid land cover classes
        mask_2030 = np.isin(lc_values_2030, LANDCOVER_VALID_HYDRO)
        mask_2050 = np.isin(lc_values_2050, LANDCOVER_VALID_HYDRO)
        
        print(f"  2030: {mask_2030.sum():,} / {len(viable_2030):,} centroids have valid land cover")
        print(f"  2050: {mask_2050.sum():,} / {len(viable_2050):,} centroids have valid land cover")
        
        viable_2030 = viable_2030[mask_2030].reset_index(drop=True)
        viable_2050 = viable_2050[mask_2050].reset_index(drop=True)
    else:
        print(f"\n[WARNING] Land cover file not found: {landcover_path}")
        print("  Skipping land cover filter - all centroids marked as viable")
    
    # Save outputs
    print("\n--- Saving Part 3 outputs (Parquet with point geometry) ---")
    viable_2030.to_parquet(out_dir / "HYDRO_VIABLE_CENTROIDS_2030.parquet")
    viable_2050.to_parquet(out_dir / "HYDRO_VIABLE_CENTROIDS_2050.parquet")
    print(f"  Saved: HYDRO_VIABLE_CENTROIDS_2030.parquet ({len(viable_2030):,} points)")
    print(f"  Saved: HYDRO_VIABLE_CENTROIDS_2050.parquet ({len(viable_2050):,} points)")
    
    return viable_2030, viable_2050


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run(workdir: str = None, era5_dir: str = None, riveratlas_path: str = None,
        download_only: bool = False, process_only: bool = False,
        min_discharge: float = None, min_order: int = None, bbox: tuple = None,
        save_gpkg: bool = False):
    """Main entry point."""
    
    # Default paths
    if workdir is None:
        workdir = Path(get_bigdata_path("bigdata_hydro_cmip6"))
    else:
        workdir = Path(workdir)
    
    if era5_dir is None:
        era5_dir = Path(get_bigdata_path("bigdata_hydro_era5_land")) / "downloads"
    else:
        era5_dir = Path(era5_dir)
    
    if riveratlas_path is None:
        riveratlas_path = Path(get_bigdata_path("bigdata_hydro_atlas")) / "RiverATLAS_Data_v10.gdb" / "RiverATLAS_v10.gdb"
    else:
        riveratlas_path = Path(riveratlas_path)
    
    cmip6_dir = workdir / "downloads"
    out_dir = workdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if min_discharge is None:
        min_discharge = MIN_DISCHARGE_M3S
    if min_order is None:
        min_order = MIN_STREAM_ORDER
    
    print("="*70)
    print("UNIFIED HYDRO PROCESSING: RUNOFF DELTA + RIVERATLAS")
    print("="*70)
    
    # Download only
    if download_only:
        run_part0_download(cmip6_dir, era5_dir)
        return
    
    # Full or process-only pipeline
    
    # Part 1: Calculate runoff deltas
    delta_2030, delta_2050, lons, lats = run_part1_delta(cmip6_dir, era5_dir, out_dir)
    
    # Part 2: Project RiverATLAS discharge using deltas
    river_gdf = run_part2_riveratlas_projections(
        riveratlas_path, out_dir, min_discharge, min_order, bbox, save_gpkg
    )
    
    # Part 3: Generate viable centroids with land cover filter
    viable_2030, viable_2050 = run_part3_viable_centroids(out_dir, LANDCOVER_PATH)
    
    print(f"\n{'='*70}")
    print("ALL PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Hydro Processing: Runoff Delta + RiverATLAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data
  python p1_f_viable_hydro.py --download-only
  
  # Process only (assumes downloads exist)
  python p1_f_viable_hydro.py --process-only
  
  # Full pipeline
  python p1_f_viable_hydro.py
  
  # With filters
  python p1_f_viable_hydro.py --min-discharge 1.0 --min-order 4
  python p1_f_viable_hydro.py --bbox -20 -40 55 40  # Africa
        """
    )
    
    parser.add_argument("--workdir", default=None, help="Working directory")
    parser.add_argument("--era5-dir", default=None, help="ERA5-Land downloads directory")
    parser.add_argument("--riveratlas", default=None, help="Path to RiverATLAS GDB")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument("--process-only", action="store_true", help="Only process (skip download)")
    parser.add_argument("--min-discharge", type=float, default=None, help="Min discharge (m³/s)")
    parser.add_argument("--min-order", type=int, default=None, help="Min stream order")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=('MINX','MINY','MAXX','MAXY'), help="Bounding box")
    parser.add_argument("--save-gpkg", action="store_true", help="Save GeoPackage outputs")
    
    args = parser.parse_args()
    
    run(
        workdir=args.workdir,
        era5_dir=args.era5_dir,
        riveratlas_path=args.riveratlas,
        download_only=args.download_only,
        process_only=args.process_only,
        min_discharge=args.min_discharge,
        min_order=args.min_order,
        bbox=tuple(args.bbox) if args.bbox else None,
        save_gpkg=args.save_gpkg,
    )
