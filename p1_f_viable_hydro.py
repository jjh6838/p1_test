"""
p1_f_viable_hydro.py
====================
Unified Hydro Processing: ERA5-Land/CMIP6 Runoff + RiverATLAS Projections

This script combines runoff-based and river-based hydro potential into a single
workflow, eliminating the circular dependency between the previous separate scripts.

Methodology (4 Parts):
----------------------
PART 1: River Proximity Mask
  - Load RiverATLAS river reaches with discharge (dis_m3_pyr)
  - Generate 5km river proximity mask (used by Part 2)
  - Output: river_proximity_mask_5km.tif + rivers GeoDataFrame (in memory)

PART 2: ERA5-Land + CMIP6 Runoff Projections
  - Download ERA5-Land runoff baseline (1995-2014)
  - Download CMIP6 total_runoff for historical + SSP245
  - Compute delta: Δ = CMIP6_future / CMIP6_historical
  - Apply delta to ERA5-Land baseline
  - Regrid to 300 arcsec (aligned with GHS-POP)
  - Apply hydro filter (water/wetland OR river proximity)
  - Output: Raster GeoTIFFs + delta grids

PART 3: RiverATLAS Projections
  - Extract delta values at river reach centroids (from Part 2)
  - Apply delta to baseline discharge
  - Output: Projected polylines (Parquet)

PART 4: Viable Hydro Centroids
  - Combine runoff-based centroids (from Part 2 rasters)
  - Combine river-based centroids (from Part 3 reaches)
  - Apply runoff threshold filter
  - Output: Unified viable centroids (Parquet)

Output Files:
-------------
Part 1 (River Mask):
  - river_proximity_mask_5km.tif

Part 2 (Runoff Rasters):
  - HYDRO_RUNOFF_baseline_300arcsec.tif
  - HYDRO_RUNOFF_2030_300arcsec.tif
  - HYDRO_RUNOFF_2050_300arcsec.tif
  - HYDRO_RUNOFF_UNCERTAINTY_2030_300arcsec.tif
  - HYDRO_RUNOFF_UNCERTAINTY_2050_300arcsec.tif
  - HYDRO_ATLAS_DELTA_2030_300arcsec.tif
  - HYDRO_ATLAS_DELTA_2050_300arcsec.tif

Part 3 (River Polylines):
  - RiverATLAS_baseline.parquet
  - RiverATLAS_projected_2030.parquet
  - RiverATLAS_projected_2050.parquet

Part 4 (Viable Centroids):
  - HYDRO_VIABLE_CENTROIDS_2030.parquet
  - HYDRO_VIABLE_CENTROIDS_2050.parquet

Dependencies:
  - p1_c_cds_landcover.py (needs landcover_2022.tif for filtering)
  - RiverATLAS_Data_v10.gdb in bigdata_hydro_atlas/
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

from config import (
    TARGET_RESOLUTION_ARCSEC,
    HYDRO_RIVER_BUFFER_M,
    LANDCOVER_VALID_HYDRO,
    HYDRO_RUNOFF_THRESHOLD_MM,
)
from p1_f_utils_hydro import (
    get_bigdata_path,
    CMIP6_MODELS, HIST_PERIOD, P2030, P2050, SECONDS_PER_YEAR,
    RIVERATLAS_COLUMNS, MIN_DISCHARGE_M3S, MIN_STREAM_ORDER,
    download_era5_land, download_cmip6_historical, download_cmip6_ssp245, extract_zip,
    load_era5_land_runoff, load_cmip6_runoff, load_riveratlas,
    era5_to_mm_per_year, compute_temporal_mean, compute_delta_ratio, apply_delta_to_era5,
    regrid_to_target, save_geotiff, save_as_parquet, get_ghs_pop_grid_params,
    create_river_proximity_mask, apply_hydro_filter, extract_delta_at_points,
)

# Paths
LANDCOVER_PATH = Path(get_bigdata_path("bigdata_landcover_cds")) / "outputs" / "landcover_2022_300arcsec.tif"


# =============================================================================
# PART 2: RUNOFF PROCESSING (ERA5-Land + CMIP6)
# =============================================================================

def run_part1_download(cmip6_dir: Path, era5_dir: Path):
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


def run_part2_runoff(cmip6_dir: Path, era5_dir: Path, out_dir: Path, river_mask_path: Path):
    """Process runoff data using delta method."""
    print("\n" + "="*70)
    print("PART 2: RUNOFF PROJECTIONS (ERA5-Land + CMIP6)")
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
    runoff_2030_list, runoff_2050_list = [], []
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
            
            # Compute deltas
            delta_2030 = compute_delta_ratio(cmip6_2030_mean, cmip6_hist_mean)
            delta_2050 = compute_delta_ratio(cmip6_2050_mean, cmip6_hist_mean)
            
            # Apply to ERA5
            runoff_2030 = apply_delta_to_era5(era5_annual, delta_2030)
            runoff_2050 = apply_delta_to_era5(era5_annual, delta_2050)
            
            # Interpolate deltas to ERA5 grid
            delta_2030_interp = delta_2030.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            delta_2050_interp = delta_2050.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            
            delta_2030_list.append(delta_2030_interp)
            delta_2050_list.append(delta_2050_interp)
            runoff_2030_list.append(runoff_2030)
            runoff_2050_list.append(runoff_2050)
            successful_models.append(model)
            print(f"  [done] {model}")
            
        except Exception as e:
            print(f"  [ERROR] {model}: {e}")
            traceback.print_exc()
    
    if not runoff_2030_list:
        raise RuntimeError("No CMIP6 models processed!")
    
    print(f"\n--- Processed {len(successful_models)} models: {successful_models} ---")
    
    # Ensemble statistics
    print("\n--- Computing ensemble statistics ---")
    runoff_2030_ens = xr.concat(runoff_2030_list, dim="model").mean("model", skipna=True)
    runoff_2050_ens = xr.concat(runoff_2050_list, dim="model").mean("model", skipna=True)
    
    runoff_2030_stack = xr.concat(runoff_2030_list, dim="model")
    runoff_2050_stack = xr.concat(runoff_2050_list, dim="model")
    unc_2030 = runoff_2030_stack.max(dim="model") - runoff_2030_stack.min(dim="model")
    unc_2050 = runoff_2050_stack.max(dim="model") - runoff_2050_stack.min(dim="model")
    
    delta_2030_ens = xr.concat(delta_2030_list, dim="model").mean("model", skipna=True)
    delta_2050_ens = xr.concat(delta_2050_list, dim="model").mean("model", skipna=True)
    
    # Regrid to target resolution
    print(f"\n--- Regridding to {TARGET_RESOLUTION_ARCSEC} arcsec ---")
    baseline_data, lons, lats = regrid_to_target(era5_annual, TARGET_RESOLUTION_ARCSEC)
    runoff_2030_data, _, _ = regrid_to_target(runoff_2030_ens, TARGET_RESOLUTION_ARCSEC)
    runoff_2050_data, _, _ = regrid_to_target(runoff_2050_ens, TARGET_RESOLUTION_ARCSEC)
    unc_2030_data, _, _ = regrid_to_target(unc_2030, TARGET_RESOLUTION_ARCSEC)
    unc_2050_data, _, _ = regrid_to_target(unc_2050, TARGET_RESOLUTION_ARCSEC)
    delta_2030_data, _, _ = regrid_to_target(delta_2030_ens, TARGET_RESOLUTION_ARCSEC)
    delta_2050_data, _, _ = regrid_to_target(delta_2050_ens, TARGET_RESOLUTION_ARCSEC)
    
    print(f"  Output shape: {runoff_2030_data.shape}")
    
    # Apply hydro filter: river_proximity OR (landcover_valid AND runoff >= threshold)
    print("\n--- Applying hydro viability filter ---")
    baseline_data = apply_hydro_filter(baseline_data, lons, lats, LANDCOVER_PATH, river_mask_path, 
                                        LANDCOVER_VALID_HYDRO, resource_threshold=HYDRO_RUNOFF_THRESHOLD_MM, verbose=True)
    runoff_2030_data = apply_hydro_filter(runoff_2030_data, lons, lats, LANDCOVER_PATH, river_mask_path, 
                                           LANDCOVER_VALID_HYDRO, resource_threshold=HYDRO_RUNOFF_THRESHOLD_MM, verbose=False)
    runoff_2050_data = apply_hydro_filter(runoff_2050_data, lons, lats, LANDCOVER_PATH, river_mask_path, 
                                           LANDCOVER_VALID_HYDRO, resource_threshold=HYDRO_RUNOFF_THRESHOLD_MM, verbose=False)
    # Uncertainty rasters: use threshold=0 to keep all cells that pass viability
    unc_2030_data = apply_hydro_filter(unc_2030_data, lons, lats, LANDCOVER_PATH, river_mask_path, 
                                        LANDCOVER_VALID_HYDRO, resource_threshold=0, verbose=False)
    unc_2050_data = apply_hydro_filter(unc_2050_data, lons, lats, LANDCOVER_PATH, river_mask_path, 
                                        LANDCOVER_VALID_HYDRO, resource_threshold=0, verbose=False)
    
    print(f"  Applied filter to 5 rasters (baseline, 2030, 2050, uncertainty)")
    
    # Save outputs (GeoTIFF)
    print("\n--- Saving Part 2 outputs (GeoTIFF) ---")
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    
    save_geotiff(baseline_data, lons, lats, out_dir / f"HYDRO_RUNOFF_baseline_{suffix}.tif")
    save_geotiff(runoff_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2030_{suffix}.tif")
    save_geotiff(runoff_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2050_{suffix}.tif")
    save_geotiff(unc_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.tif")
    save_geotiff(unc_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.tif")
    save_geotiff(delta_2030_data, lons, lats, out_dir / f"HYDRO_ATLAS_DELTA_2030_{suffix}.tif", nodata=1.0)
    save_geotiff(delta_2050_data, lons, lats, out_dir / f"HYDRO_ATLAS_DELTA_2050_{suffix}.tif", nodata=1.0)
    
    # Save outputs (Parquet)
    print("\n--- Saving Part 2 outputs (Parquet) ---")
    save_as_parquet(baseline_data, lons, lats, out_dir / f"HYDRO_RUNOFF_baseline_{suffix}.parquet", "runoff_mm")
    save_as_parquet(runoff_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2030_{suffix}.parquet", "runoff_mm")
    save_as_parquet(runoff_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2050_{suffix}.parquet", "runoff_mm")
    save_as_parquet(unc_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.parquet", "uncertainty")
    save_as_parquet(unc_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.parquet", "uncertainty")
    
    return baseline_data, runoff_2030_data, runoff_2050_data, unc_2030_data, unc_2050_data, lons, lats


# =============================================================================
# PARTS 1 & 3: RIVERATLAS (MASK + PROJECTIONS)
# =============================================================================

def run_part1_river_mask(riveratlas_path: Path, out_dir: Path,
                         min_discharge: float, min_order: int, bbox: tuple):
    """Load RiverATLAS and generate river proximity mask."""
    print("\n" + "="*70)
    print("PART 1: RIVER PROXIMITY MASK")
    print("="*70)
    
    # Load RiverATLAS
    print("\n--- Loading RiverATLAS ---")
    if not riveratlas_path.exists():
        print(f"[ERROR] RiverATLAS not found: {riveratlas_path}")
        return None, None
    
    gdf = load_riveratlas(riveratlas_path, RIVERATLAS_COLUMNS, min_discharge, min_order, bbox)
    if len(gdf) == 0:
        print("[ERROR] No river reaches loaded!")
        return None, None
    
    # Create river proximity mask
    mask_path = out_dir / f"river_proximity_mask_{int(HYDRO_RIVER_BUFFER_M/1000)}km.tif"
    create_river_proximity_mask(gdf, mask_path, HYDRO_RIVER_BUFFER_M, TARGET_RESOLUTION_ARCSEC)
    
    print(f"\n  Rivers loaded: {len(gdf):,} reaches")
    print(f"  Mask saved: {mask_path.name}")
    
    return mask_path, gdf


def run_part3_riveratlas_projections(gdf: gpd.GeoDataFrame, out_dir: Path, save_gpkg: bool = False):
    """Apply delta projections to RiverATLAS discharge."""
    print("\n" + "="*70)
    print("PART 3: RIVERATLAS PROJECTIONS")
    print("="*70)
    
    if gdf is None or len(gdf) == 0:
        print("[ERROR] No river data to process!")
        return None
    
    # Load delta rasters (generated by Part 2)
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    delta_2030_path = out_dir / f"HYDRO_ATLAS_DELTA_2030_{suffix}.tif"
    delta_2050_path = out_dir / f"HYDRO_ATLAS_DELTA_2050_{suffix}.tif"
    
    if not delta_2030_path.exists() or not delta_2050_path.exists():
        print("[ERROR] Delta rasters not found - run Part 2 first!")
        print(f"  Expected: {delta_2030_path.name}")
        print(f"  Expected: {delta_2050_path.name}")
        return None
    
    # Extract deltas at river reaches
    print("\n--- Loading delta rasters ---")
    import rasterio
    with rasterio.open(delta_2030_path) as src:
        delta_2030_data = src.read(1)
        delta_2030_transform = src.transform
    with rasterio.open(delta_2050_path) as src:
        delta_2050_data = src.read(1)
        delta_2050_transform = src.transform
    
    print("\n--- Extracting delta at river reaches ---")
    gdf = extract_delta_at_points(gdf, delta_2030_data, delta_2030_transform, 'delta_2030')
    gdf = extract_delta_at_points(gdf, delta_2050_data, delta_2050_transform, 'delta_2050')
    
    # Compute projected discharge
    print("\n--- Computing projected discharge ---")
    gdf['dis_m3_pyr_2030'] = gdf['dis_m3_pyr'] * gdf['delta_2030']
    gdf['dis_m3_pyr_2050'] = gdf['dis_m3_pyr'] * gdf['delta_2050']
    gdf['dis_change_pct_2030'] = (gdf['delta_2030'] - 1.0) * 100
    gdf['dis_change_pct_2050'] = (gdf['delta_2050'] - 1.0) * 100
    
    print(f"  Baseline mean: {gdf['dis_m3_pyr'].mean():,.2f} m³/s")
    print(f"  2030 mean: {gdf['dis_m3_pyr_2030'].mean():,.2f} m³/s ({gdf['dis_change_pct_2030'].mean():+.1f}%)")
    print(f"  2050 mean: {gdf['dis_m3_pyr_2050'].mean():,.2f} m³/s ({gdf['dis_change_pct_2050'].mean():+.1f}%)")
    
    # Save outputs
    print("\n--- Saving Part 3 outputs ---")
    
    cols_baseline = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx',
                     'run_mm_cyr', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav',
                     'ele_mt_cav', 'ele_mt_uav', 'LENGTH_KM']
    cols_baseline = [c for c in cols_baseline if c in gdf.columns]
    gdf[cols_baseline].to_parquet(out_dir / "RiverATLAS_baseline.parquet")
    
    cols_2030 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2030', 'dis_m3_pyr_2030', 
                 'dis_change_pct_2030', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']
    cols_2030 = [c for c in cols_2030 if c in gdf.columns]
    gdf[cols_2030].to_parquet(out_dir / "RiverATLAS_projected_2030.parquet")
    
    cols_2050 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2050', 'dis_m3_pyr_2050',
                 'dis_change_pct_2050', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']
    cols_2050 = [c for c in cols_2050 if c in gdf.columns]
    gdf[cols_2050].to_parquet(out_dir / "RiverATLAS_projected_2050.parquet")
    
    print(f"  Saved: RiverATLAS_baseline.parquet")
    print(f"  Saved: RiverATLAS_projected_2030.parquet")
    print(f"  Saved: RiverATLAS_projected_2050.parquet")
    
    if save_gpkg:
        gpkg_cols = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2030', 'delta_2050',
                     'dis_m3_pyr_2030', 'dis_m3_pyr_2050']
        gpkg_cols = [c for c in gpkg_cols if c in gdf.columns]
        gdf[gpkg_cols].to_file(out_dir / "RiverATLAS_projected.gpkg", driver="GPKG")
        print(f"  Saved: RiverATLAS_projected.gpkg")
    
    return gdf


# =============================================================================
# PART 4: VIABLE CENTROIDS
# =============================================================================

def run_part4_centroids(runoff_baseline: np.ndarray, runoff_2030: np.ndarray, runoff_2050: np.ndarray,
                        unc_2030: np.ndarray, unc_2050: np.ndarray,
                        lons: np.ndarray, lats: np.ndarray,
                        river_gdf: gpd.GeoDataFrame, out_dir: Path,
                        runoff_threshold: float = HYDRO_RUNOFF_THRESHOLD_MM):
    """Create unified viable hydro centroids from runoff rasters and river reaches."""
    print("\n" + "="*70)
    print("PART 4: VIABLE HYDRO CENTROIDS")
    print("="*70)
    
    import warnings
    
    # Compute deltas from rasters
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_2030 = np.where(runoff_baseline > 0, runoff_2030 / runoff_baseline, 1.0)
        delta_2050 = np.where(runoff_baseline > 0, runoff_2050 / runoff_baseline, 1.0)
    
    # -------------------------------------------------------------------------
    # Runoff-based centroids
    # -------------------------------------------------------------------------
    print(f"\n--- Creating runoff-based centroids (threshold: {runoff_threshold} mm/year) ---")
    
    if lats[0] > lats[-1]:
        lats_asc = lats[::-1]
        runoff_baseline_asc = runoff_baseline[::-1, :]
        runoff_2030_asc = runoff_2030[::-1, :]
        runoff_2050_asc = runoff_2050[::-1, :]
        delta_2030_asc = delta_2030[::-1, :]
        delta_2050_asc = delta_2050[::-1, :]
        unc_2030_asc = unc_2030[::-1, :]
        unc_2050_asc = unc_2050[::-1, :]
    else:
        lats_asc = lats
        runoff_baseline_asc = runoff_baseline
        runoff_2030_asc = runoff_2030
        runoff_2050_asc = runoff_2050
        delta_2030_asc = delta_2030
        delta_2050_asc = delta_2050
        unc_2030_asc = unc_2030
        unc_2050_asc = unc_2050
    
    lon_grid, lat_grid = np.meshgrid(lons, lats_asc)
    
    # For 2030
    valid_2030 = (runoff_2030_asc > runoff_threshold) & ~np.isnan(runoff_2030_asc)
    runoff_centroids_2030 = gpd.GeoDataFrame({
        'geometry': gpd.points_from_xy(lon_grid[valid_2030], lat_grid[valid_2030]),
        'source': 'runoff',
        'runoff_mm': runoff_2030_asc[valid_2030],
        'runoff_mm_baseline': runoff_baseline_asc[valid_2030],
        'dis_m3_pyr': np.nan,
        'delta': delta_2030_asc[valid_2030],
        'value_2030': runoff_2030_asc[valid_2030],
        'uncertainty': unc_2030_asc[valid_2030],
    }, crs="EPSG:4326")
    
    # For 2050
    valid_2050 = (runoff_2050_asc > runoff_threshold) & ~np.isnan(runoff_2050_asc)
    runoff_centroids_2050 = gpd.GeoDataFrame({
        'geometry': gpd.points_from_xy(lon_grid[valid_2050], lat_grid[valid_2050]),
        'source': 'runoff',
        'runoff_mm': runoff_2050_asc[valid_2050],
        'runoff_mm_baseline': runoff_baseline_asc[valid_2050],
        'dis_m3_pyr': np.nan,
        'delta': delta_2050_asc[valid_2050],
        'value_2050': runoff_2050_asc[valid_2050],
        'uncertainty': unc_2050_asc[valid_2050],
    }, crs="EPSG:4326")
    
    print(f"  Runoff centroids 2030: {len(runoff_centroids_2030):,}")
    print(f"  Runoff centroids 2050: {len(runoff_centroids_2050):,}")
    
    # -------------------------------------------------------------------------
    # River-based centroids
    # -------------------------------------------------------------------------
    river_centroids_2030 = None
    river_centroids_2050 = None
    
    if river_gdf is not None and len(river_gdf) > 0:
        print("\n--- Creating river-based centroids ---")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*geographic CRS.*centroid.*')
            river_points = river_gdf.geometry.centroid
        
        # 2030
        if 'delta_2030' in river_gdf.columns:
            river_centroids_2030 = gpd.GeoDataFrame({
                'geometry': river_points,
                'source': 'river',
                'runoff_mm': np.nan,
                'runoff_mm_baseline': np.nan,
                'dis_m3_pyr': river_gdf['dis_m3_pyr'].values,
                'delta': river_gdf['delta_2030'].values,
                'value_2030': river_gdf['dis_m3_pyr_2030'].values if 'dis_m3_pyr_2030' in river_gdf.columns else river_gdf['dis_m3_pyr'].values * river_gdf['delta_2030'].values,
                'uncertainty': np.nan,
            }, crs="EPSG:4326")
            print(f"  River centroids 2030: {len(river_centroids_2030):,}")
        
        # 2050
        if 'delta_2050' in river_gdf.columns:
            river_centroids_2050 = gpd.GeoDataFrame({
                'geometry': river_points,
                'source': 'river',
                'runoff_mm': np.nan,
                'runoff_mm_baseline': np.nan,
                'dis_m3_pyr': river_gdf['dis_m3_pyr'].values,
                'delta': river_gdf['delta_2050'].values,
                'value_2050': river_gdf['dis_m3_pyr_2050'].values if 'dis_m3_pyr_2050' in river_gdf.columns else river_gdf['dis_m3_pyr'].values * river_gdf['delta_2050'].values,
                'uncertainty': np.nan,
            }, crs="EPSG:4326")
            print(f"  River centroids 2050: {len(river_centroids_2050):,}")
    
    # -------------------------------------------------------------------------
    # Combine and save
    # -------------------------------------------------------------------------
    print("\n--- Combining centroids ---")
    
    # Define unified schema columns to ensure consistent dtype handling
    schema_cols = ['geometry', 'source', 'runoff_mm', 'runoff_mm_baseline', 'dis_m3_pyr', 'delta', 'uncertainty']
    
    # 2030 - filter out empty dataframes before concat
    parts_2030 = [runoff_centroids_2030]
    if river_centroids_2030 is not None and len(river_centroids_2030) > 0:
        parts_2030.append(river_centroids_2030)
    parts_2030 = [p for p in parts_2030 if p is not None and len(p) > 0]
    
    if parts_2030:
        # Ensure all parts have identical columns to avoid FutureWarning
        all_cols = ['geometry', 'source', 'value_2030', 'value_baseline', 'delta', 'uncertainty']
        normalized_parts = []
        for p in parts_2030:
            df = p.copy()
            for col in all_cols:
                if col not in df.columns:
                    df[col] = None
            normalized_parts.append(df[all_cols])
        combined_2030 = gpd.GeoDataFrame(pd.concat(normalized_parts, ignore_index=True), crs="EPSG:4326")
    else:
        combined_2030 = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
    
    # 2050 - filter out empty dataframes before concat
    parts_2050 = [runoff_centroids_2050]
    if river_centroids_2050 is not None and len(river_centroids_2050) > 0:
        parts_2050.append(river_centroids_2050)
    parts_2050 = [p for p in parts_2050 if p is not None and len(p) > 0]
    
    if parts_2050:
        # Ensure all parts have identical columns to avoid FutureWarning
        all_cols = ['geometry', 'source', 'value_2050', 'value_baseline', 'delta', 'uncertainty']
        normalized_parts = []
        for p in parts_2050:
            df = p.copy()
            for col in all_cols:
                if col not in df.columns:
                    df[col] = None
            normalized_parts.append(df[all_cols])
        combined_2050 = gpd.GeoDataFrame(pd.concat(normalized_parts, ignore_index=True), crs="EPSG:4326")
    else:
        combined_2050 = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
    
    print(f"  Combined 2030: {len(combined_2030):,} centroids")
    print(f"  Combined 2050: {len(combined_2050):,} centroids")
    
    # Save
    print("\n--- Saving Part 4 outputs ---")
    combined_2030.to_parquet(out_dir / "HYDRO_VIABLE_CENTROIDS_2030.parquet")
    combined_2050.to_parquet(out_dir / "HYDRO_VIABLE_CENTROIDS_2050.parquet")
    print(f"  Saved: HYDRO_VIABLE_CENTROIDS_2030.parquet")
    print(f"  Saved: HYDRO_VIABLE_CENTROIDS_2050.parquet")
    
    return combined_2030, combined_2050


# Need pandas for concat
import pandas as pd


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run(workdir: str = None, era5_dir: str = None, riveratlas_path: str = None,
        download_only: bool = False, process_only: bool = False, mask_only: bool = False,
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
    
    river_mask_path = out_dir / f"river_proximity_mask_{int(HYDRO_RIVER_BUFFER_M/1000)}km.tif"
    
    print("="*70)
    print("UNIFIED HYDRO PROCESSING: RUNOFF + RIVERATLAS")
    print("="*70)
    
    # Download only
    if download_only:
        run_part1_download(cmip6_dir, era5_dir)
        return
    
    # Mask only (Part 2 partial)
    if mask_only:
        print("\n--- Generating river proximity mask only ---")
        if not riveratlas_path.exists():
            print(f"[ERROR] RiverATLAS not found: {riveratlas_path}")
            return
        gdf = load_riveratlas(riveratlas_path, RIVERATLAS_COLUMNS, min_discharge, min_order, bbox)
        create_river_proximity_mask(gdf, river_mask_path, HYDRO_RIVER_BUFFER_M, TARGET_RESOLUTION_ARCSEC)
        print(f"\n{'='*70}")
        print("COMPLETE! River proximity mask generated.")
        print(f"{'='*70}")
        return
    
    # Full or process-only pipeline
    
    # Part 1: Generate river proximity mask (needed for Part 2)
    mask_path, river_gdf = run_part1_river_mask(
        riveratlas_path, out_dir, min_discharge, min_order, bbox
    )
    
    if mask_path is None:
        mask_path = river_mask_path
    
    # Part 2: Runoff processing (produces delta grids needed for Part 3)
    baseline, r2030, r2050, u2030, u2050, lons, lats = run_part2_runoff(
        cmip6_dir, era5_dir, out_dir, mask_path
    )
    
    # Part 3: Apply delta to RiverATLAS discharge
    river_gdf = run_part3_riveratlas_projections(river_gdf, out_dir, save_gpkg)
    
    # Part 4: Generate viable centroids
    run_part4_centroids(baseline, r2030, r2050, u2030, u2050, lons, lats, river_gdf, out_dir)
    
    print(f"\n{'='*70}")
    print("ALL PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Hydro Processing: ERA5-Land/CMIP6 + RiverATLAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data
  python p1_f_viable_hydro.py --download-only
  
  # Process only (assumes downloads exist)
  python p1_f_viable_hydro.py --process-only
  
  # Generate river proximity mask only
  python p1_f_viable_hydro.py --mask-only
  
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
    parser.add_argument("--mask-only", action="store_true", help="Generate river mask only")
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
        mask_only=args.mask_only,
        min_discharge=args.min_discharge,
        min_order=args.min_order,
        bbox=tuple(args.bbox) if args.bbox else None,
        save_gpkg=args.save_gpkg,
    )
