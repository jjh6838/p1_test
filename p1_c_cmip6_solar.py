"""
p1_c_cmip6_solar.py
===================
CMIP6 Delta Method for Solar PV Output (PVOUT) Projections

Methodology:
1. Load Global Solar Atlas PVOUT baseline (kWh/kWp/day)
2. Load CMIP6 rsds (surface downwelling shortwave radiation) for historical (1995-2014) and SSP245 (2021-2060)
3. Compute climate change ratio: Δ = CMIP6_future / CMIP6_historical
4. Apply delta to PVOUT: PVOUT_future = PVOUT_baseline × Δ
5. Ensemble mean across 3 CMIP6 models
6. Regrid to 300 arcsec

Output:
- PVOUT_2030_300arcsec.tif
- PVOUT_2050_300arcsec.tif
- PVOUT_UNCERTAINTY_2030_300arcsec.tif (IQR across models)
- PVOUT_UNCERTAINTY_2050_300arcsec.tif
- PVOUT_baseline_300arcsec.tif
"""

import os
import zipfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.interpolate import RegularGridInterpolator

import cdsapi

# Suppress numpy warnings for NaN operations (expected for ocean pixels)
warnings.filterwarnings('ignore', 'Mean of empty slice')
warnings.filterwarnings('ignore', 'All-NaN slice encountered')


# =============================================================================
# PATH CONFIGURATION (Local vs Cluster)
# =============================================================================

def get_bigdata_path(folder_name: str) -> str:
    """
    Get the correct path for bigdata folders.
    Checks local path first, then cluster path if not found.
    
    Args:
        folder_name: Name of the bigdata folder (e.g., 'bigdata_gadm')
    
    Returns:
        str: Path to the folder
    """
    local_path = folder_name
    cluster_path = f"/soge-home/projects/mistral/ji/{folder_name}"
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(cluster_path):
        return cluster_path
    else:
        # Return local path as default (will trigger appropriate error if needed)
        return local_path


# =============================================================================
# USER SETTINGS
# =============================================================================

# CMIP6 models for ensemble (same as wind for consistency)
CMIP6_MODELS = [
    "cesm2",
    "ec_earth3_veg_lr",
    "mpi_esm1_2_lr",
]

# Periods
HIST_PERIOD = ("1995-01-01", "2014-12-31")  # Historical baseline (aligns with CMIP6)
P2030 = ("2021-01-01", "2040-12-31")        # 20-year mean centered on ~2030
P2050 = ("2041-01-01", "2060-12-31")        # 20-year mean centered on ~2050

# CDS settings
CDS_CMIP6_DATASET = "projections-cmip6"
SCENARIO = "ssp2_4_5"

# Output resolution settings - imported from shared config for consistency
from config import POP_AGGREGATION_FACTOR, TARGET_RESOLUTION_ARCSEC, GHS_POP_NATIVE_RESOLUTION_ARCSEC


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_cmip6_historical(model: str, out_dir: Path) -> Optional[Path]:
    """
    Download CMIP6 historical rsds (1995-2014).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_rsds_historical_1995-2014.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 historical already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 rsds historical: {model} (1995-2014)...")
    
    years = [str(y) for y in range(1995, 2015)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "surface_downwelling_shortwave_radiation",
        "model": model,
        "year": years,
        "month": months,
    }
    
    try:
        client.retrieve(CDS_CMIP6_DATASET, request, str(out_file))
        print(f"[done] {model} historical downloaded")
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to download {model} historical: {e}")
        return None


def download_cmip6_ssp245(model: str, out_dir: Path) -> Optional[Path]:
    """
    Download CMIP6 SSP245 rsds (2021-2060).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_rsds_ssp245_2021-2060.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 SSP245 already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 rsds SSP245: {model} (2021-2060)...")
    
    years = [str(y) for y in range(2021, 2061)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": SCENARIO,
        "variable": "surface_downwelling_shortwave_radiation",
        "model": model,
        "year": years,
        "month": months,
    }
    
    try:
        client.retrieve(CDS_CMIP6_DATASET, request, str(out_file))
        print(f"[done] {model} SSP245 downloaded")
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to download {model} SSP245: {e}")
        return None


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract ZIP file and return list of NetCDF files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    
    nc_files = sorted(out_dir.rglob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"No NetCDF found after extracting {zip_path}")
    
    return nc_files


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_pvout_baseline(tif_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Global Solar Atlas PVOUT baseline.
    Returns (data, lons, lats) arrays.
    """
    print(f"  Loading PVOUT baseline from {tif_path.name}...")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = data.shape
        nodata = src.nodata
        
        # Generate coordinate arrays
        cols = np.arange(width)
        rows = np.arange(height)
        
        # Transform pixel coordinates to geographic coordinates
        lons = transform.c + transform.a * (cols + 0.5)
        lats = transform.f + transform.e * (rows + 0.5)
    
    # Handle nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    
    print(f"  PVOUT shape: {data.shape}")
    print(f"  PVOUT mean (valid): {np.nanmean(data):.2f} kWh/kWp/day")
    
    return data, lons, lats


def load_cmip6_rsds(nc_paths: list[Path]) -> xr.DataArray:
    """
    Load CMIP6 rsds from NetCDF files.
    Returns solar radiation DataArray with dimensions (time, lat, lon).
    """
    ds = xr.open_mfdataset([str(p) for p in nc_paths], combine="by_coords")
    
    # Find the rsds variable (skip auxiliary vars like *_bnds)
    data_vars = [v for v in ds.data_vars if not v.endswith('_bnds')]
    if not data_vars:
        raise RuntimeError("No data variables found in CMIP6 NetCDF.")
    
    var_name = data_vars[0]
    print(f"  Using variable: {var_name}")
    
    da = ds[var_name]
    da = standardize_coords(da)
    
    return da


def standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """Standardize coordinate names and ensure lon is in [-180, 180]."""
    # Rename dimensions if needed
    dim_mapping = {}
    for dim in da.dims:
        dim_lower = str(dim).lower()
        if 'lat' in dim_lower and dim != 'lat':
            dim_mapping[dim] = 'lat'
        elif 'lon' in dim_lower and dim != 'lon':
            dim_mapping[dim] = 'lon'
        elif 'longitude' in dim_lower:
            dim_mapping[dim] = 'lon'
        elif 'latitude' in dim_lower:
            dim_mapping[dim] = 'lat'
        elif dim == 'valid_time' or 'time' in dim_lower:
            if dim != 'time':
                dim_mapping[dim] = 'time'
    
    if dim_mapping:
        da = da.rename(dim_mapping)
    
    # Ensure lon is in [-180, 180]
    if 'lon' in da.coords:
        lon_values = da.lon.values
        if lon_values.max() > 180:
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    
    return da


# =============================================================================
# DELTA METHOD COMPUTATION
# =============================================================================

def compute_temporal_mean(da: xr.DataArray, start_date: str, end_date: str) -> xr.DataArray:
    """Compute mean over specified time period."""
    subset = da.sel(time=slice(start_date, end_date))
    return subset.mean("time", skipna=True)


def compute_delta_ratio(cmip6_future: xr.DataArray, cmip6_hist: xr.DataArray) -> xr.DataArray:
    """
    Compute climate change ratio: Δ = future / historical.
    Values are clipped to [0.5, 2.0] to avoid unrealistic extremes.
    """
    # Avoid division by zero
    hist_safe = cmip6_hist.where(cmip6_hist > 1.0, 1.0)  # rsds is in W/m², use 1.0 as minimum
    delta = cmip6_future / hist_safe
    
    # Clip to reasonable range
    delta = delta.clip(min=0.5, max=2.0)
    
    return delta


def interpolate_delta_to_grid(delta: xr.DataArray, target_lons: np.ndarray, 
                               target_lats: np.ndarray) -> np.ndarray:
    """
    Interpolate CMIP6 delta to target grid (PVOUT grid).
    """
    # Get delta data
    delta_data = delta.values
    delta_lons = delta.lon.values
    delta_lats = delta.lat.values
    
    # Ensure lats are ascending for interpolation
    if delta_lats[0] > delta_lats[-1]:
        delta_lats = delta_lats[::-1]
        delta_data = delta_data[::-1, :]
    
    # Create interpolator
    interp_func = RegularGridInterpolator(
        (delta_lats, delta_lons),
        delta_data,
        method='linear',
        bounds_error=False,
        fill_value=1.0  # Default to no change for out-of-bounds
    )
    
    # Ensure target lats are ascending for meshgrid
    if target_lats[0] > target_lats[-1]:
        target_lats_sorted = target_lats[::-1]
        flip_output = True
    else:
        target_lats_sorted = target_lats
        flip_output = False
    
    # Create target grid
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats_sorted)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    # Interpolate
    delta_interp = interp_func(points).reshape(lat_grid.shape).astype("float32")
    
    if flip_output:
        delta_interp = delta_interp[::-1, :]
    
    return delta_interp


def apply_delta_to_pvout(pvout_baseline: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Apply CMIP6 delta to PVOUT baseline.
    PVOUT_future = PVOUT_baseline × Δ
    """
    pvout_future = pvout_baseline * delta
    return pvout_future


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

# Cache for GHS-POP grid parameters (read once from raster)
_GHS_POP_GRID_PARAMS = None


def get_ghs_pop_grid_params() -> dict:
    """
    Get GHS-POP raster grid parameters (origin and pixel size) dynamically.
    This ensures CMIP6 outputs align with settlement centroids regardless of GHS-POP version.
    
    Returns:
        dict: {origin_lon, origin_lat, pixel_size_lon, pixel_size_lat}
    """
    global _GHS_POP_GRID_PARAMS
    
    if _GHS_POP_GRID_PARAMS is not None:
        return _GHS_POP_GRID_PARAMS
    
    ghs_pop_path = Path(get_bigdata_path('bigdata_settlements_jrc')) / 'GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif'
    
    if ghs_pop_path.exists():
        with rasterio.open(ghs_pop_path) as src:
            t = src.transform
            _GHS_POP_GRID_PARAMS = {
                'origin_lon': t.c,
                'origin_lat': t.f,
                'pixel_size_lon': t.a,  # positive
                'pixel_size_lat': t.e,  # negative
            }
            print(f"  [grid] GHS-POP params from raster:")
            print(f"         origin=({t.c:.6f}, {t.f:.6f})")
            print(f"         pixel_size=({t.a:.10f}, {t.e:.10f})")
    else:
        # Fallback to known values if raster not available
        _GHS_POP_GRID_PARAMS = {
            'origin_lon': -180.00791593130032,
            'origin_lat': 89.0995831776456,
            'pixel_size_lon': 0.008333333300326923,
            'pixel_size_lat': -0.00833333329979504,
        }
        print(f"  [grid] GHS-POP raster not found, using fallback params")
    
    return _GHS_POP_GRID_PARAMS


def regrid_to_target(data: np.ndarray, src_lons: np.ndarray, src_lats: np.ndarray,
                     target_res_arcsec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrid data array to target resolution.
    Grid is aligned with GHS-POP raster for consistency with settlement centroids.
    Returns (data, lons, lats) at target resolution.
    
    The output grid exactly matches what process_country_supply.py produces via
    rasterio.transform.xy() after aggregating GHS-POP by POP_AGGREGATION_FACTOR.
    """
    # Get GHS-POP grid parameters (origin and pixel size)
    ghs_params = get_ghs_pop_grid_params()
    
    # Calculate aggregation factor from target resolution
    ghs_native_arcsec = abs(ghs_params['pixel_size_lon']) * 3600
    agg_factor = round(target_res_arcsec / ghs_native_arcsec)
    
    # Aggregated pixel size (match GHS-POP exactly, don't use target_res_arcsec/3600)
    agg_pixel_lon = ghs_params['pixel_size_lon'] * agg_factor
    agg_pixel_lat = ghs_params['pixel_size_lat'] * agg_factor  # negative
    
    # Ensure source lats are ascending for interpolation
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[::-1, :]
    
    # Generate target grid matching rasterio.transform.xy() output
    # For aggregated GHS-POP: xy(row, col) gives pixel center at:
    #   x = origin_lon + agg_pixel_lon * (col + 0.5)
    #   y = origin_lat + agg_pixel_lat * (row + 0.5)  # agg_pixel_lat is negative
    
    # Number of pixels in aggregated grid
    n_cols = int(np.ceil(360 / abs(agg_pixel_lon)))
    n_rows = int(np.ceil(180 / abs(agg_pixel_lat)))
    
    # Generate coordinates exactly as rasterio.transform.xy would
    target_lons = ghs_params['origin_lon'] + agg_pixel_lon * (np.arange(n_cols) + 0.5)
    target_lats = ghs_params['origin_lat'] + agg_pixel_lat * (np.arange(n_rows) + 0.5)  # descending
    
    # For interpolation, need ascending lats
    target_lats_asc = target_lats[::-1]
    
    # Create interpolator
    interp_func = RegularGridInterpolator(
        (src_lats, src_lons),
        data,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Interpolate using ascending lats for correct results
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats_asc)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    target_data = interp_func(points).reshape(lat_grid.shape).astype("float32")
    
    # Flip data back to match descending lats (row 0 = north)
    target_data = target_data[::-1, :]
    
    # Return descending lats (standard GeoTIFF order: row 0 = north)
    return target_data, target_lons, target_lats


def save_geotiff(data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                 out_path: Path, nodata: float = 0) -> None:
    """Save data array as GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    height, width = data.shape
    
    # Flip data if lats are ascending (GeoTIFF expects top-left origin)
    if lats[0] < lats[-1]:
        data = data[::-1, :]
        lats = lats[::-1]
    
    transform = from_bounds(
        lons.min(), lats.min(), lons.max(), lats.max(),
        width, height
    )
    
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    
    # Replace NaN with nodata
    data = np.nan_to_num(data, nan=nodata)
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    
    print(f"  Saved: {out_path}")


def save_as_parquet(data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                    out_path: Path, value_column: str = "value", nodata: float = 0) -> None:
    """Save data array as Parquet centroids (points with values)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure lats are in correct order for grid creation
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]
    
    # Create meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten arrays
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    data_flat = data.ravel()
    
    # Filter out nodata/NaN values
    valid_mask = ~np.isnan(data_flat) & (data_flat != nodata) & (data_flat > 0)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        value_column: data_flat[valid_mask],
        'geometry': gpd.points_from_xy(lon_flat[valid_mask], lat_flat[valid_mask])
    }, crs="EPSG:4326")
    
    # Save as parquet
    gdf.to_parquet(out_path)
    print(f"  Saved: {out_path} ({len(gdf):,} points)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_download_only(workdir: Path):
    """Download all required CMIP6 datasets."""
    cmip6_dir = workdir / "downloads"
    
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING CMIP6 SOLAR RADIATION DATA")
    print("="*70)
    
    # Download CMIP6 rsds for each model
    for model in CMIP6_MODELS:
        print(f"\n--- {model} ---")
        download_cmip6_historical(model, cmip6_dir)
        download_cmip6_ssp245(model, cmip6_dir)
    
    print("\n[done] All downloads complete!")


def run_processing(workdir: Path, pvout_path: Path):
    """Process downloaded data using delta method."""
    cmip6_dl_dir = workdir / "downloads"
    cmip6_ex_dir = workdir / "extracted"
    out_dir = workdir / "outputs"
    
    print("\n" + "="*70)
    print("STEP 2: PROCESSING WITH DELTA METHOD")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Load PVOUT baseline and regrid to target resolution first (memory efficient)
    # -------------------------------------------------------------------------
    print("\n--- Loading PVOUT baseline ---")
    if not pvout_path.exists():
        raise FileNotFoundError(f"PVOUT baseline not found: {pvout_path}")
    
    pvout_baseline_full, pvout_lons_full, pvout_lats_full = load_pvout_baseline(pvout_path)
    
    # Regrid to 300 arcsec FIRST to reduce memory usage
    print(f"\n--- Regridding PVOUT baseline to {TARGET_RESOLUTION_ARCSEC} arcsec (memory optimization) ---")
    pvout_baseline, pvout_lons, pvout_lats = regrid_to_target(
        pvout_baseline_full, pvout_lons_full, pvout_lats_full, TARGET_RESOLUTION_ARCSEC
    )
    print(f"  Regridded PVOUT shape: {pvout_baseline.shape}")
    print(f"  Regridded PVOUT mean: {np.nanmean(pvout_baseline):.2f} kWh/kWp/day")
    
    # Free memory
    del pvout_baseline_full, pvout_lons_full, pvout_lats_full
    
    # -------------------------------------------------------------------------
    # Process each CMIP6 model
    # -------------------------------------------------------------------------
    delta_2030_list = []
    delta_2050_list = []
    pvout_2030_list = []
    pvout_2050_list = []
    successful_models = []
    
    for model in CMIP6_MODELS:
        print(f"\n--- Processing CMIP6 model: {model} ---")
        
        try:
            # Extract historical
            hist_zip = cmip6_dl_dir / f"{model}_rsds_historical_1995-2014.zip"
            if not hist_zip.exists():
                print(f"  [skip] Historical ZIP not found: {hist_zip}")
                continue
            
            hist_nc = extract_zip(hist_zip, cmip6_ex_dir / "historical" / model)
            cmip6_hist_rsds = load_cmip6_rsds(hist_nc)
            cmip6_hist_mean = compute_temporal_mean(cmip6_hist_rsds, HIST_PERIOD[0], HIST_PERIOD[1])
            print(f"  Historical mean rsds: {float(cmip6_hist_mean.mean()):.2f} W/m²")
            
            # Extract SSP245
            ssp_zip = cmip6_dl_dir / f"{model}_rsds_ssp245_2021-2060.zip"
            if not ssp_zip.exists():
                print(f"  [skip] SSP245 ZIP not found: {ssp_zip}")
                continue
            
            ssp_nc = extract_zip(ssp_zip, cmip6_ex_dir / "ssp245" / model)
            cmip6_ssp_rsds = load_cmip6_rsds(ssp_nc)
            
            # Compute period means for 2030 and 2050
            cmip6_2030_mean = compute_temporal_mean(cmip6_ssp_rsds, P2030[0], P2030[1])
            cmip6_2050_mean = compute_temporal_mean(cmip6_ssp_rsds, P2050[0], P2050[1])
            print(f"  2030 mean rsds: {float(cmip6_2030_mean.mean()):.2f} W/m²")
            print(f"  2050 mean rsds: {float(cmip6_2050_mean.mean()):.2f} W/m²")
            
            # Compute delta ratios
            delta_2030 = compute_delta_ratio(cmip6_2030_mean, cmip6_hist_mean)
            delta_2050 = compute_delta_ratio(cmip6_2050_mean, cmip6_hist_mean)
            print(f"  Delta 2030 mean: {float(delta_2030.mean()):.4f}")
            print(f"  Delta 2050 mean: {float(delta_2050.mean()):.4f}")
            
            # Interpolate delta to PVOUT grid
            delta_2030_interp = interpolate_delta_to_grid(delta_2030, pvout_lons, pvout_lats)
            delta_2050_interp = interpolate_delta_to_grid(delta_2050, pvout_lons, pvout_lats)
            
            # Apply delta to PVOUT
            pvout_2030 = apply_delta_to_pvout(pvout_baseline, delta_2030_interp)
            pvout_2050 = apply_delta_to_pvout(pvout_baseline, delta_2050_interp)
            
            # Store for ensemble
            delta_2030_list.append(delta_2030_interp)
            delta_2050_list.append(delta_2050_interp)
            pvout_2030_list.append(pvout_2030)
            pvout_2050_list.append(pvout_2050)
            successful_models.append(model)
            
            print(f"  [done] {model}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {model}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not pvout_2030_list:
        raise RuntimeError("No CMIP6 models processed successfully!")
    
    print(f"\n--- Successfully processed {len(successful_models)} models: {successful_models} ---")
    
    # -------------------------------------------------------------------------
    # Compute ensemble statistics
    # -------------------------------------------------------------------------
    print("\n--- Computing ensemble statistics ---")
    
    # Stack arrays for ensemble calculations
    pvout_2030_stack = np.stack(pvout_2030_list, axis=0)
    pvout_2050_stack = np.stack(pvout_2050_list, axis=0)
    
    # Ensemble mean PVOUT
    print("  Computing ensemble mean...")
    pvout_2030_ensemble = np.nanmean(pvout_2030_stack, axis=0)
    pvout_2050_ensemble = np.nanmean(pvout_2050_stack, axis=0)
    
    # Ensemble uncertainty: use range (max - min) instead of IQR
    # This is faster and nearly equivalent for 3 models
    print("  Computing ensemble uncertainty (range)...")
    pvout_2030_iqr = np.nanmax(pvout_2030_stack, axis=0) - np.nanmin(pvout_2030_stack, axis=0)
    pvout_2050_iqr = np.nanmax(pvout_2050_stack, axis=0) - np.nanmin(pvout_2050_stack, axis=0)
    
    print(f"  Ensemble PVOUT 2030 mean: {np.nanmean(pvout_2030_ensemble):.2f} kWh/kWp/day")
    print(f"  Ensemble PVOUT 2050 mean: {np.nanmean(pvout_2050_ensemble):.2f} kWh/kWp/day")
    print(f"  Baseline PVOUT mean: {np.nanmean(pvout_baseline):.2f} kWh/kWp/day")
    
    # Output grid is already at target resolution (pvout_lons, pvout_lats)
    lons = pvout_lons
    lats = pvout_lats
    
    print(f"\n--- Output shape: {pvout_2030_ensemble.shape} ---")
    
    # -------------------------------------------------------------------------
    # Save outputs (GeoTIFF and Parquet)
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs (GeoTIFF) ---")
    
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    
    save_geotiff(pvout_2030_ensemble, lons, lats, out_dir / f"PVOUT_2030_{suffix}.tif")
    save_geotiff(pvout_2050_ensemble, lons, lats, out_dir / f"PVOUT_2050_{suffix}.tif")
    save_geotiff(pvout_2030_iqr, lons, lats, out_dir / f"PVOUT_UNCERTAINTY_2030_{suffix}.tif")
    save_geotiff(pvout_2050_iqr, lons, lats, out_dir / f"PVOUT_UNCERTAINTY_2050_{suffix}.tif")
    save_geotiff(pvout_baseline, lons, lats, out_dir / f"PVOUT_baseline_{suffix}.tif")
    
    print("\n--- Saving outputs (Parquet centroids) ---")
    
    save_as_parquet(pvout_2030_ensemble, lons, lats, out_dir / f"PVOUT_2030_{suffix}.parquet", "PVOUT_2030")
    save_as_parquet(pvout_2050_ensemble, lons, lats, out_dir / f"PVOUT_2050_{suffix}.parquet", "PVOUT_2050")
    save_as_parquet(pvout_2030_iqr, lons, lats, out_dir / f"PVOUT_UNCERTAINTY_2030_{suffix}.parquet", "PVOUT_UNC_2030")
    save_as_parquet(pvout_2050_iqr, lons, lats, out_dir / f"PVOUT_UNCERTAINTY_2050_{suffix}.parquet", "PVOUT_UNC_2050")
    save_as_parquet(pvout_baseline, lons, lats, out_dir / f"PVOUT_baseline_{suffix}.parquet", "PVOUT_baseline")
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print(f"  GeoTIFF rasters:")
    print(f"    - PVOUT_2030_{suffix}.tif")
    print(f"    - PVOUT_2050_{suffix}.tif")
    print(f"    - PVOUT_UNCERTAINTY_2030_{suffix}.tif")
    print(f"    - PVOUT_UNCERTAINTY_2050_{suffix}.tif")
    print(f"    - PVOUT_baseline_{suffix}.tif")
    print(f"  Parquet centroids:")
    print(f"    - PVOUT_2030_{suffix}.parquet")
    print(f"    - PVOUT_2050_{suffix}.parquet")
    print(f"    - PVOUT_UNCERTAINTY_2030_{suffix}.parquet")
    print(f"    - PVOUT_UNCERTAINTY_2050_{suffix}.parquet")
    print(f"    - PVOUT_baseline_{suffix}.parquet")


def run(workdir: str = None,
        pvout_baseline: str = None,
        download_only: bool = False,
        process_only: bool = False):
    """
    Main entry point.
    
    Args:
        workdir: Working directory for CMIP6 data
        pvout_baseline: Path to Global Solar Atlas PVOUT.tif
        download_only: Only download data, don't process
        process_only: Only process data, assume downloads exist
    """
    # Use get_bigdata_path for default directories
    if workdir is None:
        workdir = Path(get_bigdata_path("bigdata_solar_cmip6"))
    else:
        workdir = Path(workdir)
    
    # Default PVOUT baseline path
    if pvout_baseline is None:
        pvout_path = Path(get_bigdata_path("bigdata_solar_pvout")) / "PVOUT.tif"
    else:
        pvout_path = Path(pvout_baseline)
    
    if download_only:
        run_download_only(workdir)
    elif process_only:
        run_processing(workdir, pvout_path)
    else:
        # Full pipeline
        run_download_only(workdir)
        run_processing(workdir, pvout_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CMIP6 Delta Method for Solar PV Output Projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all CMIP6 data
  python p1_c_cmip6_solar.py --download-only
  
  # Process with custom PVOUT baseline
  python p1_c_cmip6_solar.py --process-only --pvout-baseline ./bigdata_solar_pvout/PVOUT.tif
  
  # Full pipeline
  python p1_c_cmip6_solar.py
        """
    )
    
    parser.add_argument(
        "--workdir", 
        default=None,
        help="Working directory for CMIP6 downloads and outputs (auto-detects local/cluster)"
    )
    parser.add_argument(
        "--pvout-baseline",
        default=None,
        help="Path to Global Solar Atlas PVOUT.tif baseline (auto-detects local/cluster)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data, don't process"
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process data (assumes downloads exist)"
    )
    
    args = parser.parse_args()
    
    run(
        workdir=args.workdir,
        pvout_baseline=args.pvout_baseline,
        download_only=args.download_only,
        process_only=args.process_only
    )
