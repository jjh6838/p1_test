"""
p1_e_cmip6_hydro.py
===================
ERA5-Land + CMIP6 Delta Method for Runoff Projections

Methodology:
1. Load ERA5-Land runoff baseline (1995-2014 monthly means)
2. Load CMIP6 total_runoff for historical (1995-2014) and SSP245 (2021-2060)
3. Compute climate change ratio: Δ = CMIP6_future / CMIP6_historical
4. Apply delta to ERA5-Land: Runoff_future = Runoff_ERA5 × Δ
5. Ensemble mean across 3 CMIP6 models
6. Regrid to 300 arcsec (aligned with GHS-POP grid, matching process_country_supply.py)

Output:
- HYDRO_RUNOFF_2030_300arcsec.tif (mm/year)
- HYDRO_RUNOFF_2050_300arcsec.tif (mm/year)
- HYDRO_RUNOFF_UNCERTAINTY_2030_300arcsec.tif (range = max - min across models, same as solar/wind)
- HYDRO_RUNOFF_UNCERTAINTY_2050_300arcsec.tif
- HYDRO_RUNOFF_ERA5_baseline_300arcsec.tif
- HYDRO_ATLAS_DELTA_2030_300arcsec.tif (for use by p1_f_hydroatlas.py)
- HYDRO_ATLAS_DELTA_2050_300arcsec.tif

Units:
- ERA5-Land runoff: meters (accumulated monthly) → convert to mm/year
- CMIP6 total_runoff: kg m⁻² s⁻¹ → convert to mm/year
- Output: mm/year (millimeters per year)
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
from scipy.interpolate import RegularGridInterpolator

import cdsapi

# Suppress numpy/dask warnings for NaN operations (expected for ocean pixels)
warnings.filterwarnings('ignore', 'Mean of empty slice')
warnings.filterwarnings('ignore', 'All-NaN slice encountered')


# =============================================================================
# PATH CONFIGURATION (Local vs Cluster)
# =============================================================================

def get_bigdata_path(folder_name: str) -> str:
    """
    Get the correct path for bigdata folders.
    Checks local path first, then cluster path if not found.
    """
    local_path = folder_name
    cluster_path = f"/soge-home/projects/mistral/ji/{folder_name}"
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(cluster_path):
        return cluster_path
    else:
        return local_path


# =============================================================================
# USER SETTINGS
# =============================================================================

# CMIP6 models for ensemble (same as solar/wind)
CMIP6_MODELS = [
    "cesm2",
    "ec_earth3_veg_lr",
    "mpi_esm1_2_lr",
]

# Periods
HIST_PERIOD = ("1995-01-01", "2014-12-31")  # Historical baseline
P2030 = ("2021-01-01", "2040-12-31")        # 20-year mean centered on ~2030
P2050 = ("2041-01-01", "2060-12-31")        # 20-year mean centered on ~2050

# CDS settings
CDS_CMIP6_DATASET = "projections-cmip6"
CDS_ERA5_LAND_DATASET = "reanalysis-era5-land-monthly-means"
SCENARIO = "ssp2_4_5"

# Unit conversion constants
# ERA5-Land runoff is in meters (monthly accumulated)
# CMIP6 runoff is in kg/m²/s (same as mm/s since 1 kg/m² = 1 mm)
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MM_PER_METER = 1000

# Output resolution settings
from config import POP_AGGREGATION_FACTOR, TARGET_RESOLUTION_ARCSEC, GHS_POP_NATIVE_RESOLUTION_ARCSEC


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_era5_land(out_dir: Path) -> Path:
    """
    Download ERA5-Land monthly runoff (1995-2014).
    Returns path to downloaded NetCDF file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "era5_land_runoff_1995-2014.nc"
    
    if out_file.exists():
        print(f"[skip] ERA5-Land already downloaded: {out_file}")
        return out_file
    
    print("[download] ERA5-Land runoff (1995-2014)...")
    
    years = [str(y) for y in range(1995, 2015)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["runoff"],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client.retrieve(CDS_ERA5_LAND_DATASET, request, str(out_file))
    print(f"[done] ERA5-Land downloaded: {out_file}")
    return out_file


def download_cmip6_historical(model: str, out_dir: Path) -> Optional[Path]:
    """
    Download CMIP6 historical total_runoff (1995-2014).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_historical_runoff_1995-2014.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 historical already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 historical runoff: {model} (1995-2014)...")
    
    years = [str(y) for y in range(1995, 2015)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "total_runoff",
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
    Download CMIP6 SSP245 total_runoff (2021-2060).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_ssp245_runoff_2021-2060.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 SSP245 already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 SSP245 runoff: {model} (2021-2060)...")
    
    years = [str(y) for y in range(2021, 2061)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": SCENARIO,
        "variable": "total_runoff",
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

def load_era5_land_runoff(nc_path: Path) -> xr.DataArray:
    """
    Load ERA5-Land runoff data.
    ERA5-Land runoff is monthly accumulated in meters.
    Returns DataArray with dimensions (time, lat, lon).
    """
    print(f"  Loading ERA5-Land from {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    
    # Find runoff variable (could be 'ro', 'runoff', etc.)
    runoff_var = None
    for var in ds.data_vars:
        var_lower = var.lower()
        if 'runoff' in var_lower or var == 'ro':
            runoff_var = var
            break
    
    if runoff_var is None:
        # Try common ERA5-Land names
        if 'ro' in ds.data_vars:
            runoff_var = 'ro'
        else:
            raise RuntimeError(f"Cannot find runoff variable. Variables: {list(ds.data_vars)}")
    
    print(f"  Using variable: {runoff_var}")
    
    da = ds[runoff_var]
    da = standardize_coords(da)
    
    return da


def load_cmip6_runoff(nc_paths: list[Path]) -> xr.DataArray:
    """
    Load CMIP6 total_runoff from NetCDF files.
    CMIP6 runoff is in kg m⁻² s⁻¹ (equivalent to mm/s).
    Returns DataArray with dimensions (time, lat, lon).
    """
    ds = xr.open_mfdataset([str(p) for p in nc_paths], combine="by_coords")
    
    # Find the runoff variable (mrro = total runoff)
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
# UNIT CONVERSION FUNCTIONS
# =============================================================================

def era5_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """
    Convert ERA5-Land runoff from meters (monthly accumulated) to mm/year.
    
    ERA5-Land runoff is monthly accumulated, so we sum all months and convert.
    """
    # Sum annual runoff (assumes 12 months per year in the data)
    # Then convert meters to mm
    annual_mean = da.groupby('time.year').sum('time').mean('year')
    annual_mm = annual_mean * MM_PER_METER  # m → mm
    
    return annual_mm


def cmip6_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """
    Convert CMIP6 runoff from kg m⁻² s⁻¹ to mm/year.
    
    1 kg/m²/s = 1 mm/s (by definition, since 1 kg water = 1 liter = 1 mm over 1 m²)
    mm/year = mm/s × seconds_per_year
    """
    # Mean annual value (already in kg/m²/s = mm/s)
    # Convert to mm/year
    annual_mm = da * SECONDS_PER_YEAR
    
    return annual_mm


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
    Values are clipped to [0.2, 3.0] to avoid unrealistic extremes.
    (Wider range than wind since runoff changes can be more dramatic)
    """
    # Avoid division by zero
    hist_safe = cmip6_hist.where(cmip6_hist > 1e-10, 1e-10)
    delta = cmip6_future / hist_safe
    
    # Clip to reasonable range
    delta = delta.clip(min=0.2, max=3.0)
    
    return delta


def apply_delta_to_era5(era5_baseline: xr.DataArray, delta: xr.DataArray) -> xr.DataArray:
    """
    Apply CMIP6 delta to ERA5 baseline.
    Interpolates delta to ERA5 grid first.
    """
    delta_interp = delta.interp(
        lat=era5_baseline.lat,
        lon=era5_baseline.lon,
        method="linear"
    )
    
    future_runoff = era5_baseline * delta_interp
    
    return future_runoff


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

_GHS_POP_GRID_PARAMS = None


def get_ghs_pop_grid_params() -> dict:
    """
    Get GHS-POP raster grid parameters for alignment with settlement centroids.
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
                'pixel_size_lon': t.a,
                'pixel_size_lat': t.e,
            }
            print(f"  [grid] GHS-POP params from raster")
    else:
        _GHS_POP_GRID_PARAMS = {
            'origin_lon': -180.00791593130032,
            'origin_lat': 89.0995831776456,
            'pixel_size_lon': 0.008333333300326923,
            'pixel_size_lat': -0.00833333329979504,
        }
        print(f"  [grid] Using fallback GHS-POP params")
    
    return _GHS_POP_GRID_PARAMS


def regrid_to_target(da: xr.DataArray, target_res_arcsec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrid DataArray to target resolution aligned with GHS-POP grid.
    Returns (data, lons, lats) at target resolution.
    """
    src_data = da.values
    src_lons = da.lon.values
    src_lats = da.lat.values
    
    # Ensure source lats are ascending for interpolation
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]
    
    ghs_params = get_ghs_pop_grid_params()
    
    ghs_native_arcsec = abs(ghs_params['pixel_size_lon']) * 3600
    agg_factor = round(target_res_arcsec / ghs_native_arcsec)
    
    agg_pixel_lon = ghs_params['pixel_size_lon'] * agg_factor
    agg_pixel_lat = ghs_params['pixel_size_lat'] * agg_factor
    
    n_cols = int(np.ceil(360 / abs(agg_pixel_lon)))
    n_rows = int(np.ceil(180 / abs(agg_pixel_lat)))
    
    target_lons = ghs_params['origin_lon'] + agg_pixel_lon * (np.arange(n_cols) + 0.5)
    target_lats = ghs_params['origin_lat'] + agg_pixel_lat * (np.arange(n_rows) + 0.5)
    
    target_lats_asc = target_lats[::-1]
    
    interp_func = RegularGridInterpolator(
        (src_lats, src_lons),
        src_data,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats_asc)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    target_data = interp_func(points).reshape(lat_grid.shape).astype("float32")
    target_data = target_data[::-1, :]
    
    return target_data, target_lons, target_lats


def save_geotiff(data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                 out_path: Path, nodata: float = -9999) -> None:
    """Save data array as GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    height, width = data.shape
    
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
    
    data = np.nan_to_num(data, nan=nodata)
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    
    print(f"  Saved: {out_path}")


def save_as_parquet(data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                    out_path: Path, value_column: str = "value", nodata: float = -9999) -> None:
    """Save data array as Parquet centroids."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    data_flat = data.ravel()
    
    valid_mask = ~np.isnan(data_flat) & (data_flat != nodata) & (data_flat > 0)
    
    gdf = gpd.GeoDataFrame({
        value_column: data_flat[valid_mask],
        'geometry': gpd.points_from_xy(lon_flat[valid_mask], lat_flat[valid_mask])
    }, crs="EPSG:4326")
    
    gdf.to_parquet(out_path)
    print(f"  Saved: {out_path} ({len(gdf):,} points)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_download_only(workdir: Path, era5_dir: Path):
    """Download all required datasets."""
    cmip6_dir = workdir / "downloads"
    
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATA")
    print("="*70)
    
    # Download ERA5-Land
    download_era5_land(era5_dir)
    
    # Download CMIP6 for each model
    for model in CMIP6_MODELS:
        print(f"\n--- {model} ---")
        download_cmip6_historical(model, cmip6_dir)
        download_cmip6_ssp245(model, cmip6_dir)
    
    print("\n[done] All downloads complete!")


def run_processing(workdir: Path, era5_dir: Path):
    """Process downloaded data using delta method."""
    cmip6_dl_dir = workdir / "downloads"
    cmip6_ex_dir = workdir / "extracted"
    out_dir = workdir / "outputs"
    
    print("\n" + "="*70)
    print("STEP 2: PROCESSING WITH DELTA METHOD")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Load ERA5-Land baseline
    # -------------------------------------------------------------------------
    print("\n--- Loading ERA5-Land baseline ---")
    era5_file = era5_dir / "era5_land_runoff_1995-2014.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5-Land file not found: {era5_file}. Run with --download first.")
    
    era5_runoff = load_era5_land_runoff(era5_file)
    
    # Convert to mm/year and compute baseline mean
    era5_annual = era5_to_mm_per_year(era5_runoff)
    print(f"  ERA5-Land baseline shape: {era5_annual.shape}")
    print(f"  ERA5-Land mean runoff: {float(era5_annual.mean()):.1f} mm/year")
    
    # -------------------------------------------------------------------------
    # Process each CMIP6 model
    # -------------------------------------------------------------------------
    delta_2030_list = []
    delta_2050_list = []
    runoff_2030_list = []
    runoff_2050_list = []
    successful_models = []
    
    for model in CMIP6_MODELS:
        print(f"\n--- Processing CMIP6 model: {model} ---")
        
        try:
            # Extract historical
            hist_zip = cmip6_dl_dir / f"{model}_historical_runoff_1995-2014.zip"
            if not hist_zip.exists():
                print(f"  [skip] Historical ZIP not found: {hist_zip}")
                continue
            
            hist_nc = extract_zip(hist_zip, cmip6_ex_dir / "historical" / model)
            cmip6_hist_runoff = load_cmip6_runoff(hist_nc)
            cmip6_hist_mean = compute_temporal_mean(cmip6_hist_runoff, HIST_PERIOD[0], HIST_PERIOD[1])
            print(f"  Historical mean: {float(cmip6_hist_mean.mean() * SECONDS_PER_YEAR):.1f} mm/year")
            
            # Extract SSP245
            ssp_zip = cmip6_dl_dir / f"{model}_ssp245_runoff_2021-2060.zip"
            if not ssp_zip.exists():
                print(f"  [skip] SSP245 ZIP not found: {ssp_zip}")
                continue
            
            ssp_nc = extract_zip(ssp_zip, cmip6_ex_dir / "ssp245" / model)
            cmip6_ssp_runoff = load_cmip6_runoff(ssp_nc)
            
            # Compute period means
            cmip6_2030_mean = compute_temporal_mean(cmip6_ssp_runoff, P2030[0], P2030[1])
            cmip6_2050_mean = compute_temporal_mean(cmip6_ssp_runoff, P2050[0], P2050[1])
            print(f"  2030 mean: {float(cmip6_2030_mean.mean() * SECONDS_PER_YEAR):.1f} mm/year")
            print(f"  2050 mean: {float(cmip6_2050_mean.mean() * SECONDS_PER_YEAR):.1f} mm/year")
            
            # Compute delta ratios
            delta_2030 = compute_delta_ratio(cmip6_2030_mean, cmip6_hist_mean)
            delta_2050 = compute_delta_ratio(cmip6_2050_mean, cmip6_hist_mean)
            print(f"  Delta 2030 mean: {float(delta_2030.mean()):.3f}")
            print(f"  Delta 2050 mean: {float(delta_2050.mean()):.3f}")
            
            # Apply delta to ERA5-Land
            runoff_2030 = apply_delta_to_era5(era5_annual, delta_2030)
            runoff_2050 = apply_delta_to_era5(era5_annual, delta_2050)
            
            # Interpolate deltas to ERA5 grid before storing (ensures consistent grids for ensemble)
            delta_2030_interp = delta_2030.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            delta_2050_interp = delta_2050.interp(lat=era5_annual.lat, lon=era5_annual.lon, method="linear")
            
            # Store for ensemble
            delta_2030_list.append(delta_2030_interp)
            delta_2050_list.append(delta_2050_interp)
            runoff_2030_list.append(runoff_2030)
            runoff_2050_list.append(runoff_2050)
            successful_models.append(model)
            
            print(f"  [done] {model}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {model}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not runoff_2030_list:
        raise RuntimeError("No CMIP6 models processed successfully!")
    
    print(f"\n--- Successfully processed {len(successful_models)} models: {successful_models} ---")
    
    # -------------------------------------------------------------------------
    # Compute ensemble statistics
    # -------------------------------------------------------------------------
    print("\n--- Computing ensemble statistics ---")
    
    print("  Computing ensemble mean...")
    runoff_2030_ensemble = xr.concat(runoff_2030_list, dim="model").mean("model", skipna=True)
    runoff_2050_ensemble = xr.concat(runoff_2050_list, dim="model").mean("model", skipna=True)
    
    print("  Computing ensemble uncertainty (range)...")
    runoff_2030_stack = xr.concat(runoff_2030_list, dim="model")
    runoff_2050_stack = xr.concat(runoff_2050_list, dim="model")
    
    runoff_2030_iqr = runoff_2030_stack.max(dim="model") - runoff_2030_stack.min(dim="model")
    runoff_2050_iqr = runoff_2050_stack.max(dim="model") - runoff_2050_stack.min(dim="model")
    
    print(f"  Ensemble runoff 2030 mean: {float(runoff_2030_ensemble.mean()):.1f} mm/year")
    print(f"  Ensemble runoff 2050 mean: {float(runoff_2050_ensemble.mean()):.1f} mm/year")
    
    # -------------------------------------------------------------------------
    # Regrid to target resolution
    # -------------------------------------------------------------------------
    print(f"\n--- Regridding to {TARGET_RESOLUTION_ARCSEC} arcsec ---")
    
    runoff_2030_data, lons, lats = regrid_to_target(runoff_2030_ensemble, TARGET_RESOLUTION_ARCSEC)
    runoff_2050_data, _, _ = regrid_to_target(runoff_2050_ensemble, TARGET_RESOLUTION_ARCSEC)
    iqr_2030_data, _, _ = regrid_to_target(runoff_2030_iqr, TARGET_RESOLUTION_ARCSEC)
    iqr_2050_data, _, _ = regrid_to_target(runoff_2050_iqr, TARGET_RESOLUTION_ARCSEC)
    
    # Also regrid baseline
    era5_data, _, _ = regrid_to_target(era5_annual, TARGET_RESOLUTION_ARCSEC)
    
    print(f"  Output shape: {runoff_2030_data.shape}")
    
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs (GeoTIFF) ---")
    
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    
    save_geotiff(runoff_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2030_{suffix}.tif")
    save_geotiff(runoff_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2050_{suffix}.tif")
    save_geotiff(iqr_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.tif")
    save_geotiff(iqr_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.tif")
    save_geotiff(era5_data, lons, lats, out_dir / f"HYDRO_RUNOFF_ERA5_baseline_{suffix}.tif")
    
    print("\n--- Saving outputs (Parquet centroids) ---")
    
    save_as_parquet(runoff_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2030_{suffix}.parquet", "RUNOFF_2030")
    save_as_parquet(runoff_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_2050_{suffix}.parquet", "RUNOFF_2050")
    save_as_parquet(iqr_2030_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.parquet", "RUNOFF_UNC_2030")
    save_as_parquet(iqr_2050_data, lons, lats, out_dir / f"HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.parquet", "RUNOFF_UNC_2050")
    save_as_parquet(era5_data, lons, lats, out_dir / f"HYDRO_RUNOFF_ERA5_baseline_{suffix}.parquet", "RUNOFF_baseline")
    
    # -------------------------------------------------------------------------
    # Save delta grids for use by p1_f_hydroatlas.py
    # -------------------------------------------------------------------------
    print("\n--- Saving delta grids for HydroATLAS script ---")
    
    # Deltas are already on ERA5 grid (-180 to 180 longitude) from interpolation above
    delta_2030_ensemble = xr.concat(delta_2030_list, dim="model").mean("model", skipna=True)
    delta_2050_ensemble = xr.concat(delta_2050_list, dim="model").mean("model", skipna=True)
    
    print(f"  Delta 2030 ensemble mean: {float(delta_2030_ensemble.mean()):.3f}")
    print(f"  Delta 2050 ensemble mean: {float(delta_2050_ensemble.mean()):.3f}")
    
    delta_2030_data, _, _ = regrid_to_target(delta_2030_ensemble, TARGET_RESOLUTION_ARCSEC)
    delta_2050_data, _, _ = regrid_to_target(delta_2050_ensemble, TARGET_RESOLUTION_ARCSEC)
    
    save_geotiff(delta_2030_data, lons, lats, out_dir / f"HYDRO_ATLAS_DELTA_2030_{suffix}.tif", nodata=1.0)
    save_geotiff(delta_2050_data, lons, lats, out_dir / f"HYDRO_ATLAS_DELTA_2050_{suffix}.tif", nodata=1.0)
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print(f"\nGeoTIFF rasters:")
    print(f"  - HYDRO_RUNOFF_2030_{suffix}.tif")
    print(f"  - HYDRO_RUNOFF_2050_{suffix}.tif")
    print(f"  - HYDRO_RUNOFF_UNCERTAINTY_2030_{suffix}.tif")
    print(f"  - HYDRO_RUNOFF_UNCERTAINTY_2050_{suffix}.tif")
    print(f"  - HYDRO_RUNOFF_ERA5_baseline_{suffix}.tif")
    print(f"  - HYDRO_ATLAS_DELTA_2030_{suffix}.tif (for HydroATLAS)")
    print(f"  - HYDRO_ATLAS_DELTA_2050_{suffix}.tif (for HydroATLAS)")


def run(workdir: str = None,
        era5_dir: str = None,
        download_only: bool = False,
        process_only: bool = False):
    """
    Main entry point.
    """
    if workdir is None:
        workdir = Path(get_bigdata_path("bigdata_hydro_cmip6"))
    else:
        workdir = Path(workdir)
    
    if era5_dir is None:
        era5_dir = Path(get_bigdata_path("bigdata_hydro_era5_land")) / "downloads"
    else:
        era5_dir = Path(era5_dir)
    
    if download_only:
        run_download_only(workdir, era5_dir)
    elif process_only:
        run_processing(workdir, era5_dir)
    else:
        run_download_only(workdir, era5_dir)
        run_processing(workdir, era5_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ERA5-Land + CMIP6 Delta Method for Runoff Projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data
  python p1_e_cmip6_hydro.py --download-only
  
  # Process only (assumes downloads exist)
  python p1_e_cmip6_hydro.py --process-only
  
  # Full pipeline
  python p1_e_cmip6_hydro.py
        """
    )
    
    parser.add_argument(
        "--workdir", 
        default=None,
        help="Working directory for CMIP6 downloads and outputs"
    )
    parser.add_argument(
        "--era5-dir",
        default=None,
        help="Directory for ERA5-Land downloads"
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
        era5_dir=args.era5_dir,
        download_only=args.download_only,
        process_only=args.process_only
    )
