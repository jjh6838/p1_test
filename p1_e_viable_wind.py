"""
p1_e_viable_wind.py
===================
ERA5 + CMIP6 Delta Method for Wind Power Density (WPD) Projections

Methodology:
1. Load ERA5 100m wind baseline (1995-2014 monthly means)
2. Load CMIP6 sfcWind for historical (1995-2014) and SSP245 (2021-2060)
3. Compute climate change ratio: Δ = CMIP6_future / CMIP6_historical
4. Apply delta to ERA5: U100_future = U100_ERA5 × Δ
5. Compute WPD = 0.5 × ρ × U³
6. Ensemble mean across 3 CMIP6 models
7. Regrid to 300 arcsec and apply Global Wind Atlas mask

Output:
- WPD100_2030_300arcsec.tif
- WPD100_2050_300arcsec.tif
- WPD100_UNCERTAINTY_2030_300arcsec.tif (IQR across models)
- WPD100_UNCERTAINTY_2050_300arcsec.tif
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
from rasterio.features import rasterize
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

# CMIP6 models for ensemble
CMIP6_MODELS = [
    "cesm2",
    "ec_earth3_veg_lr",
    "mpi_esm1_2_lr",
]

# Periods
HIST_PERIOD = ("1995-01-01", "2014-12-31")  # Historical baseline (aligns ERA5 & CMIP6)
P2030 = ("2021-01-01", "2040-12-31")        # 20-year mean centered on ~2030
P2050 = ("2041-01-01", "2060-12-31")        # 20-year mean centered on ~2050

# CDS settings
CDS_CMIP6_DATASET = "projections-cmip6"
CDS_ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"
SCENARIO = "ssp2_4_5"

# Physical constants
AIR_DENSITY = 1.225  # kg/m³ at sea level, 15°C

# Output resolution settings - imported from shared config for consistency
from config import POP_AGGREGATION_FACTOR, TARGET_RESOLUTION_ARCSEC, GHS_POP_NATIVE_RESOLUTION_ARCSEC
from config import LANDCOVER_VALID_WIND, WIND_WPD_THRESHOLD

# Global Wind Atlas mask settings
GWA_EXCLUDE_CLASS = 12  # Class S (unsuitable for wind)

# Land cover data path
LANDCOVER_PATH = Path(get_bigdata_path("bigdata_landcover_cds")) / "outputs" / "landcover_2022_300arcsec.tif"

# Microsoft viable wind sites (pre-identified wind installations)
MS_WIND_PATH = Path(get_bigdata_path("bigdata_solar_wind_ms")) / "wind_all_2024q2_v1.gpkg"


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_era5(out_dir: Path) -> Path:
    """
    Download ERA5 monthly 100m wind components (1995-2014).
    Returns path to downloaded NetCDF file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "era5_100m_wind_1995-2014.nc"
    
    if out_file.exists():
        print(f"[skip] ERA5 already downloaded: {out_file}")
        return out_file
    
    print("[download] ERA5 100m wind components (1995-2014)...")
    
    years = [str(y) for y in range(1995, 2015)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind"
        ],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client.retrieve(CDS_ERA5_DATASET, request, str(out_file))
    print(f"[done] ERA5 downloaded: {out_file}")
    return out_file


def download_cmip6_historical(model: str, out_dir: Path) -> Optional[Path]:
    """
    Download CMIP6 historical sfcWind (1995-2014).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_historical_1995-2014.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 historical already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 historical: {model} (1995-2014)...")
    
    years = [str(y) for y in range(1995, 2015)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "near_surface_wind_speed",
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
    Download CMIP6 SSP245 sfcWind (2021-2060).
    Returns path to downloaded ZIP file or None if failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_ssp245_2021-2060.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 SSP245 already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 SSP245: {model} (2021-2060)...")
    
    years = [str(y) for y in range(2021, 2061)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": SCENARIO,
        "variable": "near_surface_wind_speed",
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

def load_era5_wind(nc_path: Path) -> xr.DataArray:
    """
    Load ERA5 100m wind and compute wind speed from u/v components.
    Returns wind speed DataArray with dimensions (time, lat, lon).
    """
    print(f"  Loading ERA5 from {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    
    # Identify u and v components (handle different naming conventions)
    u_var = None
    v_var = None
    for var in ds.data_vars:
        var_lower = var.lower()
        if 'u100' in var_lower or ('u' in var_lower and '100' in var_lower):
            u_var = var
        elif 'v100' in var_lower or ('v' in var_lower and '100' in var_lower):
            v_var = var
    
    if u_var is None or v_var is None:
        # Try standard ERA5 names
        if 'u100' in ds.data_vars:
            u_var, v_var = 'u100', 'v100'
        else:
            raise RuntimeError(f"Cannot find u100/v100 in ERA5. Variables: {list(ds.data_vars)}")
    
    print(f"  Using u={u_var}, v={v_var}")
    
    # Compute wind speed
    u = ds[u_var]
    v = ds[v_var]
    wind_speed = np.sqrt(u**2 + v**2)
    wind_speed.name = "wind_speed_100m"
    
    # Standardize coordinates
    wind_speed = standardize_coords(wind_speed)
    
    return wind_speed


def load_cmip6_wind(nc_paths: list[Path]) -> xr.DataArray:
    """
    Load CMIP6 sfcWind from NetCDF files.
    Returns wind speed DataArray with dimensions (time, lat, lon).
    """
    ds = xr.open_mfdataset([str(p) for p in nc_paths], combine="by_coords")
    
    # Find the sfcWind variable (skip auxiliary vars like *_bnds)
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
            # ERA5 uses 'valid_time', CMIP6 uses 'time'
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
    hist_safe = cmip6_hist.where(cmip6_hist > 0.1, 0.1)
    delta = cmip6_future / hist_safe
    
    # Clip to reasonable range
    delta = delta.clip(min=0.5, max=2.0)
    
    return delta


def apply_delta_to_era5(era5_baseline: xr.DataArray, delta: xr.DataArray) -> xr.DataArray:
    """
    Apply CMIP6 delta to ERA5 baseline.
    Interpolates delta to ERA5 grid first.
    """
    # Interpolate delta to ERA5 grid
    delta_interp = delta.interp(
        lat=era5_baseline.lat,
        lon=era5_baseline.lon,
        method="linear"
    )
    
    # Apply delta
    future_wind = era5_baseline * delta_interp
    
    return future_wind


def compute_wpd(wind_speed: xr.DataArray) -> xr.DataArray:
    """
    Compute Wind Power Density: WPD = 0.5 × ρ × U³
    Units: W/m²
    """
    wpd = 0.5 * AIR_DENSITY * (wind_speed ** 3)
    wpd.attrs['units'] = 'W/m²'
    wpd.attrs['long_name'] = 'Wind Power Density at 100m'
    return wpd


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
        import rasterio
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


def regrid_to_target(da: xr.DataArray, target_res_arcsec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrid DataArray to target resolution.
    Grid is aligned with GHS-POP raster for consistency with settlement centroids.
    Returns (data, lons, lats) at target resolution.
    
    The output grid exactly matches what process_country_supply.py produces via
    rasterio.transform.xy() after aggregating GHS-POP by POP_AGGREGATION_FACTOR.
    """
    # Source grid
    src_data = da.values
    src_lons = da.lon.values
    src_lats = da.lat.values
    
    # Ensure source lats are ascending for interpolation
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]
    
    # Get GHS-POP grid parameters (origin and pixel size)
    ghs_params = get_ghs_pop_grid_params()
    
    # Calculate aggregation factor from target resolution
    ghs_native_arcsec = abs(ghs_params['pixel_size_lon']) * 3600
    agg_factor = round(target_res_arcsec / ghs_native_arcsec)
    
    # Aggregated pixel size (match GHS-POP exactly, don't use target_res_arcsec/3600)
    agg_pixel_lon = ghs_params['pixel_size_lon'] * agg_factor
    agg_pixel_lat = ghs_params['pixel_size_lat'] * agg_factor  # negative
    
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
    
    # Interpolate using ascending lats for correct results
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
    
    # Flip data back to match descending lats (row 0 = north)
    target_data = target_data[::-1, :]
    
    # Return descending lats (standard GeoTIFF order: row 0 = north)
    return target_data, target_lons, target_lats


def apply_gwa_mask(data: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                   gwa_path: Path, exclude_class: int = 12) -> np.ndarray:
    """
    Apply Global Wind Atlas mask to exclude unsuitable areas.
    Sets WPD to 0 where GWA class == exclude_class.
    """
    if not gwa_path.exists():
        print(f"  [warning] GWA mask not found: {gwa_path}")
        return data
    
    print(f"  Applying GWA mask (excluding class {exclude_class})...")
    
    with rasterio.open(gwa_path) as src:
        gwa_data = src.read(1)
        gwa_transform = src.transform
        gwa_crs = src.crs
    
    # Create output grid coordinates
    height, width = data.shape
    transform = from_bounds(
        lons.min(), lats.min(), lons.max(), lats.max(),
        width, height
    )
    
    # Reproject GWA mask to match our grid
    mask_reprojected = np.empty((height, width), dtype=gwa_data.dtype)
    
    reproject(
        source=gwa_data,
        destination=mask_reprojected,
        src_transform=gwa_transform,
        src_crs=gwa_crs,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest
    )
    
    # Apply mask: set to 0 where GWA class == exclude_class
    data_masked = data.copy()
    data_masked[mask_reprojected == exclude_class] = 0
    
    # Count masked pixels
    n_masked = np.sum(mask_reprojected == exclude_class)
    print(f"  Masked {n_masked:,} pixels (class {exclude_class})")
    
    return data_masked


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
    """Save data array as Parquet centroids (points with values > 0)."""
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


def save_viable_centroids(data_projected: np.ndarray, data_baseline: np.ndarray,
                          data_uncertainty: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                          out_path: Path, year: str, resource_name: str = "wind") -> None:
    """
    Save viable centroids with unified schema across solar/wind/hydro.
    
    Schema: geometry, source, value_{year}, value_baseline, delta, uncertainty
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure lats are in correct order for grid creation
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data_projected = data_projected[::-1, :]
        data_baseline = data_baseline[::-1, :]
        data_uncertainty = data_uncertainty[::-1, :]
    
    # Create meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten arrays
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    proj_flat = data_projected.ravel()
    base_flat = data_baseline.ravel()
    unc_flat = data_uncertainty.ravel()
    
    # Filter to valid cells (projected value > 0)
    valid_mask = ~np.isnan(proj_flat) & (proj_flat > 0)
    
    # Compute delta
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = np.where(base_flat > 0, proj_flat / base_flat, 1.0)
    
    # Create GeoDataFrame with unified schema
    gdf = gpd.GeoDataFrame({
        'geometry': gpd.points_from_xy(lon_flat[valid_mask], lat_flat[valid_mask]),
        'source': resource_name,
        f'value_{year}': proj_flat[valid_mask],
        'value_baseline': base_flat[valid_mask],
        'delta': delta[valid_mask],
        'uncertainty': unc_flat[valid_mask],
    }, crs="EPSG:4326")
    
    # Save as parquet
    gdf.to_parquet(out_path)
    print(f"  Saved: {out_path} ({len(gdf):,} centroids)")


def apply_viability_filter(data: np.ndarray, 
                            target_lons: np.ndarray, 
                            target_lats: np.ndarray,
                            landcover_path: Path,
                            valid_classes: list,
                            resource_threshold: float,
                            ms_viable_path: Path = None,
                            nodata_value: float = 0) -> np.ndarray:
    """
    Apply combined viability filter:
    Keep cell if (MS site present) OR (land cover valid AND resource >= threshold).
    
    Args:
        data: Input data array at target resolution (resource values)
        target_lons: Target longitude coordinates (1D array)
        target_lats: Target latitude coordinates (1D array, descending)
        landcover_path: Path to ESA CCI land cover GeoTIFF
        valid_classes: List of valid land cover class codes
        resource_threshold: Minimum resource value for viability (when using land cover)
        ms_viable_path: Path to Microsoft viable sites GeoPackage (optional)
        nodata_value: Value to use for masked cells
    
    Returns:
        Filtered data array with invalid areas set to nodata_value
    """
    height, width = data.shape
    
    # Ensure lats are descending for standard GeoTIFF orientation
    if target_lats[0] < target_lats[-1]:
        target_lats = target_lats[::-1]
    
    target_transform = from_bounds(
        target_lons.min(), target_lats.min(), 
        target_lons.max(), target_lats.max(),
        width, height
    )
    
    # Initialize masks
    ms_mask = np.zeros((height, width), dtype=bool)
    lc_mask = np.zeros((height, width), dtype=bool)
    
    # -------------------------------------------------------------------------
    # 1. Microsoft viable sites (unconditionally viable if present)
    # -------------------------------------------------------------------------
    if ms_viable_path is not None and ms_viable_path.exists():
        print(f"\n--- Loading MS viable sites ---")
        print(f"  Loading: {ms_viable_path.name}")
        
        gdf_ms = gpd.read_file(ms_viable_path)
        print(f"  Loaded {len(gdf_ms):,} features (CRS: {gdf_ms.crs})")
        
        gdf_ms = gdf_ms.to_crs("EPSG:4326")
        
        # Filter out None and empty geometries to avoid rasterization warnings
        shapes = [(geom, 1) for geom in gdf_ms.geometry 
                  if geom is not None and not geom.is_empty and geom.is_valid]
        n_skipped = len(gdf_ms) - len(shapes)
        if n_skipped > 0:
            print(f"  Skipped {n_skipped:,} invalid/empty geometries")
        
        if shapes:
            ms_rasterized = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=target_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            ms_mask = ms_rasterized > 0
            n_ms_valid = np.sum(ms_mask)
            print(f"  Cells with MS sites: {n_ms_valid:,}")
    elif ms_viable_path is not None:
        print(f"  [WARNING] MS viable sites not found: {ms_viable_path}")
    
    # -------------------------------------------------------------------------
    # 2. Land cover filter (requires resource threshold)
    # -------------------------------------------------------------------------
    if landcover_path.exists():
        print(f"\n--- Loading land cover (valid classes: {valid_classes}) ---")
        print(f"  Loading: {landcover_path.name}")
        
        with rasterio.open(landcover_path) as src:
            lc_data = src.read(1)
            lc_transform = src.transform
            lc_crs = src.crs
        
        # Create binary mask: 1 if in valid_classes
        binary_mask = np.zeros_like(lc_data, dtype=np.uint8)
        for cls in valid_classes:
            binary_mask[lc_data == cls] = 1
        
        # Reproject to target grid using max (any presence)
        lc_mask_reprojected = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=binary_mask,
            destination=lc_mask_reprojected,
            src_transform=lc_transform,
            src_crs=lc_crs,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.max
        )
        lc_mask = lc_mask_reprojected > 0
        n_lc_valid = np.sum(lc_mask)
        print(f"  Cells with valid land cover: {n_lc_valid:,}")
    else:
        print(f"  [WARNING] Land cover file not found: {landcover_path}")
    
    # -------------------------------------------------------------------------
    # 3. Apply combined viability logic
    # -------------------------------------------------------------------------
    # Resource threshold mask
    resource_valid = data >= resource_threshold
    n_resource_valid = np.sum(resource_valid & (data > 0))
    print(f"\n--- Applying viability filter (threshold >= {resource_threshold}) ---")
    print(f"  Cells meeting resource threshold: {n_resource_valid:,}")
    
    # Combined logic: MS_present OR (landcover_valid AND resource >= threshold)
    combined_mask = ms_mask | (lc_mask & resource_valid)
    
    n_ms_only = np.sum(ms_mask & ~(lc_mask & resource_valid))
    n_lc_resource = np.sum(~ms_mask & lc_mask & resource_valid)
    n_both = np.sum(ms_mask & lc_mask & resource_valid)
    n_total = np.sum(combined_mask)
    pct_total = 100 * n_total / combined_mask.size
    
    print(f"  Viable by MS sites only: {n_ms_only:,}")
    print(f"  Viable by land cover + resource: {n_lc_resource:,}")
    print(f"  Viable by both criteria: {n_both:,}")
    print(f"  Total viable cells: {n_total:,} ({pct_total:.2f}%)")
    
    # Apply mask
    data_filtered = data.copy()
    data_filtered[~combined_mask] = nodata_value
    
    return data_filtered


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_download_only(workdir: Path):
    """Download all required datasets."""
    era5_dir = Path(get_bigdata_path("bigdata_wind_era5")) / "downloads"
    cmip6_dir = workdir / "downloads"
    
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATA")
    print("="*70)
    
    # Download ERA5
    download_era5(era5_dir)
    
    # Download CMIP6 for each model
    for model in CMIP6_MODELS:
        print(f"\n--- {model} ---")
        download_cmip6_historical(model, cmip6_dir)
        download_cmip6_ssp245(model, cmip6_dir)
    
    print("\n[done] All downloads complete!")


def run_processing(workdir: Path, gwa_mask_path: Optional[Path] = None):
    """Process downloaded data using delta method."""
    era5_dir = Path(get_bigdata_path("bigdata_wind_era5")) / "downloads"
    cmip6_dl_dir = workdir / "downloads"
    cmip6_ex_dir = workdir / "extracted"
    out_dir = workdir / "outputs"
    
    print("\n" + "="*70)
    print("STEP 2: PROCESSING WITH DELTA METHOD")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Load ERA5 baseline
    # -------------------------------------------------------------------------
    print("\n--- Loading ERA5 baseline ---")
    era5_file = era5_dir / "era5_100m_wind_1995-2014.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_file}. Run with --download first.")
    
    era5_wind = load_era5_wind(era5_file)
    era5_baseline = compute_temporal_mean(era5_wind, HIST_PERIOD[0], HIST_PERIOD[1])
    print(f"  ERA5 baseline shape: {era5_baseline.shape}")
    print(f"  ERA5 mean wind speed: {float(era5_baseline.mean()):.2f} m/s")
    
    # -------------------------------------------------------------------------
    # Process each CMIP6 model
    # -------------------------------------------------------------------------
    delta_2030_list = []
    delta_2050_list = []
    wpd_2030_list = []
    wpd_2050_list = []
    successful_models = []
    
    for model in CMIP6_MODELS:
        print(f"\n--- Processing CMIP6 model: {model} ---")
        
        try:
            # Extract historical
            hist_zip = cmip6_dl_dir / f"{model}_historical_1995-2014.zip"
            if not hist_zip.exists():
                print(f"  [skip] Historical ZIP not found: {hist_zip}")
                continue
            
            hist_nc = extract_zip(hist_zip, cmip6_ex_dir / "historical" / model)
            cmip6_hist_wind = load_cmip6_wind(hist_nc)
            cmip6_hist_mean = compute_temporal_mean(cmip6_hist_wind, HIST_PERIOD[0], HIST_PERIOD[1])
            print(f"  Historical mean: {float(cmip6_hist_mean.mean()):.2f} m/s")
            
            # Extract SSP245
            ssp_zip = cmip6_dl_dir / f"{model}_ssp245_2021-2060.zip"
            if not ssp_zip.exists():
                print(f"  [skip] SSP245 ZIP not found: {ssp_zip}")
                continue
            
            ssp_nc = extract_zip(ssp_zip, cmip6_ex_dir / "ssp245" / model)
            cmip6_ssp_wind = load_cmip6_wind(ssp_nc)
            
            # Compute period means for 2030 and 2050
            cmip6_2030_mean = compute_temporal_mean(cmip6_ssp_wind, P2030[0], P2030[1])
            cmip6_2050_mean = compute_temporal_mean(cmip6_ssp_wind, P2050[0], P2050[1])
            print(f"  2030 mean: {float(cmip6_2030_mean.mean()):.2f} m/s")
            print(f"  2050 mean: {float(cmip6_2050_mean.mean()):.2f} m/s")
            
            # Compute delta ratios
            delta_2030 = compute_delta_ratio(cmip6_2030_mean, cmip6_hist_mean)
            delta_2050 = compute_delta_ratio(cmip6_2050_mean, cmip6_hist_mean)
            print(f"  Delta 2030 mean: {float(delta_2030.mean()):.3f}")
            print(f"  Delta 2050 mean: {float(delta_2050.mean()):.3f}")
            
            # Apply delta to ERA5
            wind_2030 = apply_delta_to_era5(era5_baseline, delta_2030)
            wind_2050 = apply_delta_to_era5(era5_baseline, delta_2050)
            
            # Compute WPD
            wpd_2030 = compute_wpd(wind_2030)
            wpd_2050 = compute_wpd(wind_2050)
            
            # Store for ensemble
            delta_2030_list.append(delta_2030)
            delta_2050_list.append(delta_2050)
            wpd_2030_list.append(wpd_2030)
            wpd_2050_list.append(wpd_2050)
            successful_models.append(model)
            
            print(f"  [done] {model}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {model}: {e}")
            continue
    
    if not wpd_2030_list:
        raise RuntimeError("No CMIP6 models processed successfully!")
    
    print(f"\n--- Successfully processed {len(successful_models)} models: {successful_models} ---")
    
    # -------------------------------------------------------------------------
    # Compute ensemble statistics
    # -------------------------------------------------------------------------
    print("\n--- Computing ensemble statistics ---")
    
    # Ensemble mean WPD
    print("  Computing ensemble mean...")
    wpd_2030_ensemble = xr.concat(wpd_2030_list, dim="model").mean("model", skipna=True)
    wpd_2050_ensemble = xr.concat(wpd_2050_list, dim="model").mean("model", skipna=True)
    
    # Ensemble uncertainty: use range (max - min) instead of IQR
    # This is faster and nearly equivalent for 3 models
    print("  Computing ensemble uncertainty (range)...")
    wpd_2030_stack = xr.concat(wpd_2030_list, dim="model")
    wpd_2050_stack = xr.concat(wpd_2050_list, dim="model")
    
    wpd_2030_iqr = wpd_2030_stack.max(dim="model") - wpd_2030_stack.min(dim="model")
    wpd_2050_iqr = wpd_2050_stack.max(dim="model") - wpd_2050_stack.min(dim="model")
    
    print(f"  Ensemble WPD 2030 mean: {float(wpd_2030_ensemble.mean()):.1f} W/m²")
    print(f"  Ensemble WPD 2050 mean: {float(wpd_2050_ensemble.mean()):.1f} W/m²")
    
    # -------------------------------------------------------------------------
    # Regrid to 300 arcsec and apply mask
    # -------------------------------------------------------------------------
    print(f"\n--- Regridding to {TARGET_RESOLUTION_ARCSEC} arcsec ---")
    
    wpd_2030_data, lons, lats = regrid_to_target(wpd_2030_ensemble, TARGET_RESOLUTION_ARCSEC)
    wpd_2050_data, _, _ = regrid_to_target(wpd_2050_ensemble, TARGET_RESOLUTION_ARCSEC)
    iqr_2030_data, _, _ = regrid_to_target(wpd_2030_iqr, TARGET_RESOLUTION_ARCSEC)
    iqr_2050_data, _, _ = regrid_to_target(wpd_2050_iqr, TARGET_RESOLUTION_ARCSEC)
    
    print(f"  Output shape: {wpd_2030_data.shape}")
    
    # Apply GWA mask if provided
    if gwa_mask_path and gwa_mask_path.exists():
        wpd_2030_data = apply_gwa_mask(wpd_2030_data, lons, lats, gwa_mask_path, GWA_EXCLUDE_CLASS)
        wpd_2050_data = apply_gwa_mask(wpd_2050_data, lons, lats, gwa_mask_path, GWA_EXCLUDE_CLASS)
        iqr_2030_data = apply_gwa_mask(iqr_2030_data, lons, lats, gwa_mask_path, GWA_EXCLUDE_CLASS)
        iqr_2050_data = apply_gwa_mask(iqr_2050_data, lons, lats, gwa_mask_path, GWA_EXCLUDE_CLASS)
    
    # -------------------------------------------------------------------------
    # Apply viability filter: MS_present OR (landcover_valid AND resource >= threshold)
    # -------------------------------------------------------------------------
    wpd_2030_data = apply_viability_filter(
        wpd_2030_data, lons, lats, LANDCOVER_PATH, LANDCOVER_VALID_WIND,
        resource_threshold=WIND_WPD_THRESHOLD, ms_viable_path=MS_WIND_PATH, nodata_value=0
    )
    wpd_2050_data = apply_viability_filter(
        wpd_2050_data, lons, lats, LANDCOVER_PATH, LANDCOVER_VALID_WIND,
        resource_threshold=WIND_WPD_THRESHOLD, ms_viable_path=MS_WIND_PATH, nodata_value=0
    )
    # Uncertainty rasters: use threshold=0 to keep all cells that pass viability
    iqr_2030_data = apply_viability_filter(
        iqr_2030_data, lons, lats, LANDCOVER_PATH, LANDCOVER_VALID_WIND,
        resource_threshold=0, ms_viable_path=MS_WIND_PATH, nodata_value=0
    )
    iqr_2050_data = apply_viability_filter(
        iqr_2050_data, lons, lats, LANDCOVER_PATH, LANDCOVER_VALID_WIND,
        resource_threshold=0, ms_viable_path=MS_WIND_PATH, nodata_value=0
    )
    
    # -------------------------------------------------------------------------
    # Save outputs (GeoTIFF and Parquet)
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs (GeoTIFF) ---")
    
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    
    save_geotiff(wpd_2030_data, lons, lats, out_dir / f"WPD100_2030_{suffix}.tif")
    save_geotiff(wpd_2050_data, lons, lats, out_dir / f"WPD100_2050_{suffix}.tif")
    save_geotiff(iqr_2030_data, lons, lats, out_dir / f"WPD100_UNCERTAINTY_2030_{suffix}.tif")
    save_geotiff(iqr_2050_data, lons, lats, out_dir / f"WPD100_UNCERTAINTY_2050_{suffix}.tif")
    
    # Also save ERA5 baseline WPD for reference
    era5_wpd = compute_wpd(era5_baseline)
    era5_wpd_data, _, _ = regrid_to_target(era5_wpd, TARGET_RESOLUTION_ARCSEC)
    if gwa_mask_path and gwa_mask_path.exists():
        era5_wpd_data = apply_gwa_mask(era5_wpd_data, lons, lats, gwa_mask_path, GWA_EXCLUDE_CLASS)
    era5_wpd_data = apply_viability_filter(
        era5_wpd_data, lons, lats, LANDCOVER_PATH, LANDCOVER_VALID_WIND,
        resource_threshold=WIND_WPD_THRESHOLD, ms_viable_path=MS_WIND_PATH, nodata_value=0
    )
    save_geotiff(era5_wpd_data, lons, lats, out_dir / f"WPD100_baseline_{suffix}.tif")
    
    print("\n--- Saving outputs (Parquet viable centroids) ---")
    
    save_as_parquet(wpd_2030_data, lons, lats, out_dir / f"WPD100_2030_{suffix}.parquet", "WPD_2030")
    save_as_parquet(wpd_2050_data, lons, lats, out_dir / f"WPD100_2050_{suffix}.parquet", "WPD_2050")
    save_as_parquet(iqr_2030_data, lons, lats, out_dir / f"WPD100_UNCERTAINTY_2030_{suffix}.parquet", "WPD_UNC_2030")
    save_as_parquet(iqr_2050_data, lons, lats, out_dir / f"WPD100_UNCERTAINTY_2050_{suffix}.parquet", "WPD_UNC_2050")
    save_as_parquet(era5_wpd_data, lons, lats, out_dir / f"WPD100_baseline_{suffix}.parquet", "WPD_baseline")
    
    # Save unified viable centroids (consistent schema with solar/hydro)
    print("\n--- Saving unified viable centroids ---")
    save_viable_centroids(wpd_2030_data, era5_wpd_data, iqr_2030_data, lons, lats,
                          out_dir / "WIND_VIABLE_CENTROIDS_2030.parquet", "2030", "wind")
    save_viable_centroids(wpd_2050_data, era5_wpd_data, iqr_2050_data, lons, lats,
                          out_dir / "WIND_VIABLE_CENTROIDS_2050.parquet", "2050", "wind")
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print(f"  GeoTIFF rasters:")
    print(f"    - WPD100_2030_{suffix}.tif")
    print(f"    - WPD100_2050_{suffix}.tif")
    print(f"    - WPD100_UNCERTAINTY_2030_{suffix}.tif")
    print(f"    - WPD100_UNCERTAINTY_2050_{suffix}.tif")
    print(f"    - WPD100_baseline_{suffix}.tif")
    print(f"  Parquet centroids:")
    print(f"    - WPD100_2030_{suffix}.parquet")
    print(f"    - WPD100_2050_{suffix}.parquet")
    print(f"    - WPD100_UNCERTAINTY_2030_{suffix}.parquet")
    print(f"    - WPD100_UNCERTAINTY_2050_{suffix}.parquet")
    print(f"    - WPD100_baseline_{suffix}.parquet")
    print(f"  Unified viable centroids:")
    print(f"    - WIND_VIABLE_CENTROIDS_2030.parquet")
    print(f"    - WIND_VIABLE_CENTROIDS_2050.parquet")


def run(workdir: str = None,
        gwa_mask: str = None,
        download_only: bool = False,
        process_only: bool = False):
    """
    Main entry point.
    
    Args:
        workdir: Working directory for CMIP6 data
        gwa_mask: Path to Global Wind Atlas suitability mask
        download_only: Only download data, don't process
        process_only: Only process data, assume downloads exist
    """
    # Use get_bigdata_path for default directories
    if workdir is None:
        workdir = Path(get_bigdata_path("bigdata_wind_cmip6"))
    else:
        workdir = Path(workdir)
    
    # Default GWA mask path
    if gwa_mask is None:
        gwa_path = Path(get_bigdata_path("bigdata_wind_atlas")) / "gasp_flsclassnowake_100m.tif"
    else:
        gwa_path = Path(gwa_mask)
    
    if download_only:
        run_download_only(workdir)
    elif process_only:
        run_processing(workdir, gwa_path)
    else:
        # Full pipeline
        run_download_only(workdir)
        run_processing(workdir, gwa_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ERA5 + CMIP6 Delta Method for Wind Power Density Projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data
  python p1_e_viable_wind.py --download-only
  
  # Process with GWA mask
  python p1_e_viable_wind.py --process-only --gwa-mask ./bigdata_windatlas/gasp_flsclassnowake_100m.tif
  
  # Full pipeline
  python p1_e_viable_wind.py --gwa-mask ./bigdata_windatlas/gasp_flsclassnowake_100m.tif
        """
    )
    
    parser.add_argument(
        "--workdir", 
        default=None,
        help="Working directory for CMIP6 downloads and outputs (auto-detects local/cluster)"
    )
    parser.add_argument(
        "--gwa-mask",
        default=None,
        help="Path to Global Wind Atlas suitability mask (auto-detects local/cluster, class 12 = excluded)"
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
        gwa_mask=args.gwa_mask,
        download_only=args.download_only,
        process_only=args.process_only
    )
