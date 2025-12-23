"""
p1_f_utils_hydro.py
===================
Shared utility functions for hydro processing scripts.

Used by:
  - p1_f_viable_hydro.py
"""

import os
import zipfile
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.interpolate import RegularGridInterpolator
from shapely import get_coordinates

import cdsapi

# Suppress numpy/dask warnings for NaN operations
warnings.filterwarnings('ignore', 'Mean of empty slice')
warnings.filterwarnings('ignore', 'All-NaN slice encountered')
warnings.filterwarnings('ignore', 'CRS mismatch')


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_bigdata_path(folder_name: str) -> str:
    """Get correct path for bigdata folders (local vs cluster)."""
    local_path = folder_name
    cluster_path = f"/soge-home/projects/mistral/ji/{folder_name}"
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(cluster_path):
        return cluster_path
    else:
        return local_path


# =============================================================================
# CONSTANTS
# =============================================================================

# CMIP6 models for ensemble
CMIP6_MODELS = ["cesm2", "ec_earth3_veg_lr", "mpi_esm1_2_lr"]

# Time periods
HIST_PERIOD = ("1995-01-01", "2014-12-31")
P2030 = ("2021-01-01", "2040-12-31")
P2050 = ("2041-01-01", "2060-12-31")

# CDS settings
CDS_CMIP6_DATASET = "projections-cmip6"
CDS_ERA5_LAND_DATASET = "reanalysis-era5-land-monthly-means"
SCENARIO = "ssp2_4_5"

# Unit conversions
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MM_PER_METER = 1000

# RiverATLAS columns
RIVERATLAS_COLUMNS = [
    'HYRIV_ID', 'NEXT_DOWN', 'MAIN_RIV', 'LENGTH_KM', 'ORD_STRA', 'ORD_CLAS',
    'ORD_FLOW', 'HYBAS_L12', 'dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx',
    'run_mm_cyr', 'ele_mt_cav', 'ele_mt_uav', 'ele_mt_cmn', 'ele_mt_cmx',
    'slp_dg_cav', 'slp_dg_uav', 'sgr_dk_rav', 'UPLAND_SKM', 'tmp_dc_cyr',
    'pre_mm_cyr', 'dor_pc_pva', 'rev_mc_usu',
]

MIN_DISCHARGE_M3S = 0.1
MIN_STREAM_ORDER = 2


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_era5_land(out_dir: Path) -> Path:
    """Download ERA5-Land monthly runoff (1995-2014)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "era5_land_runoff_1995-2014.nc"
    
    if out_file.exists():
        print(f"[skip] ERA5-Land already downloaded: {out_file}")
        return out_file
    
    print("[download] ERA5-Land runoff (1995-2014)...")
    
    client = cdsapi.Client()
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["runoff"],
        "year": [str(y) for y in range(1995, 2015)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client.retrieve(CDS_ERA5_LAND_DATASET, request, str(out_file))
    print(f"[done] ERA5-Land downloaded: {out_file}")
    return out_file


def download_cmip6_historical(model: str, out_dir: Path) -> Optional[Path]:
    """Download CMIP6 historical total_runoff (1995-2014)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_historical_runoff_1995-2014.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 historical already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 historical runoff: {model} (1995-2014)...")
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "total_runoff",
        "model": model,
        "year": [str(y) for y in range(1995, 2015)],
        "month": [f"{m:02d}" for m in range(1, 13)],
    }
    
    try:
        client.retrieve(CDS_CMIP6_DATASET, request, str(out_file))
        print(f"[done] {model} historical downloaded")
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to download {model} historical: {e}")
        return None


def download_cmip6_ssp245(model: str, out_dir: Path) -> Optional[Path]:
    """Download CMIP6 SSP245 total_runoff (2021-2060)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model}_ssp245_runoff_2021-2060.zip"
    
    if out_file.exists():
        print(f"[skip] CMIP6 SSP245 already downloaded: {out_file}")
        return out_file
    
    print(f"[download] CMIP6 SSP245 runoff: {model} (2021-2060)...")
    
    client = cdsapi.Client()
    request = {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": SCENARIO,
        "variable": "total_runoff",
        "model": model,
        "year": [str(y) for y in range(2021, 2061)],
        "month": [f"{m:02d}" for m in range(1, 13)],
    }
    
    try:
        client.retrieve(CDS_CMIP6_DATASET, request, str(out_file))
        print(f"[done] {model} SSP245 downloaded")
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to download {model} SSP245: {e}")
        return None


def extract_zip(zip_path: Path, out_dir: Path) -> list:
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
    
    if 'lon' in da.coords:
        lon_values = da.lon.values
        if lon_values.max() > 180:
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    
    return da


def load_era5_land_runoff(nc_path: Path) -> xr.DataArray:
    """Load ERA5-Land runoff data."""
    print(f"  Loading ERA5-Land from {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    
    runoff_var = None
    for var in ds.data_vars:
        if 'runoff' in var.lower() or var == 'ro':
            runoff_var = var
            break
    if runoff_var is None and 'ro' in ds.data_vars:
        runoff_var = 'ro'
    if runoff_var is None:
        raise RuntimeError(f"Cannot find runoff variable. Variables: {list(ds.data_vars)}")
    
    print(f"  Using variable: {runoff_var}")
    return standardize_coords(ds[runoff_var])


def load_cmip6_runoff(nc_paths: list) -> xr.DataArray:
    """Load CMIP6 total_runoff from NetCDF files."""
    ds = xr.open_mfdataset([str(p) for p in nc_paths], combine="by_coords")
    data_vars = [v for v in ds.data_vars if not v.endswith('_bnds')]
    if not data_vars:
        raise RuntimeError("No data variables found in CMIP6 NetCDF.")
    var_name = data_vars[0]
    print(f"  Using variable: {var_name}")
    return standardize_coords(ds[var_name])


def load_riveratlas(gdb_path: Path, columns: List[str] = None,
                    min_discharge: float = None, min_order: int = None,
                    bbox: tuple = None) -> gpd.GeoDataFrame:
    """Load RiverATLAS data from FileGDB."""
    import fiona
    print(f"Loading RiverATLAS from {gdb_path}...")
    
    layers = fiona.listlayers(gdb_path)
    print(f"  Available layers: {layers}")
    
    river_layer = None
    for layer in layers:
        if 'river' in layer.lower():
            river_layer = layer
            break
    if river_layer is None:
        river_layer = layers[0]
    
    print(f"  Using layer: {river_layer}")
    
    if bbox:
        gdf = gpd.read_file(gdb_path, layer=river_layer, bbox=bbox)
    else:
        gdf = gpd.read_file(gdb_path, layer=river_layer)
    
    print(f"  Loaded {len(gdf):,} river reaches")
    
    if min_discharge and 'dis_m3_pyr' in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf['dis_m3_pyr'] >= min_discharge]
        print(f"  Filtered by discharge >= {min_discharge}: {before:,} → {len(gdf):,}")
    
    if min_order and 'ORD_STRA' in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf['ORD_STRA'] >= min_order]
        print(f"  Filtered by stream order >= {min_order}: {before:,} → {len(gdf):,}")
    
    if columns:
        available_cols = ['geometry'] + [c for c in columns if c in gdf.columns]
        gdf = gdf[available_cols]
    
    return gdf


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================

def era5_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """Convert ERA5-Land runoff from meters (monthly) to mm/year."""
    annual_mean = da.groupby('time.year').sum('time').mean('year')
    return annual_mean * MM_PER_METER


def cmip6_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """Convert CMIP6 runoff from kg/m²/s to mm/year."""
    return da * SECONDS_PER_YEAR


# =============================================================================
# DELTA METHOD FUNCTIONS
# =============================================================================

def compute_temporal_mean(da: xr.DataArray, start_date: str, end_date: str) -> xr.DataArray:
    """Compute mean over specified time period."""
    return da.sel(time=slice(start_date, end_date)).mean("time", skipna=True)


def compute_delta_ratio(cmip6_future: xr.DataArray, cmip6_hist: xr.DataArray) -> xr.DataArray:
    """Compute climate change ratio: Δ = future / historical."""
    hist_safe = cmip6_hist.where(cmip6_hist > 1e-10, 1e-10)
    delta = cmip6_future / hist_safe
    return delta.clip(min=0.2, max=3.0)


def apply_delta_to_era5(era5_baseline: xr.DataArray, delta: xr.DataArray) -> xr.DataArray:
    """Apply CMIP6 delta to ERA5 baseline."""
    delta_interp = delta.interp(lat=era5_baseline.lat, lon=era5_baseline.lon, method="linear")
    return era5_baseline * delta_interp


# =============================================================================
# REGRIDDING AND OUTPUT FUNCTIONS
# =============================================================================

_GHS_POP_GRID_PARAMS = None


def get_ghs_pop_grid_params() -> dict:
    """Get GHS-POP raster grid parameters."""
    global _GHS_POP_GRID_PARAMS
    if _GHS_POP_GRID_PARAMS is not None:
        return _GHS_POP_GRID_PARAMS
    
    ghs_pop_path = Path(get_bigdata_path('bigdata_settlements_jrc')) / 'GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif'
    
    if ghs_pop_path.exists():
        with rasterio.open(ghs_pop_path) as src:
            t = src.transform
            _GHS_POP_GRID_PARAMS = {
                'origin_lon': t.c, 'origin_lat': t.f,
                'pixel_size_lon': t.a, 'pixel_size_lat': t.e,
            }
    else:
        _GHS_POP_GRID_PARAMS = {
            'origin_lon': -180.00791593130032, 'origin_lat': 89.0995831776456,
            'pixel_size_lon': 0.008333333300326923, 'pixel_size_lat': -0.00833333329979504,
        }
    return _GHS_POP_GRID_PARAMS


def regrid_to_target(da: xr.DataArray, target_res_arcsec: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid DataArray to target resolution aligned with GHS-POP grid."""
    src_data = da.values
    src_lons = da.lon.values
    src_lats = da.lat.values
    
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
        (src_lats, src_lons), src_data,
        method='linear', bounds_error=False, fill_value=np.nan
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
    
    transform = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), width, height)
    
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "compress": "deflate", "nodata": nodata, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    
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


# =============================================================================
# RIVER PROXIMITY MASK
# =============================================================================

def create_river_proximity_mask(gdf: gpd.GeoDataFrame, out_path: Path,
                                 buffer_distance_m: float, resolution_arcsec: float) -> Path:
    """Create binary raster mask of areas within buffer distance of rivers."""
    print(f"\n--- Creating river proximity mask ({buffer_distance_m/1000:.0f}km buffer) ---")
    
    if len(gdf) == 0:
        print("  [WARNING] No river reaches to buffer!")
        return None
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Project to equal-area CRS for accurate buffering
    print(f"  Buffering {len(gdf):,} river reaches...")
    gdf_ea = gdf.to_crs("ESRI:54034")
    gdf_ea['geometry'] = gdf_ea.geometry.buffer(buffer_distance_m)
    
    print("  Dissolving buffers...")
    dissolved = gdf_ea.dissolve().to_crs("EPSG:4326")
    
    # Define output grid
    res_deg = resolution_arcsec / 3600.0
    width = int(round(360 / res_deg))
    height = int(round(180 / res_deg))
    transform = from_bounds(-180, -90, 180, 90, width, height)
    
    print(f"  Output grid: {width} x {height} pixels")
    print("  Rasterizing buffer polygons...")
    
    shapes = [(geom, 1) for geom in dissolved.geometry]
    mask = rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                     fill=0, dtype=np.uint8, all_touched=True)
    
    n_valid = np.sum(mask == 1)
    print(f"  Mask coverage: {n_valid:,} pixels ({100*n_valid/mask.size:.2f}%)")
    
    profile = {
        'driver': 'GTiff', 'dtype': 'uint8', 'width': width, 'height': height,
        'count': 1, 'crs': 'EPSG:4326', 'transform': transform, 'compress': 'lzw',
        'nodata': 255, 'tiled': True, 'blockxsize': 512, 'blockysize': 512,
    }
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask, 1)
    
    print(f"  [done] River proximity mask saved: {out_path.name}")
    return out_path


# =============================================================================
# HYDRO FILTER (LAND COVER + RIVER PROXIMITY)
# =============================================================================

def apply_hydro_filter(data: np.ndarray, target_lons: np.ndarray, target_lats: np.ndarray,
                       landcover_path: Path, river_mask_path: Path, valid_classes: list,
                       resource_threshold: float = 0, nodata_value: float = 0, 
                       verbose: bool = True) -> np.ndarray:
    """
    Apply combined hydro filter: river_proximity OR (landcover_valid AND runoff >= threshold).
    
    This matches the solar/wind pattern where:
    - Existing infrastructure (river proximity) → always viable
    - Greenfield (water/wetland landcover) → only viable if resource >= threshold
    
    Args:
        data: Runoff data array (mm/year)
        target_lons: Longitude coordinates
        target_lats: Latitude coordinates
        landcover_path: Path to land cover GeoTIFF
        river_mask_path: Path to river proximity mask GeoTIFF
        valid_classes: Valid land cover classes for hydro (water/wetland)
        resource_threshold: Minimum runoff (mm/year) for landcover-based viability
        nodata_value: Value to use for non-viable cells
        verbose: Print progress information
    
    Returns:
        Filtered data array with non-viable cells set to nodata_value
    """
    height, width = data.shape
    
    if target_lats[0] < target_lats[-1]:
        target_lats = target_lats[::-1]
    
    target_transform = from_bounds(target_lons.min(), target_lats.min(),
                                    target_lons.max(), target_lats.max(), width, height)
    
    river_mask = np.zeros((height, width), dtype=bool)
    lc_mask = np.zeros((height, width), dtype=bool)
    
    # River proximity mask (always viable if near rivers)
    if river_mask_path.exists():
        if verbose:
            print(f"\n--- Loading river proximity mask ---")
        with rasterio.open(river_mask_path) as src:
            river_data = src.read(1)
            river_transform = src.transform
            river_crs = src.crs
        
        river_mask_reprojected = np.zeros((height, width), dtype=np.uint8)
        reproject(source=river_data, destination=river_mask_reprojected,
                  src_transform=river_transform, src_crs=river_crs,
                  dst_transform=target_transform, dst_crs="EPSG:4326",
                  resampling=Resampling.max)
        
        river_mask = river_mask_reprojected > 0
        if verbose:
            print(f"  Cells near rivers: {np.sum(river_mask):,}")
    elif verbose:
        print(f"  [WARNING] River proximity mask not found: {river_mask_path}")
    
    # Land cover filter (requires threshold)
    if landcover_path.exists():
        if verbose:
            print(f"\n--- Loading land cover (valid classes: {valid_classes}) ---")
        with rasterio.open(landcover_path) as src:
            lc_data = src.read(1)
            lc_transform = src.transform
            lc_crs = src.crs
        
        binary_mask = np.zeros_like(lc_data, dtype=np.uint8)
        for cls in valid_classes:
            binary_mask[lc_data == cls] = 1
        
        lc_mask_reprojected = np.zeros((height, width), dtype=np.uint8)
        reproject(source=binary_mask, destination=lc_mask_reprojected,
                  src_transform=lc_transform, src_crs=lc_crs,
                  dst_transform=target_transform, dst_crs="EPSG:4326",
                  resampling=Resampling.max)
        
        lc_mask = lc_mask_reprojected > 0
        if verbose:
            print(f"  Cells with valid land cover: {np.sum(lc_mask):,}")
    elif verbose:
        print(f"  [WARNING] Land cover file not found: {landcover_path}")
    
    # Resource threshold mask
    resource_valid = data >= resource_threshold
    if verbose:
        print(f"\n--- Applying viability filter (threshold >= {resource_threshold} mm/year) ---")
        print(f"  Cells meeting resource threshold: {np.sum(resource_valid & (data > 0)):,}")
    
    # Combined logic: river_proximity OR (landcover_valid AND resource >= threshold)
    combined_mask = river_mask | (lc_mask & resource_valid)
    
    n_river_only = np.sum(river_mask & ~(lc_mask & resource_valid))
    n_lc_resource = np.sum(~river_mask & lc_mask & resource_valid)
    n_both = np.sum(river_mask & lc_mask & resource_valid)
    n_total = np.sum(combined_mask)
    
    if verbose:
        print(f"\n--- Combined filter (OR logic) ---")
        print(f"  Viable by river proximity only: {n_river_only:,}")
        print(f"  Viable by land cover + threshold: {n_lc_resource:,}")
        print(f"  Viable by both criteria: {n_both:,}")
        print(f"  Total viable cells: {n_total:,}")
    
    data_filtered = data.copy()
    data_filtered[~combined_mask] = nodata_value
    return data_filtered


# =============================================================================
# DELTA EXTRACTION AT POINTS
# =============================================================================

def extract_delta_at_points(gdf: gpd.GeoDataFrame, delta_data: np.ndarray,
                            delta_transform, delta_column: str = 'delta') -> gpd.GeoDataFrame:
    """Extract delta values at river reach centroids."""
    print(f"  Extracting delta values at {len(gdf):,} reaches...")
    
    if gdf.geometry.iloc[0].geom_type in ('LineString', 'MultiLineString'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*geographic CRS.*centroid.*')
            centroids = gdf.geometry.centroid
    else:
        centroids = gdf.geometry
    
    coords = get_coordinates(centroids.values)
    coords_x, coords_y = coords[:, 0], coords[:, 1]
    
    cols = ((coords_x - delta_transform.c) / delta_transform.a).astype(int)
    rows = ((coords_y - delta_transform.f) / delta_transform.e).astype(int)
    
    rows_clipped = np.clip(rows, 0, delta_data.shape[0] - 1)
    cols_clipped = np.clip(cols, 0, delta_data.shape[1] - 1)
    
    delta_values = delta_data[rows_clipped, cols_clipped]
    
    out_of_bounds = (rows < 0) | (rows >= delta_data.shape[0]) | \
                    (cols < 0) | (cols >= delta_data.shape[1])
    invalid = np.isnan(delta_values) | (delta_values <= 0)
    nodata_mask = out_of_bounds | invalid
    
    delta_values = np.where(nodata_mask, 1.0, delta_values)
    
    gdf = gdf.copy()
    gdf[delta_column] = delta_values
    
    print(f"    Extracted, {nodata_mask.sum():,} with no data (using delta=1.0)")
    return gdf
