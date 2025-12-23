"""
p1_c_prep_landcover.py
======================
Download ESA CCI Land Cover 2022 from Copernicus Climate Data Store (CDS)

This script downloads the satellite land cover classification data (v2.1.1)
which is used to filter viable areas for solar, wind, and hydro energy siting.

Output:
- bigdata_landcover_cds/downloads/ (raw ZIP from CDS)
- bigdata_landcover_cds/extracted/C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.nc (NetCDF)
- bigdata_landcover_cds/outputs/landcover_2022_10arcsec.tif (native resolution GeoTIFF)
- bigdata_landcover_cds/outputs/landcover_2022_300arcsec.tif (upscaled, aligned with GHS-POP grid)

Resolution:
- Native: ~300m (approx 10 arcsec at equator)
- Upscaled: 300 arcsec (~9km), aligned with GHS-POP/CMIP6 output grid

Land Cover Classes (ESA CCI LCCS):
- 10: Cropland, rainfed
- 20: Cropland, irrigated or post-flooding
- 30: Mosaic cropland (>50%) / natural vegetation (<50%)
- 40: Mosaic natural vegetation (>50%) / cropland (<50%)
- 50: Tree cover, broadleaved, evergreen
- 60: Tree cover, broadleaved, deciduous
- 70: Tree cover, needleleaved, evergreen
- 80: Tree cover, needleleaved, deciduous
- 90: Tree cover, mixed leaf type
- 100: Mosaic tree and shrub (>50%) / herbaceous (<50%)
- 110: Mosaic herbaceous (>50%) / tree and shrub (<50%)
- 120: Shrubland
- 130: Grassland
- 140: Lichens and mosses
- 150: Sparse vegetation
- 160: Tree cover, flooded, fresh or brackish water
- 170: Tree cover, flooded, saline water
- 180: Shrub or herbaceous cover, flooded
- 190: Urban areas
- 200: Bare areas
- 210: Water bodies
- 220: Permanent snow and ice

Usage:
    python p1_c_cds_landcover.py [--force]
    
Arguments:
    --force     Re-download even if files exist

Prerequisites:
    - CDS API key configured (~/.cdsapirc or environment variables)
    - cdsapi package installed
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.stats import mode

import cdsapi

from config import TARGET_RESOLUTION_ARCSEC, GHS_POP_NATIVE_RESOLUTION_ARCSEC, POP_AGGREGATION_FACTOR


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_bigdata_path(folder_name: str) -> str:
    """Get the correct path for bigdata folders."""
    local_path = folder_name
    cluster_path = f"/soge-home/projects/mistral/ji/{folder_name}"
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(cluster_path):
        return cluster_path
    else:
        return local_path


# Output directories
LANDCOVER_DIR = Path(get_bigdata_path("bigdata_landcover_cds"))
DOWNLOADS_DIR = LANDCOVER_DIR / "downloads"
EXTRACTED_DIR = LANDCOVER_DIR / "extracted"
OUTPUTS_DIR = LANDCOVER_DIR / "outputs"


# =============================================================================
# DOWNLOAD FUNCTION
# =============================================================================

def download_landcover_2022(downloads_dir: Path, extracted_dir: Path, force: bool = False) -> Path:
    """
    Download ESA CCI Land Cover 2022 from CDS.
    
    Args:
        downloads_dir: Directory for raw ZIP downloads
        extracted_dir: Directory for extracted NetCDF files
        force: Re-download even if file exists
    
    Returns:
        Path to extracted NetCDF file
    """
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    # Final extracted NetCDF path
    nc_file = extracted_dir / "C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.nc"
    
    if nc_file.exists() and not force:
        print(f"[skip] Land cover already downloaded: {nc_file}")
        return nc_file
    
    print("[download] ESA CCI Land Cover 2022 (v2.1.1)...")
    print("  This may take several minutes (~2GB file)...")
    
    client = cdsapi.Client()
    
    dataset = "satellite-land-cover"
    request = {
        "variable": "all",
        "year": ["2022"],
        "version": ["v2_1_1"]
    }
    
    # Store current directory to restore later
    original_cwd = os.getcwd()
    
    try:
        # Change to downloads directory for download
        os.chdir(downloads_dir)
        
        # Use recommended CDS API pattern: retrieve then download
        # CDS returns a ZIP file, let it choose the filename
        result = client.retrieve(dataset, request)
        downloaded_file = Path(result.download())
        print(f"[done] Downloaded: {downloaded_file}")
        
        # Restore working directory
        os.chdir(original_cwd)
        
        # Full path to downloaded ZIP
        downloaded_path = downloads_dir / downloaded_file.name
        
        # Extract ZIP to get NetCDF
        nc_path = extract_zip_to_netcdf(downloaded_path, extracted_dir, nc_file)
        
        return nc_path
    except Exception as e:
        os.chdir(original_cwd)
        print(f"[ERROR] Failed to download land cover: {e}")
        raise


def extract_zip_to_netcdf(zip_path: Path, out_dir: Path, target_nc: Path) -> Path:
    """
    Extract NetCDF file from CDS ZIP archive.
    
    Args:
        zip_path: Path to ZIP file
        out_dir: Output directory
        target_nc: Target NetCDF filename
    
    Returns:
        Path to extracted NetCDF file
    """
    print(f"[extract] Extracting ZIP archive...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the NetCDF file in the archive
        nc_files = [n for n in zf.namelist() if n.endswith('.nc')]
        if not nc_files:
            raise RuntimeError(f"No NetCDF files found in ZIP archive: {zf.namelist()}")
        
        nc_name = nc_files[0]
        print(f"  Found: {nc_name}")
        zf.extract(nc_name, out_dir)
    
    extracted_path = out_dir / nc_name
    
    # Rename to target name if different
    if extracted_path != target_nc:
        if target_nc.exists():
            target_nc.unlink()
        extracted_path.rename(target_nc)
    
    # Remove ZIP file
    zip_path.unlink()
    print(f"  Extracted to: {target_nc}")
    
    return target_nc


def extract_if_zipped(file_path: Path) -> Path:
    """
    Check if file is a ZIP archive and extract if so.
    
    Args:
        file_path: Path to file to check
    
    Returns:
        Path to extracted NetCDF file (or original if not zipped)
    """
    # Check magic bytes
    with open(file_path, 'rb') as f:
        magic = f.read(4)
    
    if magic[:2] != b'PK':  # Not a ZIP file
        return file_path
    
    print(f"[extract] Downloaded file is a ZIP archive, extracting...")
    
    extract_dir = file_path.parent
    
    with zipfile.ZipFile(file_path, 'r') as zf:
        # Find the NetCDF file in the archive
        nc_files = [n for n in zf.namelist() if n.endswith('.nc')]
        if not nc_files:
            raise RuntimeError(f"No NetCDF files found in ZIP archive: {zf.namelist()}")
        
        nc_name = nc_files[0]
        zf.extract(nc_name, extract_dir)
        print(f"  Extracted: {nc_name}")
    
    # Remove the ZIP file and rename extracted file
    extracted_path = extract_dir / nc_name
    
    # If extracted to same name as ZIP, we're done
    if extracted_path == file_path:
        return file_path
    
    # Remove ZIP and rename extracted file to expected name
    file_path.unlink()
    extracted_path.rename(file_path)
    
    return file_path


def convert_to_geotiff(nc_path: Path, outputs_dir: Path, force: bool = False) -> Path:
    """
    Convert NetCDF land cover to GeoTIFF for rasterio compatibility.
    
    Args:
        nc_path: Path to NetCDF file
        outputs_dir: Output directory for GeoTIFF
        force: Re-convert even if file exists
    
    Returns:
        Path to GeoTIFF file
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tif_path = outputs_dir / "landcover_2022_10arcsec.tif"
    
    if tif_path.exists() and not force:
        print(f"[skip] GeoTIFF already exists: {tif_path}")
        return tif_path
    
    # Check if file is still zipped and extract
    nc_path = extract_if_zipped(nc_path)
    
    print("[convert] NetCDF to GeoTIFF...")
    
    # Open NetCDF - try different engines
    ds = None
    for engine in ['netcdf4', 'scipy', 'h5netcdf']:
        try:
            ds = xr.open_dataset(nc_path, engine=engine)
            print(f"  Opened with engine: {engine}")
            break
        except Exception:
            continue
    
    if ds is None:
        raise RuntimeError(f"Cannot open {nc_path} with any available engine")
    
    # Find the land cover classification variable
    # Usually named 'lccs_class' or similar
    lc_var = None
    for var in ds.data_vars:
        if 'lccs' in var.lower() or 'class' in var.lower() or 'lc' in var.lower():
            lc_var = var
            break
    
    if lc_var is None:
        # Use first non-coordinate variable
        data_vars = [v for v in ds.data_vars if v not in ['crs', 'time_bnds']]
        if data_vars:
            lc_var = data_vars[0]
        else:
            raise RuntimeError(f"Cannot find land cover variable in {nc_path}")
    
    print(f"  Using variable: {lc_var}")
    
    da = ds[lc_var]
    
    # Handle time dimension if present (select first/only time)
    if 'time' in da.dims:
        da = da.isel(time=0)
    
    # Get coordinate arrays
    # ESA CCI uses 'lat' and 'lon' coordinates
    if 'lat' in da.coords and 'lon' in da.coords:
        lats = da.lat.values
        lons = da.lon.values
    elif 'latitude' in da.coords and 'longitude' in da.coords:
        lats = da.latitude.values
        lons = da.longitude.values
    else:
        raise RuntimeError(f"Cannot find lat/lon coordinates. Available: {list(da.coords)}")
    
    data = da.values
    
    # Ensure data is 2D
    if data.ndim > 2:
        data = data.squeeze()
    
    print(f"  Data shape: {data.shape}")
    print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")
    
    # Create transform (assuming regular grid, lats descending)
    height, width = data.shape
    
    # Calculate resolution
    lon_res = (lons.max() - lons.min()) / (width - 1) if width > 1 else 1
    lat_res = (lats.max() - lats.min()) / (height - 1) if height > 1 else 1
    
    # Transform from upper-left corner
    west = lons.min() - lon_res / 2
    north = lats.max() + lat_res / 2
    
    transform = rasterio.transform.from_origin(west, north, lon_res, lat_res)
    
    # Write GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw',
        'nodata': 0,  # 0 is typically nodata/ocean in ESA CCI
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
    }
    
    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data, 1)
    
    print(f"[done] GeoTIFF created: {tif_path}")
    print(f"  Resolution: {lon_res * 3600:.1f} arcsec x {lat_res * 3600:.1f} arcsec")
    
    # Print class statistics
    unique_classes, counts = np.unique(data[data > 0], return_counts=True)
    print(f"  Classes present: {len(unique_classes)}")
    
    ds.close()
    
    return tif_path


# =============================================================================
# GHS-POP GRID ALIGNMENT
# =============================================================================

# Cache for grid params
_GHS_POP_GRID_PARAMS = None


def get_ghs_pop_grid_params() -> dict:
    """
    Get GHS-POP raster grid parameters (origin and pixel size) dynamically.
    This ensures land cover outputs align with CMIP6/settlement centroids.
    
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


def upscale_landcover_to_300arcsec(input_tif: Path, output_dir: Path, force: bool = False) -> Path:
    """
    Upscale land cover from 10 arcsec to 300 arcsec, aligned with GHS-POP grid.
    
    Uses majority (mode) resampling to preserve the most common land cover class
    within each output cell.
    
    Args:
        input_tif: Path to 10 arcsec land cover GeoTIFF
        output_dir: Output directory
        force: Re-create even if file exists
    
    Returns:
        Path to 300 arcsec GeoTIFF
    """
    output_tif = output_dir / "landcover_2022_300arcsec.tif"
    
    if output_tif.exists() and not force:
        print(f"[skip] 300 arcsec GeoTIFF already exists: {output_tif}")
        return output_tif
    
    print("[upscale] Resampling to 300 arcsec (aligned with GHS-POP grid)...")
    
    # Get GHS-POP grid parameters
    ghs_params = get_ghs_pop_grid_params()
    agg_factor = POP_AGGREGATION_FACTOR  # 10 -> 300 arcsec
    
    # Aggregated pixel size
    agg_pixel_lon = ghs_params['pixel_size_lon'] * agg_factor  # 300 arcsec
    agg_pixel_lat = ghs_params['pixel_size_lat'] * agg_factor  # -300 arcsec
    
    # Target grid dimensions (global)
    n_cols = int(np.ceil(360 / abs(agg_pixel_lon)))
    n_rows = int(np.ceil(180 / abs(agg_pixel_lat)))
    
    # Target transform matching GHS-POP aggregated grid
    target_transform = rasterio.transform.from_origin(
        ghs_params['origin_lon'],
        ghs_params['origin_lat'],
        abs(agg_pixel_lon),
        abs(agg_pixel_lat)
    )
    
    with rasterio.open(input_tif) as src:
        src_data = src.read(1)
        
        # Create output array
        target_data = np.zeros((n_rows, n_cols), dtype=src_data.dtype)
        
        # Reproject using mode (majority) for categorical data
        reproject(
            source=src_data,
            destination=target_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.mode,  # Majority class
        )
    
    # Write output
    profile = {
        'driver': 'GTiff',
        'dtype': target_data.dtype,
        'width': n_cols,
        'height': n_rows,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': target_transform,
        'compress': 'lzw',
        'nodata': 0,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(target_data, 1)
    
    print(f"[done] 300 arcsec GeoTIFF created: {output_tif}")
    print(f"  Resolution: {TARGET_RESOLUTION_ARCSEC} arcsec")
    print(f"  Grid size: {n_cols} x {n_rows}")
    print(f"  Aligned with GHS-POP aggregated grid")
    
    # Print class statistics
    unique_classes = np.unique(target_data[target_data > 0])
    print(f"  Classes present: {len(unique_classes)}")
    
    return output_tif


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download ESA CCI Land Cover 2022")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ESA CCI Land Cover 2022 Download & Processing")
    print("=" * 60)
    
    # Download and extract NetCDF
    nc_path = download_landcover_2022(DOWNLOADS_DIR, EXTRACTED_DIR, force=args.force)
    
    # Convert to native resolution GeoTIFF (10 arcsec)
    tif_10arcsec = convert_to_geotiff(nc_path, OUTPUTS_DIR, force=args.force)
    
    # Upscale to 300 arcsec (aligned with GHS-POP/CMIP6 grid)
    tif_300arcsec = upscale_landcover_to_300arcsec(tif_10arcsec, OUTPUTS_DIR, force=args.force)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"  Downloads:  {DOWNLOADS_DIR}")
    print(f"  Extracted:  {nc_path}")
    print(f"  Native:     {tif_10arcsec}")
    print(f"  Upscaled:   {tif_300arcsec}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
