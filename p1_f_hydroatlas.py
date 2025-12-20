"""
p1_f_hydroatlas.py
==================
Apply CMIP6 Delta Method to HydroATLAS River Reaches

This script applies the CMIP6 runoff change projections (computed in p1_e_cmip6_hydro.py)
to RiverATLAS discharge data to estimate future hydropower potential at river reach level.

Methodology:
1. Load RiverATLAS river reaches with discharge (dis_m3_pyr) and runoff (run_mm_cyr)
2. Load CMIP6 delta grids (from p1_e_cmip6_hydro.py outputs)
3. Extract delta values at each river reach centroid
4. Apply delta: dis_m3_pyr_future = dis_m3_pyr × delta
5. Compute hydropower potential indicator (optional)
6. Save projected river reaches as Parquet/GeoPackage

Note on baseline periods:
- HydroATLAS discharge (dis_m3_pyr) is based on long-term averages (~1971-2000 from WaterGAP)
- CMIP6 delta uses 1995-2014 as historical baseline
- This temporal mismatch is acceptable for delta method since we apply *relative* changes
- The delta captures climate change signal direction and magnitude, not absolute values

Output:
- RiverATLAS_projected_2030.parquet
- RiverATLAS_projected_2050.parquet
- RiverATLAS_projected.gpkg (optional, for visualization)

Dependencies:
- Requires p1_e_cmip6_hydro.py to be run first (needs HYDRO_ATLAS_DELTA_*.tif)
- Requires RiverATLAS_Data_v10.gdb/RiverATLAS_v10.gdb in bigdata_hydro_atlas/
"""

import os
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
from shapely import get_coordinates

warnings.filterwarnings('ignore', 'CRS mismatch')


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


# =============================================================================
# SETTINGS
# =============================================================================

from config import TARGET_RESOLUTION_ARCSEC

# RiverATLAS columns to keep for analysis
RIVERATLAS_COLUMNS = [
    # Identifiers
    'HYRIV_ID',      # Unique river reach ID
    'NEXT_DOWN',     # Downstream reach ID
    'MAIN_RIV',      # Main river flag
    'LENGTH_KM',     # Reach length in km
    'ORD_STRA',      # Stream order (Strahler)
    'ORD_CLAS',      # Stream order class
    'ORD_FLOW',      # Flow order
    'HYBAS_L12',     # HydroBASINS level 12 ID
    
    # Hydrology (key variables)
    # dis_m3_p* = discharge at pour point in m³/s
    # Suffix: yr=annual avg, mn=annual min, mx=annual max
    'dis_m3_pyr',    # Mean annual discharge at pour point (m³/s)
    'dis_m3_pmn',    # Minimum monthly discharge at pour point (m³/s)
    'dis_m3_pmx',    # Maximum monthly discharge at pour point (m³/s)
    'run_mm_cyr',    # Land surface runoff (mm/year) - for delta matching
    
    # Physiography (for head estimation)
    'ele_mt_cav',    # Elevation at reach centroid (m)
    'ele_mt_uav',    # Mean upstream elevation (m)
    'ele_mt_cmn',    # Minimum elevation in reach
    'ele_mt_cmx',    # Maximum elevation in reach
    'slp_dg_cav',    # Slope at reach (degrees)
    'slp_dg_uav',    # Mean upstream slope
    'sgr_dk_rav',    # Stream gradient (‰)
    
    # Area
    'UPLAND_SKM',    # Upstream drainage area (km²)
    
    # Climate (for context)
    'tmp_dc_cyr',    # Mean annual temperature (°C × 10)
    'pre_mm_cyr',    # Mean annual precipitation (mm)
    
    # Regulation (existing dams)
    'dor_pc_pva',    # Degree of regulation (%)
    'rev_mc_usu',    # Reservoir volume upstream (million m³)
]

# Minimum discharge threshold for hydropower relevance (m³/s)
# Note: RiverATLAS dis_m3_pyr is actually in m³/s (same as DIS_AV_CMS), not m³/year
# 0.1 m³/s is a reasonable minimum for small hydro
MIN_DISCHARGE_M3S = 0.1

# Minimum stream order for filtering (optional)
MIN_STREAM_ORDER = 2


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_riveratlas(gdb_path: Path, 
                    columns: List[str] = None,
                    min_discharge: float = None,
                    min_order: int = None,
                    bbox: tuple = None) -> gpd.GeoDataFrame:
    """
    Load RiverATLAS data from FileGDB.
    
    Args:
        gdb_path: Path to RiverATLAS_Data_v10.gdb
        columns: List of columns to load (None = all)
        min_discharge: Minimum discharge threshold (m³/year)
        min_order: Minimum stream order
        bbox: Bounding box (minx, miny, maxx, maxy) for spatial filter
    
    Returns:
        GeoDataFrame with river reaches
    """
    print(f"Loading RiverATLAS from {gdb_path}...")
    
    # List layers in GDB
    import fiona
    layers = fiona.listlayers(gdb_path)
    print(f"  Available layers: {layers}")
    
    # Find the river reach layer (usually 'RiverATLAS_v10' or similar)
    river_layer = None
    for layer in layers:
        if 'river' in layer.lower():
            river_layer = layer
            break
    
    if river_layer is None:
        river_layer = layers[0]  # Use first layer as fallback
    
    print(f"  Using layer: {river_layer}")
    
    # Load data
    if bbox:
        gdf = gpd.read_file(gdb_path, layer=river_layer, bbox=bbox)
    else:
        gdf = gpd.read_file(gdb_path, layer=river_layer)
    
    print(f"  Loaded {len(gdf):,} river reaches")
    print(f"  Columns: {list(gdf.columns)[:20]}...")  # Show first 20
    
    # Filter by discharge if specified
    if min_discharge and 'dis_m3_pyr' in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf['dis_m3_pyr'] >= min_discharge]
        print(f"  Filtered by discharge >= {min_discharge:,.0f} m³/year: {before:,} → {len(gdf):,}")
    
    # Filter by stream order if specified
    if min_order and 'ORD_STRA' in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf['ORD_STRA'] >= min_order]
        print(f"  Filtered by stream order >= {min_order}: {before:,} → {len(gdf):,}")
    
    # Select specific columns if requested
    if columns:
        available_cols = ['geometry'] + [c for c in columns if c in gdf.columns]
        missing = [c for c in columns if c not in gdf.columns]
        if missing:
            print(f"  Warning: Missing columns: {missing}")
        gdf = gdf[available_cols]
    
    return gdf


def load_delta_raster(tif_path: Path) -> tuple:
    """
    Load CMIP6 delta raster.
    
    Returns:
        (data, transform, crs)
    """
    print(f"  Loading delta raster: {tif_path.name}")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    
    print(f"    Shape: {data.shape}, Range: [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
    
    return data, transform, crs


def extract_delta_at_points(gdf: gpd.GeoDataFrame, 
                            delta_data: np.ndarray,
                            delta_transform,
                            delta_column: str = 'delta') -> gpd.GeoDataFrame:
    """
    Extract delta values at river reach centroids.
    
    Args:
        gdf: GeoDataFrame with river reaches (LineString or Point geometry)
        delta_data: 2D numpy array with delta values
        delta_transform: Rasterio transform for the delta raster
        delta_column: Name for the output delta column
    
    Returns:
        GeoDataFrame with delta column added
    """
    import warnings
    print(f"  Extracting delta values at {len(gdf):,} reaches...")
    
    # Get centroids for LineString geometries
    # Note: Using geographic CRS centroids is acceptable here since we're just
    # extracting raster values at ~10km resolution - minor inaccuracy is negligible
    if gdf.geometry.iloc[0].geom_type in ('LineString', 'MultiLineString'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*geographic CRS.*centroid.*')
            centroids = gdf.geometry.centroid
    else:
        centroids = gdf.geometry
    
    # Vectorized extraction using shapely.get_coordinates
    coords = get_coordinates(centroids.values)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    
    # Convert coordinates to row/col indices (vectorized)
    # transform: (a, b, c, d, e, f) where x = c + col*a, y = f + row*e
    # So: col = (x - c) / a, row = (y - f) / e
    a, b, c, d, e, f = delta_transform.to_gdal()  # c, a, b, f, d, e in GDAL order
    # Rasterio transform: (a, b, c, d, e, f) = (scale_x, 0, origin_x, 0, scale_y, origin_y)
    cols = ((coords_x - delta_transform.c) / delta_transform.a).astype(int)
    rows = ((coords_y - delta_transform.f) / delta_transform.e).astype(int)
    
    # Clip to valid bounds
    rows_clipped = np.clip(rows, 0, delta_data.shape[0] - 1)
    cols_clipped = np.clip(cols, 0, delta_data.shape[1] - 1)
    
    # Extract values (vectorized)
    delta_values = delta_data[rows_clipped, cols_clipped]
    
    # Mark out-of-bounds and invalid values as 1.0 (no change)
    out_of_bounds = (rows < 0) | (rows >= delta_data.shape[0]) | \
                    (cols < 0) | (cols >= delta_data.shape[1])
    invalid = np.isnan(delta_values) | (delta_values <= 0)
    nodata_mask = out_of_bounds | invalid
    
    delta_values = np.where(nodata_mask, 1.0, delta_values)
    nodata_count = nodata_mask.sum()
    
    gdf = gdf.copy()
    gdf[delta_column] = delta_values
    
    print(f"    Extracted values, {nodata_count:,} reaches with no data (using delta=1.0)")
    print(f"    Delta range: [{np.nanmin(delta_values):.3f}, {np.nanmax(delta_values):.3f}]")
    
    return gdf


# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================

def compute_projected_discharge(gdf: gpd.GeoDataFrame,
                                 delta_2030_col: str = 'delta_2030',
                                 delta_2050_col: str = 'delta_2050',
                                 baseline_col: str = 'dis_m3_pyr') -> gpd.GeoDataFrame:
    """
    Compute projected discharge for 2030 and 2050.
    
    discharge_future = discharge_baseline × delta
    
    Note: HydroATLAS baseline is ~1971-2000, while CMIP6 delta uses 1995-2014.
    This is acceptable since delta captures *relative* climate change signal.
    """
    gdf = gdf.copy()
    
    # Projected discharge
    gdf['dis_m3_pyr_2030'] = gdf[baseline_col] * gdf[delta_2030_col]
    gdf['dis_m3_pyr_2050'] = gdf[baseline_col] * gdf[delta_2050_col]
    
    # Percent change
    gdf['dis_change_pct_2030'] = (gdf[delta_2030_col] - 1.0) * 100
    gdf['dis_change_pct_2050'] = (gdf[delta_2050_col] - 1.0) * 100
    
    print(f"\n  Projected discharge statistics:")
    print(f"    Baseline mean: {gdf[baseline_col].mean():,.0f} m³/year")
    print(f"    2030 mean: {gdf['dis_m3_pyr_2030'].mean():,.0f} m³/year ({gdf['dis_change_pct_2030'].mean():+.1f}%)")
    print(f"    2050 mean: {gdf['dis_m3_pyr_2050'].mean():,.0f} m³/year ({gdf['dis_change_pct_2050'].mean():+.1f}%)")
    
    return gdf


def compute_hydro_potential_indicator(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute simple hydropower potential indicator.
    
    Hydropower potential is proportional to: Discharge × Head
    
    We use stream gradient (sgr_dk_rav) as a proxy for available head.
    This is a simplified indicator, not actual power potential.
    
    P_indicator = discharge (m³/s) × gradient (‰) × constant
    """
    gdf = gdf.copy()
    
    # Convert annual discharge to m³/s
    seconds_per_year = 365.25 * 24 * 3600
    
    if 'dis_m3_pyr' in gdf.columns and 'sgr_dk_rav' in gdf.columns:
        # Baseline potential indicator
        discharge_m3s = gdf['dis_m3_pyr'] / seconds_per_year
        gradient = gdf['sgr_dk_rav'].clip(lower=0.1)  # ‰ (per mille)
        
        # Simple indicator (arbitrary units, for relative comparison)
        gdf['hydro_potential_baseline'] = discharge_m3s * gradient
        
        # 2030 potential
        if 'dis_m3_pyr_2030' in gdf.columns:
            discharge_2030_m3s = gdf['dis_m3_pyr_2030'] / seconds_per_year
            gdf['hydro_potential_2030'] = discharge_2030_m3s * gradient
        
        # 2050 potential
        if 'dis_m3_pyr_2050' in gdf.columns:
            discharge_2050_m3s = gdf['dis_m3_pyr_2050'] / seconds_per_year
            gdf['hydro_potential_2050'] = discharge_2050_m3s * gradient
        
        print(f"\n  Hydropower potential indicator computed")
        print(f"    (P ∝ Q × gradient, arbitrary units for relative comparison)")
    else:
        print(f"\n  [warning] Cannot compute hydro potential: missing dis_m3_pyr or sgr_dk_rav")
    
    return gdf


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_outputs(gdf: gpd.GeoDataFrame, 
                 out_dir: Path,
                 save_gpkg: bool = False) -> None:
    """
    Save projected RiverATLAS data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet (efficient for large datasets)
    parquet_path = out_dir / "RiverATLAS_projected.parquet"
    gdf.to_parquet(parquet_path)
    print(f"\n  Saved: {parquet_path} ({len(gdf):,} reaches)")
    
    # Save separate files for each year (smaller, focused)
    cols_2030 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2030', 
                 'dis_m3_pyr_2030', 'dis_change_pct_2030']
    cols_2050 = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'delta_2050',
                 'dis_m3_pyr_2050', 'dis_change_pct_2050']
    
    # Add optional columns if present
    for col in ['hydro_potential_baseline', 'hydro_potential_2030', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']:
        if col in gdf.columns:
            if '2030' in col or col in ['hydro_potential_baseline', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']:
                cols_2030.append(col)
            if '2050' in col or col in ['hydro_potential_baseline', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav']:
                cols_2050.append(col)
    
    cols_2030 = [c for c in cols_2030 if c in gdf.columns]
    cols_2050 = [c for c in cols_2050 if c in gdf.columns]
    
    gdf[cols_2030].to_parquet(out_dir / "RiverATLAS_projected_2030.parquet")
    gdf[cols_2050].to_parquet(out_dir / "RiverATLAS_projected_2050.parquet")
    print(f"  Saved: RiverATLAS_projected_2030.parquet")
    print(f"  Saved: RiverATLAS_projected_2050.parquet")
    
    # Save baseline file (original RiverATLAS data without projections)
    cols_baseline = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx',
                     'run_mm_cyr', 'ORD_STRA', 'UPLAND_SKM', 'sgr_dk_rav', 
                     'ele_mt_cav', 'ele_mt_uav', 'LENGTH_KM']
    cols_baseline = [c for c in cols_baseline if c in gdf.columns]
    if 'hydro_potential_baseline' in gdf.columns:
        cols_baseline.append('hydro_potential_baseline')
    gdf[cols_baseline].to_parquet(out_dir / "RiverATLAS_baseline.parquet")
    print(f"  Saved: RiverATLAS_baseline.parquet")
    
    # Save as GeoPackage for visualization (optional, can be large)
    if save_gpkg:
        gpkg_path = out_dir / "RiverATLAS_projected.gpkg"
        # Select key columns for GPKG to reduce size
        gpkg_cols = ['HYRIV_ID', 'geometry', 'dis_m3_pyr', 
                     'delta_2030', 'delta_2050',
                     'dis_m3_pyr_2030', 'dis_m3_pyr_2050',
                     'dis_change_pct_2030', 'dis_change_pct_2050']
        gpkg_cols = [c for c in gpkg_cols if c in gdf.columns]
        gdf[gpkg_cols].to_file(gpkg_path, driver="GPKG")
        print(f"  Saved: {gpkg_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(riveratlas_path: str = None,
        delta_dir: str = None,
        out_dir: str = None,
        min_discharge: float = None,
        min_order: int = None,
        bbox: tuple = None,
        save_gpkg: bool = False):
    """
    Main entry point.
    
    Args:
        riveratlas_path: Path to RiverATLAS_Data_v10.gdb
        delta_dir: Directory containing HYDRO_DELTA_*.tif from p1_e_cmip6_hydro.py
        out_dir: Output directory
        min_discharge: Filter reaches by minimum discharge (m³/year)
        min_order: Filter reaches by minimum stream order
        bbox: Bounding box (minx, miny, maxx, maxy) for spatial filter
        save_gpkg: Whether to save GeoPackage output (large file)
    """
    # Default paths
    if riveratlas_path is None:
        riveratlas_path = Path(get_bigdata_path("bigdata_hydro_atlas")) / "RiverATLAS_Data_v10.gdb" / "RiverATLAS_v10.gdb"
    else:
        riveratlas_path = Path(riveratlas_path)
    
    if delta_dir is None:
        delta_dir = Path(get_bigdata_path("bigdata_hydro_cmip6")) / "outputs"
    else:
        delta_dir = Path(delta_dir)
    
    if out_dir is None:
        out_dir = Path(get_bigdata_path("bigdata_hydro_atlas")) / "outputs"
    else:
        out_dir = Path(out_dir)
    
    # Use defaults if not specified
    if min_discharge is None:
        min_discharge = MIN_DISCHARGE_M3S
    if min_order is None:
        min_order = MIN_STREAM_ORDER
    
    print("="*70)
    print("HydroATLAS + CMIP6 Delta Projection")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Check for delta rasters
    # -------------------------------------------------------------------------
    suffix = f"{TARGET_RESOLUTION_ARCSEC}arcsec"
    delta_2030_path = delta_dir / f"HYDRO_ATLAS_DELTA_2030_{suffix}.tif"
    delta_2050_path = delta_dir / f"HYDRO_ATLAS_DELTA_2050_{suffix}.tif"
    
    if not delta_2030_path.exists() or not delta_2050_path.exists():
        print("\n[ERROR] Delta rasters not found!")
        print(f"  Expected: {delta_2030_path}")
        print(f"  Expected: {delta_2050_path}")
        print("\n  Please run p1_e_cmip6_hydro.py first to generate delta grids.")
        return
    
    # -------------------------------------------------------------------------
    # Load RiverATLAS
    # -------------------------------------------------------------------------
    print("\n--- Loading RiverATLAS ---")
    
    if not riveratlas_path.exists():
        print(f"\n[ERROR] RiverATLAS not found: {riveratlas_path}")
        return
    
    gdf = load_riveratlas(
        riveratlas_path,
        columns=RIVERATLAS_COLUMNS,
        min_discharge=min_discharge,
        min_order=min_order,
        bbox=bbox
    )
    
    if len(gdf) == 0:
        print("\n[ERROR] No river reaches loaded after filtering!")
        return
    
    # -------------------------------------------------------------------------
    # Load and apply delta values
    # -------------------------------------------------------------------------
    print("\n--- Loading CMIP6 delta rasters ---")
    
    delta_2030_data, delta_2030_transform, _ = load_delta_raster(delta_2030_path)
    delta_2050_data, delta_2050_transform, _ = load_delta_raster(delta_2050_path)
    
    print("\n--- Extracting delta at river reaches ---")
    
    gdf = extract_delta_at_points(gdf, delta_2030_data, delta_2030_transform, 'delta_2030')
    gdf = extract_delta_at_points(gdf, delta_2050_data, delta_2050_transform, 'delta_2050')
    
    # -------------------------------------------------------------------------
    # Compute projections
    # -------------------------------------------------------------------------
    print("\n--- Computing projected discharge ---")
    
    gdf = compute_projected_discharge(gdf)
    gdf = compute_hydro_potential_indicator(gdf)
    
    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n--- Summary Statistics ---")
    
    # By change direction
    increasing_2030 = (gdf['delta_2030'] > 1.0).sum()
    decreasing_2030 = (gdf['delta_2030'] < 1.0).sum()
    increasing_2050 = (gdf['delta_2050'] > 1.0).sum()
    decreasing_2050 = (gdf['delta_2050'] < 1.0).sum()
    
    print(f"\n  2030 projections:")
    print(f"    Increasing discharge: {increasing_2030:,} reaches ({100*increasing_2030/len(gdf):.1f}%)")
    print(f"    Decreasing discharge: {decreasing_2030:,} reaches ({100*decreasing_2030/len(gdf):.1f}%)")
    
    print(f"\n  2050 projections:")
    print(f"    Increasing discharge: {increasing_2050:,} reaches ({100*increasing_2050/len(gdf):.1f}%)")
    print(f"    Decreasing discharge: {decreasing_2050:,} reaches ({100*decreasing_2050/len(gdf):.1f}%)")
    
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs ---")
    
    save_outputs(gdf, out_dir, save_gpkg=save_gpkg)
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply CMIP6 Delta to HydroATLAS River Reaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (global, min discharge 3M m³/year, min order 2)
  python p1_f_hydroatlas.py
  
  # Filter to larger rivers only
  python p1_f_hydroatlas.py --min-discharge 10000000 --min-order 4
  
  # Process specific region (bbox: minx, miny, maxx, maxy)
  python p1_f_hydroatlas.py --bbox -20 -40 55 40  # Africa
  
  # Save GeoPackage for visualization
  python p1_f_hydroatlas.py --save-gpkg

Prerequisites:
  - Run p1_e_cmip6_hydro.py first to generate HYDRO_ATLAS_DELTA_*.tif
  - Download RiverATLAS_Data_v10.gdb to bigdata_hydro_atlas/
        """
    )
    
    parser.add_argument(
        "--riveratlas",
        default=None,
        help="Path to RiverATLAS_Data_v10.gdb/RiverATLAS_v10.gdb"
    )
    parser.add_argument(
        "--delta-dir",
        default=None,
        help="Directory containing HYDRO_ATLAS_DELTA_*.tif files"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--min-discharge",
        type=float,
        default=None,
        help=f"Minimum discharge threshold in m³/s (default: {MIN_DISCHARGE_M3S})"
    )
    parser.add_argument(
        "--min-order",
        type=int,
        default=None,
        help=f"Minimum Strahler stream order (default: {MIN_STREAM_ORDER})"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
        help="Bounding box for spatial filter"
    )
    parser.add_argument(
        "--save-gpkg",
        action="store_true",
        help="Save GeoPackage output (large file, for visualization)"
    )
    
    args = parser.parse_args()
    
    run(
        riveratlas_path=args.riveratlas,
        delta_dir=args.delta_dir,
        out_dir=args.out_dir,
        min_discharge=args.min_discharge,
        min_order=args.min_order,
        bbox=tuple(args.bbox) if args.bbox else None,
        save_gpkg=args.save_gpkg
    )
