import geopandas as gpd
from pathlib import Path
import argparse
import os
import re

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_bigdata_path(folder_name):
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


def get_year_from_scenario(scenario: str) -> int:
    """
    Extract year (2030 or 2050) from scenario string.
    E.g., '2030_supply_100%' -> 2030, '2050_supply_100%_add_v2' -> 2050
    """
    match = re.match(r'^(\d{4})', scenario)
    if match:
        return int(match.group(1))
    return 2030  # Default to 2030 if not found


def get_cmip6_layer_paths(year: int) -> dict:
    """
    Get paths to CMIP6 WPD and PVOUT parquet files for given year.
    Returns dict with layer names as keys and paths as values.
    """
    wind_dir = Path(get_bigdata_path("bigdata_wind_cmip6")) / "outputs"
    solar_dir = Path(get_bigdata_path("bigdata_solar_cmip6")) / "outputs"
    suffix = "300arcsec"
    
    return {
        # Wind layers (clip to GADM + EEZ)
        "wpd": wind_dir / f"WPD100_{year}_{suffix}.parquet",
        "wpd_uncertainty": wind_dir / f"WPD100_UNCERTAINTY_{year}_{suffix}.parquet",
        "wpd_baseline": wind_dir / f"WPD100_ERA5_baseline_{suffix}.parquet",
        # Solar layers (clip to GADM only)
        "pvout": solar_dir / f"PVOUT_{year}_{suffix}.parquet",
        "pvout_uncertainty": solar_dir / f"PVOUT_UNCERTAINTY_{year}_{suffix}.parquet",
        "pvout_baseline": solar_dir / f"PVOUT_baseline_{suffix}.parquet",
    }


def load_country_boundary(iso3: str) -> tuple:
    """
    Load country boundaries from GADM and EEZ datasets.
    
    Returns:
        tuple: (gadm_geometry, eez_geometry) - either can be None if not found
    """
    gadm_geom = None
    eez_geom = None
    
    # Load GADM boundary (land)
    gadm_path = Path(get_bigdata_path("bigdata_gadm")) / "gadm_410-levels.gpkg"
    if gadm_path.exists():
        try:
            gadm = gpd.read_file(gadm_path, layer="ADM_0", where=f"GID_0 = '{iso3}'")
            if not gadm.empty:
                gadm_geom = gadm.union_all()
                print(f"[INFO] Loaded GADM boundary for {iso3}")
        except Exception as e:
            print(f"[WARN] Failed to load GADM for {iso3}: {e}")
    
    # Load EEZ boundary (maritime)
    eez_path = Path(get_bigdata_path("bigdata_eez")) / "eez_v12.gpkg"
    if eez_path.exists():
        try:
            eez = gpd.read_file(eez_path, where=f"ISO_TER1 = '{iso3}'")
            if not eez.empty:
                eez_geom = eez.union_all()
                print(f"[INFO] Loaded EEZ boundary for {iso3}")
        except Exception as e:
            print(f"[WARN] Failed to load EEZ for {iso3}: {e}")
    
    return gadm_geom, eez_geom


def load_cmip6_layers_clipped(iso3: str, year: int) -> dict:
    """
    Load CMIP6 WPD and PVOUT layers, clipped to country boundaries.
    - WPD layers: clipped to GADM + EEZ union (offshore wind)
    - PVOUT layers: clipped to GADM only (no floating solar)
    
    Returns:
        dict: {layer_name: GeoDataFrame} for layers that exist and have data
    """
    cmip6_paths = get_cmip6_layer_paths(year)
    
    # Check if any CMIP6 files exist
    existing_files = {k: v for k, v in cmip6_paths.items() if v.exists()}
    if not existing_files:
        print(f"[WARN] No CMIP6 parquet files found. Run p1_c_cmip6_solar.py and p1_d_cmip6_wind.py first.")
        return {}
    
    # Load country boundaries
    gadm_geom, eez_geom = load_country_boundary(iso3)
    
    if gadm_geom is None:
        print(f"[WARN] No GADM boundary found for {iso3}, skipping CMIP6 layers")
        return {}
    
    # Create boundary geometries
    gadm_only = gadm_geom
    gadm_plus_eez = gadm_geom.union(eez_geom) if eez_geom else gadm_geom
    
    # Create GeoDataFrames for clipping
    gadm_gdf = gpd.GeoDataFrame(geometry=[gadm_only], crs="EPSG:4326")
    gadm_eez_gdf = gpd.GeoDataFrame(geometry=[gadm_plus_eez], crs="EPSG:4326")
    
    clipped_layers = {}
    
    for layer_name, parquet_path in cmip6_paths.items():
        if not parquet_path.exists():
            print(f"[WARN] CMIP6 file not found: {parquet_path.name}")
            continue
        
        try:
            print(f"[INFO] Loading CMIP6 layer '{layer_name}' from {parquet_path.name}")
            gdf = gpd.read_parquet(parquet_path)
            
            # Determine which boundary to use
            if layer_name.startswith("wpd"):
                # Wind: use GADM + EEZ
                clip_gdf = gadm_eez_gdf
                boundary_type = "GADM+EEZ"
            else:
                # Solar: use GADM only
                clip_gdf = gadm_gdf
                boundary_type = "GADM"
            
            # Clip using spatial join (faster for points)
            clipped = gpd.sjoin(gdf, clip_gdf, predicate="within", how="inner")
            
            # Drop the index_right column from sjoin
            if "index_right" in clipped.columns:
                clipped = clipped.drop(columns=["index_right"])
            
            if not clipped.empty:
                clipped_layers[layer_name] = clipped
                print(f"[INFO] Clipped '{layer_name}' to {boundary_type}: {len(clipped):,} points")
            else:
                print(f"[WARN] No data after clipping '{layer_name}' to {boundary_type}")
                
        except Exception as e:
            print(f"[WARN] Failed to load/clip CMIP6 layer '{layer_name}': {e}")
    
    return clipped_layers


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parquet_to_gpkg(base_dir, scenario, iso3):
    """
    Combine four Parquet files into one GPKG with four layers.
    Input directory: outputs_per_country/parquet/<scenario>/
    Output GPKG:     outputs_per_country/<scenario>_<iso3>.gpkg (basic 4 layers)
                     outputs_per_country/<scenario>_<iso3>_add.gpkg (if additional layers exist)
                     outputs_per_country/<scenario>_<iso3>_add_v2.gpkg (if _add_v2 files exist)
    """

    base_dir = Path(base_dir)
    in_dir = base_dir / "parquet" / scenario
    
    # Check for _add_v2 files first (supply analysis with siting)
    add_v2_files = {
        "centroids":    in_dir / f"centroids_{iso3}_add_v2.parquet",
        "polylines":    in_dir / f"polylines_{iso3}_add_v2.parquet",
        "grid_lines":   in_dir / f"grid_lines_{iso3}_add_v2.parquet",
        "facilities":   in_dir / f"facilities_{iso3}_add_v2.parquet",
    }
    
    has_add_v2 = any(fp.exists() for fp in add_v2_files.values())
    
    # Four core layers
    core_files = {
        "centroids":    in_dir / f"centroids_{iso3}.parquet",
        "polylines":    in_dir / f"polylines_{iso3}.parquet",
        "grid_lines":   in_dir / f"grid_lines_{iso3}.parquet",
        "facilities":   in_dir / f"facilities_{iso3}.parquet",
    }
    
    # Additional layers (siting results)
    additional_files = {
        "siting_clusters": in_dir / f"siting_clusters_{iso3}.parquet",
        "siting_settlements": in_dir / f"siting_settlements_{iso3}.parquet",
        "siting_networks": in_dir / f"siting_networks_{iso3}.parquet",
    }
    
    # Check if any additional layers exist
    has_additional = any(fp.exists() for fp in additional_files.values())
    
    # Debug output
    print(f"\n[DEBUG] Checking files for {iso3} in scenario: {scenario}")
    print(f"  has_add_v2: {has_add_v2}")
    print(f"  has_additional (siting): {has_additional}")
    if has_additional:
        print(f"  Found siting layers:")
        for name, fp in additional_files.items():
            if fp.exists():
                print(f"    - {name}: {fp.name}")
    
    # Determine output filename and files to process
    if has_add_v2:
        # If scenario already ends with _add_v2, strip it and re-add after iso3
        if scenario.endswith("_add_v2"):
            base_scenario = scenario[:-7]  # Remove _add_v2
            out_gpkg = base_dir / f"{base_scenario}_{iso3}_add_v2.gpkg"
        else:
            out_gpkg = base_dir / f"{scenario}_{iso3}_add_v2.gpkg"
        files = add_v2_files
    elif has_additional:
        out_gpkg = base_dir / f"{scenario}_{iso3}_add.gpkg"
        files = {**core_files, **additional_files}
    else:
        out_gpkg = base_dir / f"{scenario}_{iso3}.gpkg"
        files = core_files

    # Remove old GPKG if exists (avoid append/mixed layers)
    if out_gpkg.exists():
        out_gpkg.unlink()

    for layer, fp in files.items():
        if not fp.exists():
            print(f"[WARN] Missing file for {layer}: {fp}")
            continue

        print(f"[INFO] Writing layer '{layer}' from {fp}")
        gdf = gpd.read_parquet(fp)

        # enforce CRS if needed:
        # gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")

    # -------------------------------------------------------------------------
    # Add CMIP6 layers (WPD and PVOUT) - always included regardless of stage
    # -------------------------------------------------------------------------
    year = get_year_from_scenario(scenario)
    print(f"\n[INFO] Loading CMIP6 layers for year {year}...")
    
    cmip6_layers = load_cmip6_layers_clipped(iso3, year)
    
    if cmip6_layers:
        print(f"\n--- Writing CMIP6 layers ---")
        for layer_name, gdf in cmip6_layers.items():
            print(f"[INFO] Writing CMIP6 layer '{layer_name}': {len(gdf):,} points")
            gdf.to_file(out_gpkg, layer=layer_name, driver="GPKG")
        print(f"[INFO] Added {len(cmip6_layers)} CMIP6 layers to GPKG")
    else:
        print(f"[WARN] No CMIP6 layers added (files not found or empty)")

    print(f"[DONE] Created {out_gpkg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Parquet files into GPKG for a single country")
    parser.add_argument("iso3", help="ISO3 country code (e.g., KEN)")
    parser.add_argument("--scenario", default="2030_supply_100%", help="Scenario name (default: 2030_supply_100%%)")
    parser.add_argument("--base-dir", default="outputs_per_country", help="Base directory (default: outputs_per_country)")
    
    args = parser.parse_args()
    
    parquet_to_gpkg(
        base_dir=args.base_dir,
        scenario=args.scenario,
        iso3=args.iso3
    )
