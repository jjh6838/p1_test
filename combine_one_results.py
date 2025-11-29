import geopandas as gpd
from pathlib import Path
import argparse

def parquet_to_gpkg(base_dir, scenario, iso3):
    """
    Combine four Parquet files into one GPKG with four layers.
    Input directory: outputs_per_country/parquet/<scenario>/
    Output GPKG:     outputs_per_country/<scenario>_<iso3>.gpkg (basic 4 layers)
                     outputs_per_country/<scenario>_<iso3>_v2.gpkg (if additional layers exist)
    """

    base_dir = Path(base_dir)
    in_dir = base_dir / "parquet" / scenario
    
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
    
    # Determine output filename
    if has_additional:
        out_gpkg = base_dir / f"{scenario}_{iso3}_v2.gpkg"
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
