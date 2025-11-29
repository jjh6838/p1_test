import geopandas as gpd
from pathlib import Path

def parquet_to_gpkg(base_dir, scenario, iso3):
    """
    Combine four Parquet files into one GPKG with four layers.
    Input directory: outputs_per_country/parquet/<scenario>/
    Output GPKG:     outputs_per_country/<scenario>_<iso3>.gpkg
    """

    base_dir = Path(base_dir)
    in_dir = base_dir / "parquet" / scenario
    out_gpkg = base_dir / f"{scenario}_{iso3}.gpkg"

    # Four expected input files
    files = {
        "centroids":    in_dir / f"centroids_{iso3}.parquet",
        "polylines":    in_dir / f"polylines_{iso3}.parquet",
        "grid_lines":   in_dir / f"grid_lines_{iso3}.parquet",
        "facilities":   in_dir / f"facilities_{iso3}.parquet",
        "siting_clusters": in_dir / f"siting_clusters_{iso3}.parquet",
        "sitting_settlements": in_dir / f"siting_settlements_{iso3}.parquet",
        "siting_networks": in_dir / f"siting_networks_{iso3}.parquet",
    }

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


# Example usage (update scenario and iso3 as needed):
parquet_to_gpkg(
    base_dir="outputs_per_country",
    scenario="2030_supply_100%",
    iso3="KEN"
)
