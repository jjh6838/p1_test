#!/usr/bin/env python3
"""
Supply analysis per country - adapted for Snakemake workflow
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def get_country_bbox(admin_boundaries, buffer=0.1):
    """Get bounding box for a country with buffer"""
    bounds = admin_boundaries.total_bounds
    minx, miny, maxx, maxy = bounds
    # Add buffer
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    return [minx, miny, maxx, maxy]

def process_country_supply(country_iso3, output_dir="outputs_per_country"):
    """Process supply analysis for a single country"""
    print(f"Processing country: {country_iso3}")
    
    # Set your common CRS (WGS84)
    COMMON_CRS = "EPSG:4326"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load admin boundaries for this country
    print(f"Loading admin boundaries for {country_iso3}...")
    admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410.gpkg')
    admin_boundaries = admin_boundaries[admin_boundaries['GID_0'] == country_iso3]
    
    if admin_boundaries.empty:
        print(f"No boundaries found for country {country_iso3}")
        return None
    
    # Get country bounding box
    country_bbox = get_country_bbox(admin_boundaries)
    minx, miny, maxx, maxy = country_bbox
    
    print(f"Country bbox: {country_bbox}")
    
    # Simplify geometry for faster processing
    print("Simplifying geometry...")
    admin_boundaries['geometry'] = admin_boundaries['geometry'].simplify(tolerance=0.1, preserve_topology=True)
    # Tolerance of 0.001° ≈ 100m simplification

    # Load grid data for this country
    print("Loading grid data...")
    try:
        grid = gpd.read_file('bigdata_gridfinder/grid.gpkg', bbox=country_bbox)
        if grid.crs != COMMON_CRS:
            grid = grid.to_crs(COMMON_CRS)
    except Exception as e:
        print(f"Warning: Could not load grid data for {country_iso3}: {e}")
        grid = gpd.GeoDataFrame(columns=['geometry'], crs=COMMON_CRS)
    
    # Load and mask population raster
    print("Loading population raster...")
    try:
        with rasterio.open('bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif') as src:
            # Use bounding box to window read only the relevant part
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
            pop_data = src.read(1, window=window)
            
            # Get the transform for the windowed data
            windowed_transform = rasterio.windows.transform(window, src.transform)
            
            # Calculate centroids for each cell
            print("Calculating cell centroids...")
            rows, cols = pop_data.shape
            centroids_x = []
            centroids_y = []
            
            for row in range(rows):
                for col in range(cols):
                    # Convert pixel coordinates to geographic coordinates
                    x, y = rasterio.transform.xy(windowed_transform, row, col)
                    centroids_x.append(x)
                    centroids_y.append(y)
            
            # Create centroids GeoDataFrame
            centroids_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(centroids_x, centroids_y),
                crs=COMMON_CRS
            )
            
            # Calculate population per centroid (cell)
            centroids_gdf["Population_centroid"] = pop_data.flatten()
            
            # Filter centroids to only those within country boundaries
            print("Filtering centroids to country boundaries...")
            centroids_gdf = gpd.sjoin(centroids_gdf, admin_boundaries, how='inner', predicate='within')
            
    except Exception as e:
        print(f"Error processing population data for {country_iso3}: {e}")
        return None
    
    # Calculate total population in the country
    total_country_population = centroids_gdf["Population_centroid"].sum()
    
    if total_country_population == 0:
        print(f"No population found for country {country_iso3}")
        return None
    
    print(f"Total population in {country_iso3}: {total_country_population:,.0f}")
    
    # Load national supply data for this specific country
    try:
        supply_df = pd.read_excel("outputs_processed_data/p1_a_ember_2024_30.xlsx")
        
        # Filter for this country using ISO3_code column
        if 'ISO3_code' in supply_df.columns:
            country_supply = supply_df[supply_df['ISO3_code'] == country_iso3]
            if country_supply.empty:
                print(f"Warning: No supply data found for {country_iso3} in supply file")
                # Create empty supply data - will use default values below
                supply_df = pd.DataFrame()
            else:
                supply_df = country_supply
                print(f"Loaded supply data for {country_iso3}: {len(supply_df)} records")
        else:
            print(f"Warning: ISO3_code column not found in supply data, using all data")
            # Keep full supply_df and let original logic handle it
            
    except Exception as e:
        print(f"Warning: Could not load supply data for {country_iso3}: {e}")
        # Create dummy supply data
        supply_df = pd.DataFrame()
    
    # Define supply types and years
    supply_types = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]
    years = [2030, 2050]  # Focus on future projections only
    
    # Calculate and assign total demand for each centroid for each year
    # Methodology: Assumes national generation = national demand (supply-demand equivalence)
    for year in years:
        # Calculate total national supply for this year (projected generation)
        total_national_supply = 0
        if not supply_df.empty:
            for supply_type in supply_types:
                col = f"{supply_type}_{year}_MWh"
                if col in supply_df.columns:
                    total_national_supply += supply_df[col].sum()
        
        # If no supply data, use a default per capita value
        if total_national_supply == 0:
            # Default: 5 MWh per person per year (rough estimate)
            total_national_supply = total_country_population * 5
            print(f"Using default supply for {country_iso3} in {year}: {total_national_supply:,.0f} MWh")
        
        # Spatial allocation using population share methodology:
        # Demand_centroid = Population_Share_centroid × National_Supply_year (supply-demand equivalence)
        # Population_Share_centroid = Population_centroid / Total_Population
        demand_col = f"Total_Demand_{year}_centroid"
        centroids_gdf[demand_col] = (
            centroids_gdf["Population_centroid"] / total_country_population * total_national_supply
        )
    
    # Filter out centroids with zero population
    centroids_filtered = centroids_gdf[centroids_gdf["Population_centroid"] > 0]
    
    print(f"Filtered centroids: {len(centroids_filtered)} with population > 0")
    
    # Add country identifier
    centroids_filtered['GID_0'] = country_iso3
    
    # Save results
    output_file = output_path / f"supply_analysis_{country_iso3}.parquet"
    centroids_filtered.to_parquet(output_file)
    print(f"Results saved to {output_file}")
    
    # Also save as CSV for easier inspection
    csv_file = output_path / f"supply_analysis_{country_iso3}.csv"
    centroids_filtered.drop('geometry', axis=1).to_csv(csv_file, index=False)
    
    # Create a simple visualization if requested
    create_country_plot = True  # Set to False for cluster runs
    if create_country_plot and len(centroids_filtered) < 10000:  # Only plot if not too many points
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot admin boundaries
            admin_boundaries.boundary.plot(ax=ax, color='red', linewidth=2, label='Admin Boundary')
            
            # Plot grid if available
            if not grid.empty:
                grid.plot(ax=ax, color='white', linewidth=0.5, alpha=0.8, label='Grid')
            
            # Plot centroids
            centroids_filtered.plot(ax=ax, color='orange', markersize=2, alpha=0.6, label='Population Centroids')
            
            plt.title(f"{country_iso3}: Population Distribution and Grid", fontsize=14)
            plt.xlabel("Longitude", fontsize=12)
            plt.ylabel("Latitude", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = output_path / f"supply_analysis_{country_iso3}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Could not create plot for {country_iso3}: {e}")
    
    return str(output_file)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Process supply analysis for a country')
    parser.add_argument('country_iso3', help='ISO3 country code')
    parser.add_argument('--output-dir', default='outputs_per_country', help='Output directory')
    
    args = parser.parse_args()
    
    result = process_country_supply(args.country_iso3, args.output_dir)
    if result:
        print(f"Successfully processed {args.country_iso3}")
        return 0
    else:
        print(f"Failed to process {args.country_iso3}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
