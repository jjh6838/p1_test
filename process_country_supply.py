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
    admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410-levels.gpkg', layer="ADM_0")
    admin_boundaries = admin_boundaries[admin_boundaries['GID_0'] == country_iso3]
    
    if admin_boundaries.empty:
        print(f"No boundaries found for country {country_iso3}")
        return None
    
    # Get country name for output
    country_name = admin_boundaries['NAME_0'].iloc[0] if 'NAME_0' in admin_boundaries.columns else country_iso3
    
    # Get country bounding box
    country_bbox = get_country_bbox(admin_boundaries)
    minx, miny, maxx, maxy = country_bbox
    
    print(f"Country bbox: {country_bbox}")
    
    # Simplify geometry for faster processing
    print("Simplifying geometry...")
    admin_boundaries['geometry'] = admin_boundaries['geometry'].simplify(tolerance=0.1, preserve_topology=True)
    # Note: Tolerance should be adjusted to 0.001 for better accuracy; currently set to 0.1 for test (faster processing)

    # Load and mask population raster (2023 baseline)
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
            
            # Calculate population per centroid (cell) - this is the 2023 baseline
            centroids_gdf["Population_centroid"] = pop_data.flatten()
            
            # Filter centroids to only those within country boundaries
            print("Filtering centroids to country boundaries...")
            centroids_gdf = gpd.sjoin(centroids_gdf, admin_boundaries, how='inner', predicate='within')
            
    except Exception as e:
        print(f"Error processing population data for {country_iso3}: {e}")
        return None
    
    # Calculate total population in the country from raster (2023 baseline)
    total_country_population_2023 = centroids_gdf["Population_centroid"].sum()
    
    if total_country_population_2023 == 0:
        print(f"No population found for country {country_iso3}")
        return None
    
    print(f"Total population in {country_iso3} (2023 baseline): {total_country_population_2023:,.0f}")
    
    # Load country-level population projections from p1_a dataset
    print("Loading country-level population projections...")
    try:
        pop_projections_df = pd.read_excel("outputs_processed_data/p1_a_ember_2024_30.xlsx")
        
        # Filter for this country using ISO3_code column
        country_pop_data = None
        if 'ISO3_code' in pop_projections_df.columns:
            country_pop_match = pop_projections_df[pop_projections_df['ISO3_code'] == country_iso3]
            if not country_pop_match.empty:
                country_pop_data = country_pop_match.iloc[0]
                print(f"Found population projections for {country_iso3}")
            else:
                print(f"Warning: No population projections found for {country_iso3}")
        
        # Extract country-level population totals for each year
        if country_pop_data is not None:
            # Convert from thousands to actual population counts
            pop_2024 = country_pop_data.get('PopTotal_2024', 0) * 1000
            pop_2030 = country_pop_data.get('PopTotal_2030', 0) * 1000
            pop_2050 = country_pop_data.get('PopTotal_2050', 0) * 1000
            
            print(f"Country population projections:")
            print(f"  2024: {pop_2024:,.0f}")
            print(f"  2030: {pop_2030:,.0f}")
            print(f"  2050: {pop_2050:,.0f}")
        else:
            # Use 2023 baseline for all years if no projections available
            pop_2024 = total_country_population_2023
            pop_2030 = total_country_population_2023
            pop_2050 = total_country_population_2023
            print(f"Using 2023 baseline population for all years: {total_country_population_2023:,.0f}")
            
    except Exception as e:
        print(f"Warning: Could not load population projections for {country_iso3}: {e}")
        # Use 2023 baseline for all years
        pop_2024 = total_country_population_2023
        pop_2030 = total_country_population_2023
        pop_2050 = total_country_population_2023
    
    # Load national supply data for this specific country
    print("Loading supply projections...")
    try:
        supply_df = pd.read_excel("outputs_processed_data/p1_a_ember_2024_30.xlsx")
        
        # Filter for this country using ISO3_code column
        if 'ISO3_code' in supply_df.columns:
            country_supply = supply_df[supply_df['ISO3_code'] == country_iso3]
            if country_supply.empty:
                print(f"Warning: No supply data found for {country_iso3} in supply file")
                supply_df = pd.DataFrame()
            else:
                supply_df = country_supply
                print(f"Loaded supply data for {country_iso3}: {len(supply_df)} records")
        else:
            print(f"Warning: ISO3_code column not found in supply data")
            supply_df = pd.DataFrame()
            
    except Exception as e:
        print(f"Warning: Could not load supply data for {country_iso3}: {e}")
        supply_df = pd.DataFrame()
    
    # Define supply types and years
    supply_types = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]
    years = [2024, 2030, 2050]
    country_populations = {2024: pop_2024, 2030: pop_2030, 2050: pop_2050}
    
    # First, calculate population share for each centroid based on 2023 spatial distribution
    population_share = centroids_gdf["Population_centroid"] / total_country_population_2023
    
    # Calculate spatially distributed population for each year
    print("\nAllocating population projections to centroids...")
    for year in years:
        total_country_population_year = country_populations[year]
        
        # Allocate projected population to each centroid using 2023 spatial pattern
        pop_col = f"Population_{year}_centroid"
        centroids_gdf[pop_col] = population_share * total_country_population_year
        
        print(f"  {year} population allocated: {centroids_gdf[pop_col].sum():,.0f} total")
    
    # Calculate total demand for each centroid for each year
    for year in years:
        print(f"\nProcessing energy demand for year {year}...")
        
        # Get country-level population for this year
        total_country_population_year = country_populations[year]
        
        # Calculate total national supply for this year (projected generation)
        total_national_supply = 0
        if not supply_df.empty:
            for supply_type in supply_types:
                col = f"{supply_type}_{year}_MWh"
                if col in supply_df.columns:
                    supply_value = supply_df[col].iloc[0] if not pd.isna(supply_df[col].iloc[0]) else 0
                    total_national_supply += supply_value
        
        # If no supply data, use a default per capita value
        if total_national_supply == 0:
            # Default: 0 MWh per person per year (Can put global average 3.2 MWh per person per year if needed but all data have been calculated)
            total_national_supply = total_country_population_year * 0
            print(f"Using default supply for {country_iso3} in {year}: {total_national_supply:,.0f} MWh")
        else:
            print(f"Total national supply for {year}: {total_national_supply:,.0f} MWh")
        
        # Calculate demand using supply-demand equivalence assumption
        # Demand_centroid = Population_Share_centroid × National_Supply_year
        demand_col = f"Total_Demand_{year}_centroid"
        centroids_gdf[demand_col] = population_share * total_national_supply
        
        print(f"  Allocated {total_national_supply:,.0f} MWh across {len(centroids_gdf)} centroids")
        print(f"  Per capita demand: {total_national_supply/total_country_population_year:.2f} MWh/person/year")
    
    # Filter out centroids with zero population
    centroids_filtered = centroids_gdf[centroids_gdf["Population_centroid"] > 0].copy()
    
    print(f"\nFiltered centroids: {len(centroids_filtered)} with population > 0")
    
    # Add country identifiers
    centroids_filtered['GID_0'] = country_iso3
    centroids_filtered['NAME_0'] = country_name
    
    # Keep the specified columns for simplified output - now including population projections
    final_columns = [
        'geometry',
        'GID_0',
        'NAME_0',
        'Population_centroid',          # 2023 baseline population per centroid
        'Population_2024_centroid',     # 2024 population projection per centroid
        'Population_2030_centroid',     # 2030 population projection per centroid
        'Population_2050_centroid',     # 2050 population projection per centroid
        'Total_Demand_2024_centroid',
        'Total_Demand_2030_centroid', 
        'Total_Demand_2050_centroid'
        ]
    
    # Only keep columns that exist in the dataframe
    available_columns = [col for col in final_columns if col in centroids_filtered.columns]
    centroids_simplified = centroids_filtered[available_columns].copy()
    
    print(f"Final output columns: {available_columns}")
    
    # Save results
    output_file = output_path / f"supply_analysis_{country_iso3}.parquet"
    centroids_simplified.to_parquet(output_file)
    print(f"Results saved to {output_file}")
    
    # Also save as CSV for easier inspection (without geometry)
    csv_file = output_path / f"supply_analysis_{country_iso3}.csv"
    csv_columns = [col for col in available_columns if col != 'geometry']
    centroids_simplified[csv_columns].to_csv(csv_file, index=False)
    print(f"CSV saved to {csv_file}")
    
    # Print summary statistics - now including population projections
    print(f"\n=== SUMMARY for {country_iso3} ===")
    print(f"Population centroids: {len(centroids_simplified):,}")
    print(f"Total population (2023 baseline): {centroids_simplified['Population_centroid'].sum():,.0f}")
    
    # Population projections summary
    for year in years:
        pop_col = f"Population_{year}_centroid"
        if pop_col in centroids_simplified.columns:
            total_pop = centroids_simplified[pop_col].sum()
            print(f"Total population {year}: {total_pop:,.0f}")
    
    # Demand projections summary
    for year in years:
        demand_col = f"Total_Demand_{year}_centroid"
        if demand_col in centroids_simplified.columns:
            total_demand = centroids_simplified[demand_col].sum()
            pop_col = f"Population_{year}_centroid"
            if pop_col in centroids_simplified.columns:
                pop_year = centroids_simplified[pop_col].sum()
                per_capita = total_demand / pop_year if pop_year > 0 else 0
                print(f"Total demand {year}: {total_demand:,.0f} MWh ({per_capita:.2f} MWh/capita)")
    
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
