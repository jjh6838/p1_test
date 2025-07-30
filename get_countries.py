#!/usr/bin/env python3
"""
Get list of countries from energy demand data for Snakemake workflow
Uses ISO3_code column from p1_a_ember_2024_30.xlsx
Automatically filters out countries without GADM boundaries
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

def get_country_list():
    """Get list of countries that have both demand data AND GADM boundaries"""
    demand_file = "outputs_processed_data/p1_a_ember_2024_30.xlsx"
    gadm_file = "bigdata_gadm/gadm_410.gpkg"
    
    # Check if files exist
    if not Path(demand_file).exists():
        print(f"Error: Demand data file not found: {demand_file}")
        return []
    
    if not Path(gadm_file).exists():
        print(f"Error: GADM boundaries file not found: {gadm_file}")
        return []
    
    print(f"Loading country list from energy demand data: {demand_file}")
    
    try:
        # Load demand data
        demand_df = pd.read_excel(demand_file)
        
        if 'ISO3_code' not in demand_df.columns:
            print(f"Error: 'ISO3_code' column not found in {demand_file}")
            print(f"Available columns: {list(demand_df.columns)}")
            return []
        
        # Get countries from demand data
        demand_countries = demand_df['ISO3_code'].dropna().unique()
        demand_countries = set(str(c).strip() for c in demand_countries if pd.notna(c) and str(c).strip())
        
        print(f"Found {len(demand_countries)} countries in demand data")
        
        # Load GADM boundaries to check which countries exist
        print("Checking which countries have GADM boundaries...")
        admin_df = gpd.read_file(gadm_file, columns=['GID_0'])
        gadm_countries = set(admin_df['GID_0'].dropna().unique())
        gadm_countries = {str(c).strip() for c in gadm_countries if pd.notna(c)}
        
        # Only keep countries that exist in both datasets
        valid_countries = demand_countries & gadm_countries
        missing_countries = demand_countries - gadm_countries
        
        print(f"Countries with both demand data AND boundaries: {len(valid_countries)}")
        if missing_countries:
            print(f"Countries with demand data but NO boundaries (will be skipped): {len(missing_countries)}")
            for country in sorted(list(missing_countries)[:5]):  # Show first 5
                print(f"  - {country}")
            if len(missing_countries) > 5:
                print(f"  ... and {len(missing_countries) - 5} more")
        
        # Sort for consistent ordering
        countries = sorted(list(valid_countries))
        
        print(f"\nFinal country list: {len(countries)} countries will be processed")
        for country in countries[:10]:  # Show first 10
            print(f"  {country}")
        if len(countries) > 10:
            print(f"  ... and {len(countries) - 10} more")
        
        # Save to file for Snakemake
        with open('countries_list.txt', 'w') as f:
            for country in countries:
                f.write(f"{country}\n")
        
        print(f"\nCountry list saved to countries_list.txt")
        print(f"Ready to process {len(countries)} countries in parallel!")
        
        return countries
        
    except Exception as e:
        print(f"Error processing country list: {e}")
        return []

if __name__ == "__main__":
    countries = get_country_list()
