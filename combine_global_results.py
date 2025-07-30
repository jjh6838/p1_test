#!/usr/bin/env python3
"""
Combine all country supply analysis results into global dataset
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

def combine_country_results(input_dir="outputs_per_country", output_file="global_supply_analysis.parquet"):
    """Combine all country results into a single global dataset"""
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist")
        return False
    
    # Find all country parquet files
    country_files = list(input_path.glob("supply_analysis_*.parquet"))
    
    if not country_files:
        print(f"No country files found in {input_dir}")
        return False
    
    print(f"Found {len(country_files)} country files to combine")
    
    # Read and combine all country data
    all_data = []
    failed_countries = []
    
    for file_path in tqdm(country_files, desc="Combining countries"):
        try:
            country_data = gpd.read_parquet(file_path)
            all_data.append(country_data)
            country = file_path.stem.split('_')[-1]  # Extract country code from filename
            print(f"  Loaded {country}: {len(country_data)} records")
        except Exception as e:
            country = file_path.stem.split('_')[-1]
            failed_countries.append(country)
            print(f"  Failed to load {country}: {e}")
    
    if not all_data:
        print("No valid country data found")
        return False
    
    # Combine all data
    print("Combining all country data...")
    global_data = gpd.pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset: {len(global_data)} total records from {len(all_data)} countries")
    
    # Summary statistics
    print("\nSummary by country:")
    country_summary = global_data.groupby('GID_0').agg({
        'Population_centroid': ['count', 'sum'],
        'Total_Demand_2024_centroid': 'sum',
        'Total_Demand_2030_centroid': 'sum',
        'Total_Demand_2050_centroid': 'sum'
    }).round(0)
    
    country_summary.columns = ['Num_Centroids', 'Total_Population', 'Demand_2024_MWh', 'Demand_2030_MWh', 'Demand_2050_MWh']
    print(country_summary.head(10))
    
    # Save global dataset
    print(f"\nSaving global dataset to {output_file}...")
    global_data.to_parquet(output_file)
    
    # Also save summary statistics
    summary_file = Path(output_file).with_suffix('.csv').with_name('global_supply_summary.csv')
    country_summary.to_csv(summary_file)
    
    # Save as CSV without geometry for easier analysis
    csv_file = Path(output_file).with_suffix('.csv')
    global_data.drop('geometry', axis=1).to_csv(csv_file, index=False)
    
    print(f"Global dataset saved to:")
    print(f"  - {output_file} (parquet with geometry)")
    print(f"  - {csv_file} (CSV without geometry)")
    print(f"  - {summary_file} (summary by country)")
    
    if failed_countries:
        print(f"\nFailed to process {len(failed_countries)} countries: {failed_countries}")
    
    # Global totals
    total_population = global_data['Population_centroid'].sum()
    total_demand_2030 = global_data['Total_Demand_2030_centroid'].sum()
    
    print(f"\nGlobal totals:")
    print(f"  Total population: {total_population:,.0f}")
    print(f"  Total demand 2030: {total_demand_2030:,.0f} MWh")
    print(f"  Average demand per capita 2030: {total_demand_2030/total_population:.1f} MWh/person")
    
    return True

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Combine country supply analysis results')
    parser.add_argument('--input-dir', default='outputs_per_country', help='Input directory with country files')
    parser.add_argument('--output-file', default='global_supply_analysis.parquet', help='Output file name')
    
    args = parser.parse_args()
    
    success = combine_country_results(args.input_dir, args.output_file)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
