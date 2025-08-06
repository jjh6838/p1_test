#!/usr/bin/env python3
"""
Combine all country supply analysis results into global dataset
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
import sys
import logging
from tqdm import tqdm
from datetime import datetime

def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file is None:
        log_file = Path("outputs_global/logs/combine_results.log")
    
    # Create log directory
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def combine_global_results(input_dir="outputs_per_country", output_file="outputs_global/global_supply_analysis.parquet"):
    """Combine all country results into a single global dataset"""
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting global results combination at {datetime.now()}")
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Create input and output directories if they don't exist
    input_path.mkdir(exist_ok=True)
    output_path.parent.mkdir(exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return False
    
    # Find all country parquet files
    country_files = list(input_path.glob("supply_analysis_*.parquet"))
    
    if not country_files:
        logger.error(f"No country files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(country_files)} country files to combine")
    
    # Read and combine all country data
    all_data = []
    failed_countries = []
    
    for file_path in tqdm(country_files, desc="Combining countries"):
        try:
            country_data = gpd.read_parquet(file_path)
            all_data.append(country_data)
            country = file_path.stem.split('_')[-1]  # Extract country code from filename
            logger.info(f"Loaded {country}: {len(country_data)} records")
        except Exception as e:
            country = file_path.stem.split('_')[-1]
            failed_countries.append(country)
            logger.error(f"Failed to load {country}: {e}")
    
    if not all_data:
        logger.error("No valid country data found")
        return False
    
    # Combine all data
    logger.info("Combining all country data...")
    global_df = gpd.pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Combined dataset: {len(global_df)} total records from {len(all_data)} countries")
    
    # Expected columns after the update
    expected_columns = [
        'Population_centroid',
        'GID_0',
        'NAME_0', 
        'Total_Demand_2024_centroid',  # Added 2024 back
        'Total_Demand_2030_centroid',
        'Total_Demand_2050_centroid',
        'geometry'
    ]
    
    # Summary statistics
    print(f"\n=== GLOBAL SUMMARY ===")
    print(f"Total countries: {global_df['GID_0'].nunique()}")
    print(f"Total population centroids: {len(global_df):,}")
    print(f"Total population (2025 baseline): {global_df['Population_centroid'].sum():,.0f}")
    
    # Summary for all three years now
    years = [2024, 2030, 2050]
    for year in years:
        demand_col = f"Total_Demand_{year}_centroid"
        if demand_col in global_df.columns:
            total_demand = global_df[demand_col].sum()
            avg_per_capita = total_demand / global_df['Population_centroid'].sum()
            print(f"Global demand {year}: {total_demand:,.0f} MWh ({avg_per_capita:.2f} MWh/capita)")
    
    # Updated column validation
    for col in expected_columns:
        if col not in global_df.columns:
            print(f"Warning: Expected column '{col}' not found")
        else:
            print(f"✓ Column '{col}': {global_df[col].count():,} non-null values")
    
    # Create summary by country
    country_summary = global_df.groupby('GID_0').agg({
        'Population_centroid': ['count', 'sum'],
        'Total_Demand_2030_centroid': 'sum',
        'Total_Demand_2050_centroid': 'sum'
    }).round(0)
    country_summary.columns = ['Num_Centroids', 'Total_Population', 'Demand_2030_MWh', 'Demand_2050_MWh']
    
    # Save global dataset
    print(f"\nSaving global dataset to {output_file}...")
    global_df.to_parquet(output_file)
    
    # Also save summary statistics
    summary_file = Path(output_file).with_suffix('.csv').with_name('global_supply_summary.csv')
    country_summary.to_csv(summary_file)
    
    # Save as CSV without geometry for easier analysis
    csv_file = Path(output_file).with_suffix('.csv')
    global_df.drop('geometry', axis=1).to_csv(csv_file, index=False)
    
    print(f"Global dataset saved to:")
    print(f"  - {output_file} (parquet with geometry)")
    print(f"  - {csv_file} (CSV without geometry)")
    print(f"  - {summary_file} (summary by country)")
    
    if failed_countries:
        print(f"\nFailed to process {len(failed_countries)} countries: {failed_countries}")
    
    # Global totals
    total_population = global_df['Population_centroid'].sum()
    total_demand_2030 = global_df['Total_Demand_2030_centroid'].sum()
    
    print(f"\nGlobal totals:")
    print(f"  Total population: {total_population:,.0f}")
    print(f"  Total demand 2030: {total_demand_2030:,.0f} MWh")
    print(f"  Average demand per capita 2030: {total_demand_2030/total_population:.1f} MWh/person")
    
    return True

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Combine country supply analysis results')
    parser.add_argument('--input-dir', default='outputs_per_country', help='Input directory with country files')
    parser.add_argument('--output-file', default='outputs_global/global_supply_analysis.parquet', help='Output file name')
    parser.add_argument('--log-file', default='outputs_global/logs/combine_results.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(Path(args.log_file))
    logger.info("="*60)
    logger.info("STARTING GLOBAL RESULTS COMBINATION")
    logger.info("="*60)
    
    success = combine_global_results(args.input_dir, args.output_file)
    
    if success:
        logger.info("Global results combination completed successfully!")
    else:
        logger.error("Global results combination failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
