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

def combine_layer_data(input_dir, layer_name, countries_list=None):
    """Combine data from a specific layer across all countries"""
    
    input_path = Path(input_dir)
    
    # Find all country GPKG files (updated from parquet)
    country_files = list(input_path.glob("supply_analysis_*.gpkg"))
    
    if countries_list:
        # Filter to only specified countries
        country_files = [f for f in country_files if f.stem.split('_')[-1] in countries_list]
    
    if not country_files:
        logging.warning(f"No country files found for layer {layer_name}")
        return None
    
    logging.info(f"Combining {layer_name} layer from {len(country_files)} countries")
    
    # Read and combine data from this layer
    layer_data_list = []
    failed_countries = []
    
    for file_path in tqdm(country_files, desc=f"Processing {layer_name}"):
        try:
            country_code = file_path.stem.split('_')[-1]
            
            # Check if this layer exists in the file
            try:
                layers = gpd.list_layers(file_path)
                layer_names = [layer[0] for layer in layers]
            except:
                logging.warning(f"Could not read layers from {country_code}")
                continue
            
            if layer_name not in layer_names:
                logging.warning(f"Layer '{layer_name}' not found in {country_code}")
                continue
            
            # Read the specific layer
            layer_data = gpd.read_file(file_path, layer=layer_name)
            
            if layer_data.empty:
                logging.warning(f"Empty {layer_name} data for {country_code}")
                continue
            
            layer_data_list.append(layer_data)
            logging.info(f"Loaded {layer_name} from {country_code}: {len(layer_data)} records")
            
        except Exception as e:
            failed_countries.append(country_code)
            logging.error(f"Failed to load {layer_name} from {country_code}: {e}")
    
    if not layer_data_list:
        logging.error(f"No valid {layer_name} data found")
        return None
    
    # Combine all layer data
    logging.info(f"Combining {layer_name} data from {len(layer_data_list)} countries...")
    combined_data = pd.concat(layer_data_list, ignore_index=True)
    
    logging.info(f"Combined {layer_name}: {len(combined_data)} total records")
    
    if failed_countries:
        logging.warning(f"Failed to process {layer_name} from {len(failed_countries)} countries: {failed_countries}")
    
    return combined_data

def combine_global_results(input_dir="outputs_per_country", output_file="outputs_global/global_supply_analysis.parquet", countries_list=None):
    """Combine all country results into global datasets by layer"""
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting global results combination at {datetime.now()}")
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_dir = output_path.parent
    
    # Create input and output directories if they don't exist
    input_path.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return False
    
    # Find all country GPKG files (updated from parquet)
    country_files = list(input_path.glob("supply_analysis_*.gpkg"))
    
    # Filter by countries list if provided
    if countries_list:
        country_files = [f for f in country_files if f.stem.split('_')[-1] in countries_list]
        logger.info(f"Filtered to {len(country_files)} countries from provided list")
    
    if not country_files:
        logger.error(f"No country files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(country_files)} country files to combine")
    
    # Define layers to process
    layers_to_process = ['centroids', 'grid_lines', 'facilities']
    
    results = {}
    
    # Process each layer
    for layer_name in layers_to_process:
        logger.info(f"Processing {layer_name} layer...")
        
        combined_data = combine_layer_data(input_path, layer_name)
        
        if combined_data is not None and not combined_data.empty:
            results[layer_name] = combined_data
            
            # Save layer data
            layer_file_gpkg = output_dir / f"global_{layer_name}.gpkg"
            layer_file_csv = output_dir / f"global_{layer_name}.csv"
            
            # Save as GPKG (with geometry)
            combined_data.to_file(layer_file_gpkg, driver="GPKG")
            logger.info(f"Saved {layer_name} to {layer_file_gpkg}")
            
            # Save as CSV (without geometry)
            csv_data = combined_data.drop('geometry', axis=1) if 'geometry' in combined_data.columns else combined_data
            csv_data.to_csv(layer_file_csv, index=False)
            logger.info(f"Saved {layer_name} to {layer_file_csv}")
            
        else:
            logger.warning(f"No data found for {layer_name} layer")
    
    # Process centroids layer for legacy compatibility
    if 'centroids' in results:
        centroids_data = results['centroids']
        
        # Save as legacy parquet format
        centroids_data.to_parquet(output_file)
        logger.info(f"Saved legacy format to {output_file}")
        
        # Save as CSV without geometry for easier analysis
        csv_file = Path(output_file).with_suffix('.csv')
        centroids_data.drop('geometry', axis=1).to_csv(csv_file, index=False)
        logger.info(f"Saved legacy CSV to {csv_file}")
    else:
        # Fallback to original approach if no centroids layer
        logger.warning("No centroids layer found, attempting original parquet combination...")
        
        # Try to find parquet files as fallback
        parquet_files = list(input_path.glob("supply_analysis_*.parquet"))
        if parquet_files:
            all_data = []
            for file_path in tqdm(parquet_files, desc="Combining parquet files"):
                try:
                    country_data = gpd.read_parquet(file_path)
                    all_data.append(country_data)
                    country = file_path.stem.split('_')[-1]
                    logger.info(f"Loaded {country}: {len(country_data)} records")
                except Exception as e:
                    country = file_path.stem.split('_')[-1]
                    logger.error(f"Failed to load {country}: {e}")
            
            if all_data:
                global_df = gpd.pd.concat(all_data, ignore_index=True)
                global_df.to_parquet(output_file)
                results['centroids'] = global_df  # For summary generation
    
    # Create a combined GPKG with all layers
    if results:
        logger.info("Creating combined GPKG with all layers...")
        combined_gpkg = output_dir / "global_supply_analysis_all_layers.gpkg"
        
        for layer_name, data in results.items():
            data.to_file(combined_gpkg, driver="GPKG", layer=layer_name)
            logger.info(f"Added {layer_name} layer to combined GPKG")
        
        logger.info(f"Combined GPKG saved to {combined_gpkg}")
    
    # Generate summary statistics (updated logic)
    if 'centroids' in results:
        centroids_data = results['centroids']
        
        # Expected columns after the update
        expected_columns = [
            'Population_centroid',
            'GID_0',
            'NAME_0', 
            'Total_Demand_2024_centroid',
            'Total_Demand_2030_centroid',
            'Total_Demand_2050_centroid',
            'geometry'
        ]
        
        # Summary statistics
        print(f"\n=== GLOBAL SUMMARY ===")
        print(f"Total countries: {centroids_data['GID_0'].nunique()}")
        print(f"Total population centroids: {len(centroids_data):,}")
        print(f"Total population (2025 baseline): {centroids_data['Population_centroid'].sum():,.0f}")
        
        # Summary for all three years
        years = [2024, 2030, 2050]
        for year in years:
            demand_col = f"Total_Demand_{year}_centroid"
            if demand_col in centroids_data.columns:
                total_demand = centroids_data[demand_col].sum()
                avg_per_capita = total_demand / centroids_data['Population_centroid'].sum()
                print(f"Global demand {year}: {total_demand:,.0f} MWh ({avg_per_capita:.2f} MWh/capita)")
        
        # Updated column validation
        for col in expected_columns:
            if col not in centroids_data.columns:
                print(f"Warning: Expected column '{col}' not found")
            else:
                print(f"✓ Column '{col}': {centroids_data[col].count():,} non-null values")
        
        # Create summary by country
        country_summary = centroids_data.groupby('GID_0').agg({
            'Population_centroid': ['count', 'sum'],
            'Total_Demand_2030_centroid': 'sum',
            'Total_Demand_2050_centroid': 'sum'
        }).round(0)
        country_summary.columns = ['Num_Centroids', 'Total_Population', 'Demand_2030_MWh', 'Demand_2050_MWh']
        
        # Save summary statistics
        summary_file = Path(output_file).with_suffix('.csv').with_name('global_supply_summary.csv')
        country_summary.to_csv(summary_file)
        
        print(f"\nSaving global dataset to {output_file}...")
        print(f"Global dataset saved to:")
        print(f"  - {output_file} (parquet with geometry)")
        print(f"  - {Path(output_file).with_suffix('.csv')} (CSV without geometry)")
        print(f"  - {summary_file} (summary by country)")
        
        # Print layer outputs
        print(f"\nLayer-based outputs:")
        for layer_name in results.keys():
            print(f"  - global_{layer_name}.gpkg (GIS data)")
            print(f"  - global_{layer_name}.csv (tabular data)")
        
        if results:
            print(f"  - global_supply_analysis_all_layers.gpkg (combined GIS)")
        
        # Global totals
        total_population = centroids_data['Population_centroid'].sum()
        total_demand_2030 = centroids_data['Total_Demand_2030_centroid'].sum()
        
        print(f"\nGlobal totals:")
        print(f"  Total population: {total_population:,.0f}")
        print(f"  Total demand 2030: {total_demand_2030:,.0f} MWh")
        print(f"  Average demand per capita 2030: {total_demand_2030/total_population:.1f} MWh/person")
    
    logger.info(f"Global results combination completed at {datetime.now()}")
    return True

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Combine country supply analysis results')
    parser.add_argument('--input-dir', default='outputs_per_country', help='Input directory with country files')
    parser.add_argument('--output-file', default='outputs_global/global_supply_analysis.parquet', help='Output file name')
    parser.add_argument('--countries-file', help='File with list of countries to process (optional)')
    parser.add_argument('--log-file', default='outputs_global/logs/combine_results.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(Path(args.log_file))
    logger.info("="*60)
    logger.info("STARTING GLOBAL RESULTS COMBINATION")
    logger.info("="*60)
    
    # Load countries list if specified
    countries_list = None
    if args.countries_file and Path(args.countries_file).exists():
        countries_list = [c.strip() for c in Path(args.countries_file).read_text().splitlines() if c.strip()]
        logger.info(f"Processing specific countries from {args.countries_file}: {len(countries_list)} countries")
    
    success = combine_global_results(args.input_dir, args.output_file, countries_list)
    
    if success:
        logger.info("Global results combination completed successfully!")
    else:
        logger.error("Global results combination failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
