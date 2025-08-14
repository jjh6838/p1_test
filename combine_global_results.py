#!/usr/bin/env python3
"""
Combine all country supply analysis results into global dataset
Production mode only - processes Parquet files for efficient global analysis
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

def find_parquet_files(input_dir, layer_name):
    """Find all parquet files for a specific layer across all countries"""
    input_path = Path(input_dir)
    parquet_files = []
    
    # Look for parquet files in the main parquet directory
    parquet_dir = input_path / "parquet"
    if parquet_dir.exists():
        pattern = f"{layer_name}_*.parquet"
        matching_files = list(parquet_dir.glob(pattern))
        parquet_files.extend(matching_files)
    
    return parquet_files

def combine_layer_data(input_dir, layer_name, countries_list=None):
    """Combine data from a specific layer across all countries (Parquet files only)"""
    
    input_path = Path(input_dir)
    
    # Find all parquet files for this layer
    parquet_files = find_parquet_files(input_path, layer_name)
    
    if countries_list:
        # Filter to only specified countries
        filtered_files = []
        for file_path in parquet_files:
            country_code = file_path.stem.split('_')[-1]
            if country_code in countries_list:
                filtered_files.append(file_path)
        parquet_files = filtered_files
    
    if not parquet_files:
        logging.warning(f"No Parquet files found for layer {layer_name}")
        return None
    
    logging.info(f"Combining {layer_name} layer from {len(parquet_files)} Parquet files")
    
    # Read and combine data from this layer
    layer_data_list = []
    failed_countries = []
    
    for file_path in tqdm(parquet_files, desc=f"Processing {layer_name}"):
        try:
            country_code = file_path.stem.split('_')[-1]
            
            # Read the parquet file using geopandas for proper geometry handling
            try:
                layer_data = gpd.read_parquet(file_path)
                logging.info(f"{layer_name} from {country_code} loaded as GeoDataFrame with geometry")
            except Exception:
                # Fallback to pandas if geopandas fails
                layer_data = pd.read_parquet(file_path)
                # Check if there's a geometry column and convert it
                if 'geometry' in layer_data.columns:
                    from shapely import wkt
                    # Try to convert from WKT if stored as text
                    try:
                        layer_data['geometry'] = layer_data['geometry'].apply(wkt.loads)
                        layer_data = gpd.GeoDataFrame(layer_data, crs='EPSG:4326')
                        logging.info(f"{layer_name} from {country_code} converted from WKT to GeoDataFrame")
                    except Exception:
                        logging.info(f"{layer_name} from {country_code} has geometry column but conversion failed")
                else:
                    logging.info(f"{layer_name} from {country_code} has no geometry column (production mode)")
            
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
    
    # Only ensure it's a GeoDataFrame if it has geometry
    if 'geometry' in combined_data.columns and not combined_data['geometry'].isna().all():
        combined_data = gpd.GeoDataFrame(combined_data, crs='EPSG:4326')
        logging.info(f"Combined {layer_name}: {len(combined_data)} total records (with geometry)")
    else:
        logging.info(f"Combined {layer_name}: {len(combined_data)} total records (tabular data only)")
    
    if failed_countries:
        logging.warning(f"Failed to process {layer_name} from {len(failed_countries)} countries: {failed_countries}")
    
    return combined_data

def generate_global_summary(combined_layers, logger):
    """Generate and log global summary statistics"""
    logger.info("\n" + "="*80)
    logger.info("GLOBAL SUPPLY ANALYSIS SUMMARY")
    logger.info("="*80)
    
    # Get centroids and facilities data
    centroids_gdf = combined_layers.get('centroids')
    facilities_gdf = combined_layers.get('facilities')
    
    if centroids_gdf is not None:
        # Country count
        countries = centroids_gdf['GID_0'].nunique()
        logger.info(f"Countries processed: {countries}")
        
        # Supply status summary
        if 'supply_status' in centroids_gdf.columns:
            status_counts = centroids_gdf['supply_status'].value_counts()
            total_centroids = len(centroids_gdf)
            
            logger.info(f"\nGlobal centroid supply status:")
            logger.info(f"  Total centroids: {total_centroids:,}")
            for status in ['Filled', 'Partially Filled', 'Not Filled', 'No Demand']:
                count = status_counts.get(status, 0)
                pct = (count / total_centroids * 100) if total_centroids > 0 else 0
                logger.info(f"  {status}: {count:,} centroids ({pct:.1f}%)")
        
        # Facility type summary
        if 'supplying_facility_type' in centroids_gdf.columns:
            # Parse comma-separated facility types
            all_types = []
            for types_str in centroids_gdf['supplying_facility_type'].dropna():
                if types_str and types_str != '':
                    types = [t.strip() for t in str(types_str).split(',')]
                    all_types.extend(types)
            
            if all_types:
                type_counts = pd.Series(all_types).value_counts()
                logger.info(f"\nSupplying facility types:")
                for ftype, count in type_counts.head(10).items():
                    if ftype:  # Skip empty strings
                        logger.info(f"  {ftype}: {count:,} connections")
    
    if facilities_gdf is not None:
        # Facility summary
        facility_countries = facilities_gdf['GID_0'].nunique()
        total_facilities = len(facilities_gdf)
        logger.info(f"\nGlobal facilities:")
        logger.info(f"  Countries with facilities: {facility_countries}")
        logger.info(f"  Total facilities: {total_facilities:,}")
        
        if 'Grouped_Type' in facilities_gdf.columns:
            facility_type_counts = facilities_gdf['Grouped_Type'].value_counts()
            logger.info(f"  Facility types:")
            for ftype, count in facility_type_counts.items():
                logger.info(f"    {ftype}: {count:,} facilities")
        
        if 'total_mwh' in facilities_gdf.columns:
            total_capacity = facilities_gdf['total_mwh'].sum()
            logger.info(f"  Total global capacity: {total_capacity:,.0f} MWh/year")
    
    # Layer summary
    logger.info(f"\nLayers successfully combined:")
    for layer_name, layer_data in combined_layers.items():
        logger.info(f"  {layer_name}: {len(layer_data):,} records")
    
    logger.info("="*80)

def combine_global_results(input_dir="outputs_per_country", output_file="global_supply_analysis.gpkg", 
                         countries_list=None):
    """
    Combine all country results into global datasets by layer (Parquet input mode)
    
    Args:
        input_dir: Directory containing country subdirectories with Parquet files
        output_file: Output file (supports .gpkg, .parquet extensions)
        countries_list: Optional list of countries to include
    """
    
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
    
    # Check for Parquet files
    parquet_dir = input_path / "parquet"
    has_parquet = parquet_dir.exists() and list(parquet_dir.glob("*.parquet"))
    
    if not has_parquet:
        logger.error(f"No Parquet files found in {parquet_dir}")
        logger.error("Expected structure: outputs_per_country/parquet/layer_ISO3.parquet")
        return False
    
    # Determine output format from file extension
    output_format = 'gpkg' if output_file.endswith('.gpkg') else 'parquet'
    
    logger.info(f"Processing Parquet files, Output format: {output_format}")
    
    # Define layers to combine
    layers_to_combine = ['centroids', 'facilities', 'grid_lines', 'polylines']
    
    # Combine each layer
    combined_layers = {}
    
    for layer_name in layers_to_combine:
        logger.info(f"\n--- Processing {layer_name} layer ---")
        
        combined_data = combine_layer_data(
            input_dir=input_dir,
            layer_name=layer_name,
            countries_list=countries_list
        )
        
        if combined_data is not None and not combined_data.empty:
            combined_layers[layer_name] = combined_data
            logger.info(f"Successfully combined {layer_name}: {len(combined_data)} records")
        else:
            logger.warning(f"No data available for {layer_name} layer")
    
    if not combined_layers:
        logger.error("No layers were successfully combined")
        return False
    
    # Save combined data based on output format
    if output_format == 'gpkg':
        # Save as GPKG with multiple layers (only for data with geometry)
        logger.info(f"Saving global analysis as GPKG: {output_path}")
        
        layers_with_geometry = {}
        layers_without_geometry = {}
        
        for layer_name, layer_data in combined_layers.items():
            logger.info(f"Processing {layer_name}: {len(layer_data)} records")
            
            # Check if it has geometry and is a GeoDataFrame
            if isinstance(layer_data, gpd.GeoDataFrame) and 'geometry' in layer_data.columns:
                layers_with_geometry[layer_name] = layer_data
                layer_data.to_file(output_path, layer=layer_name, driver="GPKG")
                logger.info(f"Saved {layer_name} to GPKG (with geometry)")
            else:
                layers_without_geometry[layer_name] = layer_data
                # Save as CSV for non-geometry data
                csv_file = output_dir / f"global_{layer_name}.csv"
                layer_data.to_csv(csv_file, index=False)
                logger.info(f"Saved {layer_name} to CSV (tabular data): {csv_file.name}")
        
        if layers_with_geometry:
            logger.info(f"Global GPKG saved with {len(layers_with_geometry)} geometry layers")
        if layers_without_geometry:
            logger.info(f"Additional {len(layers_without_geometry)} layers saved as CSV files")
    
    else:
        # Save as individual Parquet files
        logger.info(f"Saving global analysis as Parquet files in: {output_dir}")
        
        for layer_name, layer_data in combined_layers.items():
            parquet_file = output_dir / f"global_{layer_name}.parquet"
            layer_data.to_parquet(parquet_file)
            logger.info(f"Saved {layer_name}: {len(layer_data)} records → {parquet_file.name}")
    
    # Generate global summary
    generate_global_summary(combined_layers, logger)
    
    logger.info(f"Global combination completed successfully at {datetime.now()}")
    return True

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Combine country supply analysis results into global dataset (Production mode - Parquet files)')
    parser.add_argument('--input-dir', default='outputs_per_country', 
                       help='Input directory with country subdirectories containing Parquet files')
    parser.add_argument('--output', default='global_supply_analysis.gpkg', 
                       help='Output file (.gpkg for visualization, .parquet for data)')
    parser.add_argument('--countries-file', 
                       help='File with list of countries to process (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("STARTING GLOBAL RESULTS COMBINATION (PRODUCTION MODE)")
    logger.info("="*60)
    
    # Load countries list if specified
    countries_list = None
    if args.countries_file and Path(args.countries_file).exists():
        countries_list = [c.strip() for c in Path(args.countries_file).read_text().splitlines() if c.strip()]
        logger.info(f"Processing specific countries from {args.countries_file}: {len(countries_list)} countries")
    
    success = combine_global_results(
        input_dir=args.input_dir,
        output_file=args.output,
        countries_list=countries_list
    )
    
    if success:
        logger.info("Global results combination completed successfully!")
        print(f"\n🎉 Success! Global analysis saved to: {args.output}")
        if args.output.endswith('.gpkg'):
            print("💡 You can now open this GPKG file in QGIS for global visualization!")
    else:
        logger.error("Global results combination failed!")
        print("\n❌ Failed to create global analysis. Check the logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
