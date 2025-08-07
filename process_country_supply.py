#!/usr/bin/env python3
"""
Supply analysis per country - adapted for Snakemake workflow with network analysis
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
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

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

def calculate_centroids_chunk(row_chunk, cols, windowed_transform):
    """Calculate centroids for a chunk of rows"""
    centroids_x = []
    centroids_y = []
    
    for row in row_chunk:
        for col in range(cols):
            # Convert pixel coordinates to geographic coordinates
            x, y = rasterio.transform.xy(windowed_transform, row, col)
            centroids_x.append(x)
            centroids_y.append(y)
    
    return centroids_x, centroids_y

def calculate_nearest_facility_chunk(centroid_chunk, facilities_gdf):
    """Calculate nearest facility for a chunk of centroids"""
    results = []
    
    # Convert to appropriate projected CRS for distance calculations
    # Use UTM zone based on the centroid of the data
    if not centroid_chunk.empty and not facilities_gdf.empty:
        # Get approximate center to determine appropriate UTM zone
        center_lon = centroid_chunk.geometry.centroid.x.mean()
        center_lat = centroid_chunk.geometry.centroid.y.mean()
        
        # Calculate UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
        
        # Project both datasets to UTM for accurate distance calculation
        centroids_utm = centroid_chunk.to_crs(utm_crs)
        facilities_utm = facilities_gdf.to_crs(utm_crs)
    else:
        centroids_utm = centroid_chunk
        facilities_utm = facilities_gdf
    
    for idx, centroid in centroids_utm.iterrows():
        # Find nearest facility using projected coordinates
        distances = facilities_utm.geometry.distance(centroid.geometry)
        if len(distances) > 0:
            nearest_idx = distances.idxmin()
            nearest_facility = facilities_gdf.loc[nearest_idx]  # Use original for attribute access
            
            result = {
                'index': idx,
                'nearest_facility_distance': distances.min(),
                'nearest_facility_type': nearest_facility.get('Grouped_Type', ''),
                'nearest_facility_capacity': nearest_facility.get('Adjusted_Capacity_MW', np.nan),
                'nearest_facility_gem_id': nearest_facility.get('GEM unit/phase ID', '')
            }
        else:
            result = {
                'index': idx,
                'nearest_facility_distance': np.nan,
                'nearest_facility_type': '',
                'nearest_facility_capacity': np.nan,
                'nearest_facility_gem_id': ''
            }
        results.append(result)
    return results

def split_intersecting_edges(lines):
    """Split lines at their intersections and return all unique segments."""
    # Merge all lines
    merged = unary_union(lines)
    
    # Find all intersection points
    if isinstance(merged, (LineString, MultiLineString)):
        # Create a list of all segments
        segments = []
        if isinstance(merged, LineString):
            segments.append(merged)
        else:
            segments.extend(list(merged.geoms))
            
        # Split each segment at intersection points
        final_segments = []
        for segment in segments:
            coords = list(segment.coords)
            for i in range(len(coords) - 1):
                final_segments.append(LineString([coords[i], coords[i + 1]]))
                
        return final_segments
    return []

def connect_points_to_grid_chunk(point_chunk, grid_nodes, point_type):
    """Connect a chunk of points to nearest grid nodes"""
    connections = []
    for point in point_chunk:
        point_coord = (point.x, point.y)
        
        # Find nearest node in the grid network
        if grid_nodes:
            nearest_node = min(grid_nodes, 
                             key=lambda n: Point(n).distance(point))
            
            # Add edge if within distance threshold (10km for centroids)
            distance = Point(nearest_node).distance(point)
            if point_type == 'pop_centroid' and distance > 10000:
                continue
            connections.append((point_coord, nearest_node, distance))
    return connections

def create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf, n_threads=1):
    """Create network graph from facilities, grid lines, and population centroids"""
    # Initialize empty graph
    G = nx.Graph()
    
    # 1. Process grid lines and split at intersections
    single_lines = []
    for _, row in grid_lines_gdf.iterrows():
        if isinstance(row.geometry, MultiLineString):
            single_lines.extend(list(row.geometry.geoms))
        else:
            single_lines.append(row.geometry)
    
    # Split lines at intersections
    split_lines = split_intersecting_edges(single_lines)
    print(f"Original line count: {len(single_lines)}, After splitting: {len(split_lines)}")
    
    # 2. Create nodes at all line endpoints and intersections
    nodes = set()
    for line in split_lines:
        coords = list(line.coords)
        nodes.add(coords[0])
        nodes.add(coords[-1])
    
    # 3. Add facility and centroid nodes
    facility_nodes = set((point.x, point.y) for point in facilities_gdf.geometry)
    pop_centroid_nodes = set((point.x, point.y) for point in centroids_gdf.geometry)
    
    # 4. Add all nodes to the graph with their types
    for node in nodes:
        G.add_node(node, pos=node, type='grid_line')
    for node in facility_nodes:
        G.add_node(node, pos=node, type='facility')
    for node in pop_centroid_nodes:
        G.add_node(node, pos=node, type='pop_centroid')
    
    # 5. Create edges from split lines
    for line in split_lines:
        coords = list(line.coords)
        G.add_edge(coords[0], coords[-1], weight=line.length)
    
    # 6. Connect facilities and centroids to nearest grid nodes
    grid_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'grid_line']
    
    if n_threads > 1 and (len(facilities_gdf) + len(centroids_gdf)) > 100:
        print(f"Using {n_threads} threads for network connections...")
        
        # Process facilities and centroids in parallel
        for point_gdf, point_type in [(facilities_gdf, 'facility'), (centroids_gdf, 'pop_centroid')]:
            if len(point_gdf) > 0:
                # Split points into chunks
                points = list(point_gdf.geometry)
                chunk_size = max(10, len(points) // n_threads)
                chunks = [points[i:i+chunk_size] for i in range(0, len(points), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=n_threads) as executor:
                    connection_results = list(executor.map(
                        partial(connect_points_to_grid_chunk, grid_nodes=grid_nodes, point_type=point_type),
                        chunks
                    ))
                
                # Add connections to graph
                for chunk_connections in connection_results:
                    for point_coord, nearest_node, distance in chunk_connections:
                        G.add_edge(point_coord, nearest_node, weight=distance)
    else:
        # Serial processing
        for point_gdf, point_type in [(facilities_gdf, 'facility'), (centroids_gdf, 'pop_centroid')]:
            for point in point_gdf.geometry:
                point_coord = (point.x, point.y)
                
                # Find nearest node in the grid network
                if grid_nodes:
                    nearest_node = min(grid_nodes, 
                                     key=lambda n: Point(n).distance(point))
                    
                    # Add edge if within distance threshold (10km for centroids)
                    distance = Point(nearest_node).distance(point)
                    if point_type == 'pop_centroid' and distance > 10000:
                        continue
                    G.add_edge(point_coord, nearest_node, weight=distance)
    
    return G

def load_energy_facilities(country_iso3, year=2024):
    """Load energy facilities data for a specific country and year"""
    try:
        # Define sheet names for different years
        sheet_mapping = {
            2024: 'Grouped_cur_fac_lvl',
            2030: 'Grouped_2030_fac_lvl', 
            2050: 'Grouped_2050_fac_lvl'
        }
        
        sheet_name = sheet_mapping.get(year, 'Grouped_cur_fac_lvl')
        
        # Load facilities data
        facilities_df = pd.read_excel("outputs_processed_data/p1_a_ember_gem_2024_fac_lvl.xlsx", 
                                    sheet_name=sheet_name)
        
        # Filter for the specified country
        country_facilities = facilities_df[facilities_df['Country Code'] == country_iso3].copy()
        
        if country_facilities.empty:
            print(f"No facilities found for {country_iso3} in {year}")
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame from facilities
        geometry = gpd.points_from_xy(country_facilities['Longitude'], country_facilities['Latitude'])
        facilities_gdf = gpd.GeoDataFrame(country_facilities, geometry=geometry, crs="EPSG:4326")
        
        print(f"Loaded {len(facilities_gdf)} facilities for {country_iso3} in {year}")
        return facilities_gdf
        
    except Exception as e:
        print(f"Error loading facilities data for {country_iso3}: {e}")
        return gpd.GeoDataFrame()

def load_grid_lines(country_bbox, country_boundaries):
    """Load and clip grid lines for the country"""
    try:
        # Load global grid data
        grid_lines = gpd.read_file('bigdata_gridfinder/grid.gpkg')
        
        # Clip to country bounding box first for performance
        minx, miny, maxx, maxy = country_bbox
        bbox_geom = gpd.GeoDataFrame([1], geometry=[
            Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        ], crs="EPSG:4326")
        
        grid_clipped = gpd.clip(grid_lines, bbox_geom)
        
        # Further clip to exact country boundaries
        if not grid_clipped.empty:
            grid_country = gpd.clip(grid_clipped, country_boundaries)
            print(f"Loaded {len(grid_country)} grid line segments for country")
            return grid_country
        else:
            print("No grid lines found in country area")
            return gpd.GeoDataFrame()
            
    except Exception as e:
        print(f"Error loading grid data: {e}")
        return gpd.GeoDataFrame()

def process_country_supply(country_iso3, output_dir="outputs_per_country", n_threads=1):
    """Process supply analysis for a single country"""
    print(f"Processing country: {country_iso3} with {n_threads} threads")
    
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
            
            # Get dimensions
            rows, cols = pop_data.shape
            
            # Calculate centroids for each cell
            print(f"Calculating cell centroids for {rows}x{cols} grid using {n_threads} threads...")
            
            if n_threads > 1 and rows > 100:
                # Parallel processing for large grids
                chunk_size = max(10, rows // n_threads)
                row_chunks = [list(range(i, min(i + chunk_size, rows))) for i in range(0, rows, chunk_size)]
                
                with ThreadPoolExecutor(max_workers=n_threads) as executor:
                    centroid_results = list(executor.map(
                        partial(calculate_centroids_chunk, cols=cols, windowed_transform=windowed_transform),
                        row_chunks
                    ))
                
                # Combine results
                centroids_x = []
                centroids_y = []
                for chunk_x, chunk_y in centroid_results:
                    centroids_x.extend(chunk_x)
                    centroids_y.extend(chunk_y)
            else:
                # Serial processing for small grids
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
    
    # Load country-level population projections from p1_b dataset
    print("Loading country-level population projections...")
    try:
        pop_projections_df = pd.read_excel("outputs_processed_data/p1_b_ember_2024_30_50.xlsx")
        
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
        supply_df = pd.read_excel("outputs_processed_data/p1_b_ember_2024_30_50.xlsx")

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
    
    # === NETWORK ANALYSIS ===
    print(f"\n=== NETWORK ANALYSIS for {country_iso3} ===")
    
    # Load energy facilities for 2024 (baseline year)
    facilities_gdf = load_energy_facilities(country_iso3, year=2024)
    
    # Load grid lines
    grid_lines_gdf = load_grid_lines(country_bbox, admin_boundaries)
    
    # Perform network analysis only if we have both facilities and grid data
    network_results = {}
    if not facilities_gdf.empty and not grid_lines_gdf.empty:
        try:
            print("Creating network graph...")
            network_graph = create_network_graph(facilities_gdf, grid_lines_gdf, centroids_filtered, n_threads)
            
            print(f"Network created with {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges")
            
            # Calculate basic network metrics
            network_results = {
                'total_nodes': network_graph.number_of_nodes(),
                'total_edges': network_graph.number_of_edges(),
                'grid_nodes': len([n for n, d in network_graph.nodes(data=True) if d['type'] == 'grid_line']),
                'facility_nodes': len([n for n, d in network_graph.nodes(data=True) if d['type'] == 'facility']),
                'centroid_nodes': len([n for n, d in network_graph.nodes(data=True) if d['type'] == 'pop_centroid'])
            }
            
            print(f"Network composition:")
            print(f"  Grid nodes: {network_results['grid_nodes']}")
            print(f"  Facility nodes: {network_results['facility_nodes']}")
            print(f"  Population centroid nodes: {network_results['centroid_nodes']}")
            
            # Add facility information to centroids if available
            if len(facilities_gdf) > 0:
                # Calculate distance to nearest facility for each centroid
                centroids_filtered['nearest_facility_distance'] = np.nan
                centroids_filtered['nearest_facility_type'] = ''
                centroids_filtered['nearest_facility_capacity'] = np.nan
                centroids_filtered['nearest_facility_gem_id'] = ''
                
                print(f"Calculating nearest facilities for {len(centroids_filtered)} centroids using {n_threads} threads...")
                
                if n_threads > 1 and len(centroids_filtered) > 100:
                    # Parallel processing for large datasets
                    chunk_size = max(10, len(centroids_filtered) // n_threads)
                    chunks = [centroids_filtered.iloc[i:i+chunk_size] for i in range(0, len(centroids_filtered), chunk_size)]
                    
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        chunk_results = list(executor.map(
                            partial(calculate_nearest_facility_chunk, facilities_gdf=facilities_gdf),
                            chunks
                        ))
                    
                    # Combine results
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            idx = result['index']
                            centroids_filtered.loc[idx, 'nearest_facility_distance'] = result['nearest_facility_distance']
                            centroids_filtered.loc[idx, 'nearest_facility_type'] = result['nearest_facility_type']
                            centroids_filtered.loc[idx, 'nearest_facility_capacity'] = result['nearest_facility_capacity']
                            centroids_filtered.loc[idx, 'nearest_facility_gem_id'] = result['nearest_facility_gem_id']
                else:
                    # Serial processing for small datasets or single thread
                    # Convert to appropriate projected CRS for distance calculations
                    if not centroids_filtered.empty and not facilities_gdf.empty:
                        # Get approximate center to determine appropriate UTM zone
                        center = centroids_filtered.geometry.unary_union.centroid
                        center_lon = center.x
                        center_lat = center.y
                        # Calculate UTM zone
                        utm_zone = int((center_lon + 180) / 6) + 1
                        utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
                        # Project both datasets to UTM for accurate centroid and distance calculation
                        centroids_utm = centroids_filtered.to_crs(utm_crs)
                        facilities_utm = facilities_gdf.to_crs(utm_crs)
                        # Calculate centroids in projected CRS
                        centroids_utm['projected_centroid'] = centroids_utm.geometry.centroid
                        for idx, centroid in centroids_utm.iterrows():
                            # Find nearest facility using projected coordinates
                            distances = facilities_utm.geometry.distance(centroid['projected_centroid'])
                            if len(distances) > 0:
                                nearest_idx = distances.idxmin()
                                nearest_facility = facilities_gdf.loc[nearest_idx]  # Use original for attributes
                                centroids_filtered.loc[idx, 'nearest_facility_distance'] = distances.min()
                                centroids_filtered.loc[idx, 'nearest_facility_type'] = nearest_facility.get('Grouped_Type', '')
                                centroids_filtered.loc[idx, 'nearest_facility_capacity'] = nearest_facility.get('Adjusted_Capacity_MW', np.nan)
                                centroids_filtered.loc[idx, 'nearest_facility_gem_id'] = nearest_facility.get('GEM unit/phase ID', '')
                    else:
                        # Fallback for empty datasets
                        for idx, centroid in centroids_filtered.iterrows():
                            centroids_filtered.loc[idx, 'nearest_facility_distance'] = np.nan
                            centroids_filtered.loc[idx, 'nearest_facility_type'] = ''
                            centroids_filtered.loc[idx, 'nearest_facility_capacity'] = np.nan
                            centroids_filtered.loc[idx, 'nearest_facility_gem_id'] = ''
                
                print(f"Added facility proximity information to centroids")
            
        except Exception as e:
            print(f"Warning: Network analysis failed: {e}")
            network_results = {}
    else:
        if facilities_gdf.empty:
            print("No energy facilities found for network analysis")
        if grid_lines_gdf.empty:
            print("No grid lines found for network analysis")
    
    # Save network analysis results
    if network_results:
        network_summary_file = output_path / f"network_summary_{country_iso3}.txt"
        with open(network_summary_file, 'w') as f:
            f.write(f"Network Analysis Summary for {country_iso3}\n")
            f.write("=" * 50 + "\n")
            for key, value in network_results.items():
                f.write(f"{key}: {value}\n")
        print(f"Network summary saved to {network_summary_file}")
    
    # Keep the specified columns for simplified output - now including network analysis results
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
        'Total_Demand_2050_centroid',
        'nearest_facility_distance',    # Distance to nearest energy facility
        'nearest_facility_type',        # Type of nearest facility (e.g., Solar, Wind, Fossil)
        'nearest_facility_capacity',    # Capacity of nearest facility in MW
        'nearest_facility_gem_id'       # GEM unit/phase ID of nearest facility
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
    
    # Print summary statistics - now including population projections and network analysis
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
    
    # Network analysis summary
    if 'nearest_facility_distance' in centroids_simplified.columns:
        facility_data = centroids_simplified.dropna(subset=['nearest_facility_distance'])
        if len(facility_data) > 0:
            print(f"\nNetwork Analysis Results:")
            print(f"Centroids with facility access: {len(facility_data):,} out of {len(centroids_simplified):,}")
            print(f"Average distance to nearest facility: {facility_data['nearest_facility_distance'].mean():.2f} degrees")
            print(f"Median distance to nearest facility: {facility_data['nearest_facility_distance'].median():.2f} degrees")
            
            # Facility type breakdown
            if 'nearest_facility_type' in centroids_simplified.columns:
                facility_types = facility_data['nearest_facility_type'].value_counts()
                print(f"Nearest facility types:")
                for ftype, count in facility_types.items():
                    if ftype:  # Skip empty strings
                        print(f"  {ftype}: {count} centroids")
    
    if network_results:
        print(f"\nNetwork Statistics:")
        print(f"Total network nodes: {network_results.get('total_nodes', 'N/A')}")
        print(f"Total network edges: {network_results.get('total_edges', 'N/A')}")
        print(f"Energy facilities in network: {network_results.get('facility_nodes', 'N/A')}")
    
    return str(output_file)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Process supply analysis for a country')
    parser.add_argument('country_iso3', help='ISO3 country code')
    parser.add_argument('--output-dir', default='outputs_per_country', help='Output directory')
    parser.add_argument('--no-network', action='store_true', help='Skip network analysis')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for processing')
    
    args = parser.parse_args()
    
    # Set the number of threads for parallel processing
    if args.threads > 1:
        print(f"Using {args.threads} threads for parallel processing")
        # Configure parallel processing libraries
        import os
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads)
        
        # Set pandas and numpy to use multiple threads
        try:
            import numba
            numba.set_num_threads(args.threads)
        except ImportError:
            pass
    
    result = process_country_supply(args.country_iso3, args.output_dir, n_threads=args.threads)
    if result:
        print(f"Successfully processed {args.country_iso3}")
        return 0
    else:
        print(f"Failed to process {args.country_iso3}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
