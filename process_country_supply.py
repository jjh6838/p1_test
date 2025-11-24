#!/usr/bin/env python3
"""
# ==================================================================================================
# SCRIPT PURPOSE: Detailed Country-Level Electricity Supply and Demand Network Analysis
# ==================================================================================================
#
# WHAT THIS CODE DOES:
# This script performs a granular, network-based analysis of electricity supply and demand for a
# single country. It models how electricity flows from generation facilities (supply) to population
# centers (demand) through the electrical grid.
#
# WHY THIS IS NEEDED:
# While other scripts project national totals, this script provides spatial detail. It helps answer:
# - Which population centers are served by which power plants?
# - What are the network distances for electricity transmission?
# - Are there parts of the grid that are critical for connecting supply and demand?
# - Which areas might be underserved based on network topology?
#
# KEY STEPS:
# 1. Load Geospatial Data: Imports administrative boundaries, population density rasters,
#    power plant locations, and electrical grid lines for the specified country.
# 2. Project Demand: Allocates national-level electricity demand projections (for 2024, 2030, 2050)
#    down to fine-grained population centroids.
# 3. Build Network Graph: Constructs a comprehensive `networkx` graph representing the entire
#    electrical network, including grid lines, facilities, and centroids as nodes. It also
#    "stitches" together disconnected parts of the grid.
# 4. Calculate Distances: Computes the shortest path network distance from every population
#    centroid to all nearby power plants. This is a computationally intensive step that uses
#    parallel processing.
# 5. Allocate Supply: Implements a supply-demand matching algorithm. It iterates through power
#    plants (largest first) and allocates their electricity to the nearest population centers
#    until the plant's capacity is exhausted or the centers' demands are met.
#
# OUTPUTS:
# - Test Mode: A single GeoPackage (.gpkg) file containing detailed layers for centroids,
#   facilities, grid lines, and the specific polylines of active supply routes.
# - Production Mode: Lightweight Parquet files for each layer, optimized for large-scale analysis.
# ==================================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import networkx as nx
from pathlib import Path
import argparse
import sys
import os
import math
from affine import Affine
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
import time
from contextlib import contextmanager

# Suppress warnings
# warnings.filterwarnings("ignore")

# Constants
COMMON_CRS = "EPSG:4326"  # WGS84 for input/output
YEARS = [2024, 2030, 2050]
DEMAND_TYPES = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]
POP_AGGREGATION_FACTOR = 10  # Aggregate from native 30" grid to 300" (~10x10 pixels)

# Get optimal number of workers based on available CPUs
MAX_WORKERS = min(72, max(1, os.cpu_count() or 1))
print(f"Parallel processing configured for {MAX_WORKERS} workers")

# Cache for storing calculated paths to avoid re-computing shortest paths for the same node pairs
path_cache = {}

@contextmanager
def timer(name):
    """Simple timer context manager"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"  [{name}]: {elapsed:.2f}s")

def get_utm_crs(lon, lat):
    """Get appropriate UTM CRS for given coordinates. UTM is essential for accurate distance calculations in meters."""
    utm_zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone}" if lat >= 0 else f"EPSG:{32700 + utm_zone}"

def get_country_bbox(admin_boundaries, buffer=0.1):
    """Get bounding box for a country with buffer"""
    bounds = admin_boundaries.total_bounds
    return [bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer]

def aggregate_raster(data, transform, factor):
    """Aggregate a raster array by the provided factor using block sums."""
    if factor <= 1:
        return data, transform

    rows, cols = data.shape
    new_rows = math.ceil(rows / factor)
    new_cols = math.ceil(cols / factor)

    pad_rows = new_rows * factor - rows
    pad_cols = new_cols * factor - cols
    if pad_rows or pad_cols:
        data = np.pad(data, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

    reshaped = data.reshape(new_rows, factor, new_cols, factor)
    aggregated = reshaped.sum(axis=(1, 3))
    new_transform = transform * Affine.scale(factor, factor)

    return aggregated, new_transform

def load_admin_boundaries(country_iso3):
    """Load administrative boundaries for a specific country from the GADM dataset."""
    admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410-levels.gpkg', layer="ADM_0")
    country_data = admin_boundaries[admin_boundaries['GID_0'] == country_iso3]
    
    if country_data.empty:
        raise ValueError(f"No boundaries found for country {country_iso3}")
    
    return country_data

def load_population_centroids(country_bbox, admin_boundaries):
    """Load and process population centroids from the GHS-POP raster data.
    This is optimized to only process pixels with non-zero population, which significantly speeds up processing for sparsely populated areas.
    """
    minx, miny, maxx, maxy = country_bbox
    # Source: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
    # Select: 2025 eporch, 30 arcsec, WGS84(4326)

    with rasterio.open('bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif') as src:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        pop_data = src.read(1, window=window)
        windowed_transform = rasterio.windows.transform(window, src.transform)

        # Aggregate 30" raster to 300" (~10x10) blocks for performance smoother centroids
        pop_data, windowed_transform = aggregate_raster(pop_data, windowed_transform, POP_AGGREGATION_FACTOR)
        print(f"Aggregated population raster to {POP_AGGREGATION_FACTOR * 30}\" resolution")

        # OPTIMIZATION: Only process non-zero pixels
        nonzero_mask = pop_data > 0
        nonzero_indices = np.where(nonzero_mask)

        total_pixels = pop_data.size
        nonzero_pixels = len(nonzero_indices[0])
        print(f"Population raster: {nonzero_pixels:,} populated pixels out of {total_pixels:,} total ({nonzero_pixels/total_pixels*100:.1f}%)")
        
        if nonzero_pixels == 0:
            print("No population in this area")
            return gpd.GeoDataFrame(geometry=[], crs=COMMON_CRS)
        
        # Convert row/col indices to coordinates
        rows = nonzero_indices[0]
        cols = nonzero_indices[1]
        
        # Transform pixel coordinates to geographic coordinates
        xs, ys = rasterio.transform.xy(windowed_transform, rows, cols)
        
        # Create GeoDataFrame directly
        centroids_gdf = gpd.GeoDataFrame(
            {
                'geometry': gpd.points_from_xy(xs, ys),
                'Population_centroid': pop_data[nonzero_mask]
            },
            crs=COMMON_CRS
        )
        
        # Filter to country boundaries
        centroids_gdf = gpd.sjoin(centroids_gdf, admin_boundaries, how='inner', predicate='within')
        centroids_gdf = centroids_gdf.drop(columns=['index_right'])
        
        print(f"Loaded {len(centroids_gdf):,} population centroids within country boundaries")
        return centroids_gdf

def load_population_and_demand_projections(centroids_gdf, country_iso3):
    """
    Loads country-level population and demand projections for 2024, 2030, and 2050.
    It then allocates these national totals down to the individual population centroids based on their baseline population share.
    This assumes the spatial distribution of population remains constant, while the total numbers change.
    """
    print("Loading country-level population projections...")
    
    # Calculate baseline 2025 population from spatial data
    total_country_population_2025 = centroids_gdf["Population_centroid"].sum()
    print(f"Baseline population from spatial data (JRC 2025): {total_country_population_2025:,.0f}")

    # Initialize population projections (will be overwritten if data is found)
    pop_2024 = 0
    pop_2030 = 0
    pop_2050 = 0
    used_baseline = False
    
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
            used_baseline = False 
            
            print(f"Country population projections:")
            print(f"  2024: {pop_2024:,.0f}")
            print(f"  2030: {pop_2030:,.0f}")
            print(f"  2050: {pop_2050:,.0f}")
            
    except Exception as e:
        print(f"Warning: Could not load population projections for {country_iso3}: {e}")
        used_baseline = True 

    if used_baseline:
        print("Warning: Need to review p1_b_ember_2024_30_50.xlsx to add missing population data")

    # Load national demand data for this specific country
    print("Loading demand projections...")
    try:
        demand_df = pd.read_excel("outputs_processed_data/p1_b_ember_2024_30_50.xlsx")

        # Filter for this country using ISO3_code column
        if 'ISO3_code' in demand_df.columns:
            country_demand = demand_df[demand_df['ISO3_code'] == country_iso3]
            if country_demand.empty:
                print(f"Warning: No demand data found for {country_iso3} in demand file")
                demand_df = pd.DataFrame()
            else:
                demand_df = country_demand
                print(f"Loaded demand data for {country_iso3}: {len(demand_df)} records")
        else:
            print(f"Warning: ISO3_code column not found in demand data")
            demand_df = pd.DataFrame()
            
    except Exception as e:
        print(f"Warning: Could not load demand data for {country_iso3}: {e}")
        demand_df = pd.DataFrame()

    # Define demand types and years
    demand_types = DEMAND_TYPES
    years = [2024, 2030, 2050]
    country_populations = {2024: pop_2024, 2030: pop_2030, 2050: pop_2050}

    # First, calculate population share for each centroid based on JRC spatial population share
    population_share = centroids_gdf["Population_centroid"] / total_country_population_2025

    # Calculate spatially distributed population for each year
    print("\nAllocating population projections to centroids...")
    for year in years:
        total_country_population_year = country_populations[year]
        
        # Allocate projected population to each centroid using JRC spatial population share
        pop_col = f"Population_{year}_centroid"
        centroids_gdf[pop_col] = population_share * total_country_population_year
        
        print(f"  {year} population allocated: {centroids_gdf[pop_col].sum():,.0f} total")

    # Calculate total demand for each centroid for each year
    for year in years:
        print(f"\nProcessing energy demand for year {year}...")
        
        # Get country-level population for this year
        total_country_population_year = country_populations[year]
        
        # Calculate total national demand for this year (projected generation)
        total_national_demand = 0
        if not demand_df.empty:
            for demand_type in demand_types:
                col = f"{demand_type}_{year}_MWh"
                if col in demand_df.columns:
                    demand_value = demand_df[col].iloc[0] if not pd.isna(demand_df[col].iloc[0]) else 0
                    total_national_demand += demand_value
        
        # If no demand data, use a default per capita value
        if total_national_demand == 0:
            # Default: 0 MWh per person per year (Can put global average 3.2 MWh per person per year if needed but all data have been calculated)
            total_national_demand = total_country_population_year * 0
            print(f"Using default demand for {country_iso3} in {year}: {total_national_demand:,.0f} MWh")
        else:
            print(f"Total national demand for {year}: {total_national_demand:,.0f} MWh")
        
        # Calculate centroid demand on a prorata basis
        # Demand_centroid = Population_Share_centroid × National_Demand_year
        demand_col = f"Total_Demand_{year}_centroid"
        centroids_gdf[demand_col] = population_share * total_national_demand
        
        print(f"  Allocated {total_national_demand:,.0f} MWh across {len(centroids_gdf)} centroids")
        print(f"  Per capita demand: {total_national_demand/total_country_population_year:.2f} MWh/person/year")

    # Filter out centroids with zero population
    centroids_filtered = centroids_gdf[centroids_gdf["Population_centroid"] > 0].copy()
    
    print(f"\nFiltered centroids: {len(centroids_filtered)} with population > 0")
    
    return centroids_filtered

def load_energy_facilities(country_iso3, year=2024):
    """Load energy facilities for a specific country and year from the processed facility-level data."""
    sheet_mapping = {2024: 'Grouped_cur_fac_lvl', 2030: 'Grouped_2030_fac_lvl', 2050: 'Grouped_2050_fac_lvl'}
    sheet_name = sheet_mapping.get(year, 'Grouped_cur_fac_lvl')
    
    try:
        facilities_df = pd.read_excel("outputs_processed_data/p1_a_ember_gem_2024_fac_lvl.xlsx", sheet_name=sheet_name)
        country_facilities = facilities_df[facilities_df['Country Code'] == country_iso3].copy()
        
        if country_facilities.empty:
            return gpd.GeoDataFrame()
        
        geometry = gpd.points_from_xy(country_facilities['Longitude'], country_facilities['Latitude'])
        facilities_gdf = gpd.GeoDataFrame(country_facilities, geometry=geometry, crs=COMMON_CRS)
        
        print(f"Loaded {len(facilities_gdf)} facilities for {country_iso3}")
        return facilities_gdf
        
    except Exception as e:
        print(f"Error loading facilities: {e}")
        return gpd.GeoDataFrame()

def load_grid_lines(country_bbox, admin_boundaries):
    """Load and clip grid lines from GridFinder data. Uses parallel processing for very large countries to speed up clipping."""
    try:
        grid_lines = gpd.read_file('bigdata_gridfinder/grid.gpkg') # updated on November 23, 2025
        minx, miny, maxx, maxy = country_bbox
        
        # Initial bbox filter
        grid_lines_filtered = grid_lines.cx[minx:maxx, miny:maxy]
        
        if len(grid_lines_filtered) > 10000:
            print(f"Large grid dataset ({len(grid_lines_filtered)} lines) - using parallel clipping")
            
            # Split into chunks for parallel processing
            chunk_size = 1000
            chunks = [grid_lines_filtered.iloc[i:i+chunk_size] 
                     for i in range(0, len(grid_lines_filtered), chunk_size)]
            
            def clip_chunk(chunk):
                return gpd.clip(chunk, admin_boundaries)
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                clipped_chunks = list(executor.map(clip_chunk, chunks))
            
            # Combine results
            grid_country = pd.concat(clipped_chunks, ignore_index=True)
        else:
            # Use standard clipping for smaller datasets
            grid_country = gpd.clip(grid_lines_filtered, admin_boundaries)
        
        print(f"Loaded {len(grid_country)} grid line segments")
        return grid_country
        
    except Exception as e:
        print(f"Error loading grid data: {e}")
        return gpd.GeoDataFrame()

def split_intersecting_edges(lines):
    """
    Splits all lines at their intersection points. This is crucial for creating a valid network graph where junctions become nodes.
    It uses an adaptive approach: a memory-intensive but fast method for small datasets, and a chunked, slower but more memory-efficient
    method for very large grid networks (e.g., China).
    """
    num_lines = len(lines)
    
    if num_lines > 10000:  # Very large grid networks (like CHN)
        print(f"Processing {num_lines:,} grid lines using chunked approach...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 500  # Smaller chunks for very large datasets
        all_segments = []
        
        for i in range(0, num_lines, chunk_size):
            chunk_end = min(i + chunk_size, num_lines)
            chunk_lines = lines[i:chunk_end]
            
            # Process chunk
            chunk_merged = unary_union(chunk_lines)
            
            if isinstance(chunk_merged, (LineString, MultiLineString)):
                segments = [chunk_merged] if isinstance(chunk_merged, LineString) else list(chunk_merged.geoms)
                for segment in segments:
                    coords = list(segment.coords)
                    for j in range(len(coords) - 1):
                        all_segments.append(LineString([coords[j], coords[j + 1]]))
            
            # Progress report
            if (i + chunk_size) % 5000 == 0:
                print(f"    Processed {i + chunk_size:,}/{num_lines:,} grid lines...")
        
        print(f"Split into {len(all_segments):,} segments")
        return all_segments
        
    elif num_lines > 1000:  # Original parallel processing
        print(f"Processing {num_lines:,} grid lines using {MAX_WORKERS} parallel workers...")
        
        # Your existing parallel processing code here...
        batch_size = max(1, num_lines // MAX_WORKERS)
        batches = [lines[i:i+batch_size] for i in range(0, num_lines, batch_size)]
        
        def process_batch(batch_lines):
            batch_merged = unary_union(batch_lines)
            if isinstance(batch_merged, (LineString, MultiLineString)):
                segments = [batch_merged] if isinstance(batch_merged, LineString) else list(batch_merged.geoms)
                batch_segments = []
                for segment in segments:
                    coords = list(segment.coords)
                    for i in range(len(coords) - 1):
                        batch_segments.append(LineString([coords[i], coords[i + 1]]))
                return batch_segments
            return []
        
        # Process batches in parallel
        all_segments = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for i, future in enumerate(futures):
                batch_segments = future.result()
                all_segments.extend(batch_segments)
                print(f"  Processed grid batch {i+1}/{len(batches)}")
        
        return all_segments
    else:
        # Use original method for smaller datasets
        merged = unary_union(lines)
        
        if isinstance(merged, (LineString, MultiLineString)):
            segments = [merged] if isinstance(merged, LineString) else list(merged.geoms)
            final_segments = []
            
            for segment in segments:
                coords = list(segment.coords)
                for i in range(len(coords) - 1):
                    final_segments.append(LineString([coords[i], coords[i + 1]]))
            
            return final_segments
        return []

def stitch_network_components(network_graph, max_distance_km=10):
    """
    Connect disconnected network components using minimum spanning tree approach.
    Each component connects to only one other component (its nearest neighbor).
    Uses Jeju approach: 10km threshold, closest pair between components.
    
    Parameters:
    - network_graph: NetworkX graph
    - max_distance_km: Maximum distance in kilometers to connect components (default: 10km)
    
    Returns:
    - Modified NetworkX graph with connected components
    """
    if network_graph is None or len(network_graph.nodes) == 0:
        return network_graph
    
    # Find connected components
    components = list(nx.connected_components(network_graph))
    initial_components = len(components)
    
    # Filter out isolated components (single nodes) - they don't need to be connected
    significant_components = [comp for comp in components if len(comp) > 1]
    isolated_components = len(components) - len(significant_components)
    
    if len(significant_components) <= 1:
        if isolated_components > 0:
            print(f"Network has {len(significant_components)} significant component(s) and {isolated_components} isolated node(s)")
        else:
            print(f"Network already connected: {len(significant_components)} significant component(s)")
        return network_graph
    
    print(f"Found {len(significant_components)} significant components and {isolated_components} isolated nodes. Stitching significant components within {max_distance_km}km...")
    
    max_distance_m = max_distance_km * 1000  # Convert to meters
    connections_added = 0
    
    # Sort components by size (largest first) and create indexed list
    component_sizes = [(i, len(comp)) for i, comp in enumerate(significant_components)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size, largest first
    
    print(f"Component sizes: {[size for _, size in component_sizes]} nodes")
    
    # Use minimum spanning tree approach: start with largest component, connect others one by one
    # This creates a connected network with minimal connections
    unconnected_components = [idx for idx, _ in component_sizes[1:]]  # All except the largest
    connected_components = [component_sizes[0][0]]  # Start with largest component
    
    print(f"Starting with largest component (index {connected_components[0]}, {component_sizes[0][1]} nodes)")
    
    if unconnected_components:
        
        # Connect remaining components one by one to the nearest already-connected component
        while unconnected_components:
            best_connection = None
            best_distance = float('inf')
            best_component_idx = None
            
            # Find the shortest connection between any unconnected component and any connected component
            for unconnected_idx in unconnected_components:
                unconnected_component = significant_components[unconnected_idx]
                
                for connected_idx in connected_components:
                    connected_component = significant_components[connected_idx]
                    
                    # Find closest pair between these two components
                    for node1 in unconnected_component:
                        for node2 in connected_component:
                            if isinstance(node1, tuple) and isinstance(node2, tuple) and len(node1) == 2 and len(node2) == 2:
                                distance = ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_connection = (node1, node2)
                                    best_component_idx = unconnected_idx
            
            # Add the best connection if within distance threshold
            if best_connection and best_distance <= max_distance_m:
                node1, node2 = best_connection
                network_graph.add_edge(node1, node2, weight=best_distance, edge_type='component_stitch')
                connections_added += 1
                
                # Move the connected component from unconnected to connected list
                connected_components.append(best_component_idx)
                unconnected_components.remove(best_component_idx)
                
                component_size = len(significant_components[best_component_idx])
                print(f"  Connected component {best_component_idx+1} ({component_size} nodes) to network: {best_distance/1000:.2f}km")
            else:
                # No more components can be connected within distance threshold
                print(f"  Remaining {len(unconnected_components)} components are beyond {max_distance_km}km threshold")
                break
    
    # Check final connectivity
    final_components = len(list(nx.connected_components(network_graph)))
    final_significant = len([comp for comp in nx.connected_components(network_graph) if len(comp) > 1])
    print(f"Stitching complete: {len(significant_components)} → {final_significant} significant components, {connections_added} connections added")
    
    return network_graph

def create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf):
    """
    Creates the master network graph.
    This function combines grid lines, power plants (facilities), and population centers (centroids) into a single graph.
    - Grid line intersections and endpoints become 'grid_line' nodes.
    - Facilities and centroids are added as their own node types.
    - Facilities and centroids are connected to their nearest node on the grid.
    - The graph is then "stitched" to connect any isolated sub-networks.
    This function is heavily optimized using spatial indexing (cKDTree) for fast nearest-neighbor searches.
    """
    # Get UTM CRS for accurate distance calculations
    center_lon = facilities_gdf.geometry.union_all().centroid.x if not facilities_gdf.empty else grid_lines_gdf.geometry.union_all().centroid.x
    center_lat = facilities_gdf.geometry.union_all().centroid.y if not facilities_gdf.empty else grid_lines_gdf.geometry.union_all().centroid.y
    utm_crs = get_utm_crs(center_lon, center_lat)
    
    print(f"Projecting to {utm_crs}")
    
    # Project to UTM
    facilities_utm = facilities_gdf.to_crs(utm_crs)
    grid_lines_utm = grid_lines_gdf.to_crs(utm_crs)
    centroids_utm = centroids_gdf.to_crs(utm_crs)
    
    # Initialize graph
    G = nx.Graph()
    
    # Process grid lines
    single_lines = []
    for _, row in grid_lines_utm.iterrows():
        if isinstance(row.geometry, MultiLineString):
            single_lines.extend(list(row.geometry.geoms))
        else:
            single_lines.append(row.geometry)
    
    # OPTIMIZATION 1: Timer for splitting
    print(f"Splitting {len(single_lines)} grid lines at intersections...")
    split_start = time.time()
    split_lines = split_intersecting_edges(single_lines)
    print(f"  Split complete in {time.time() - split_start:.2f}s: {len(single_lines)} -> {len(split_lines)} segments")
    
    # Create nodes from line endpoints
    grid_nodes = set()
    for line in split_lines:
        coords = list(line.coords)
        grid_nodes.add(coords[0])
        grid_nodes.add(coords[-1])
    
    print(f"Adding {len(grid_nodes)} grid nodes to graph...")
    
    # Add nodes to graph
    for node in grid_nodes:
        G.add_node(node, type='grid_line')
    
    # Add facility nodes
    facility_nodes = set()
    for i, (idx, point) in enumerate(zip(facilities_gdf.index, facilities_utm.geometry)):
        node_coord = (point.x, point.y)
        facility_nodes.add(node_coord)
        G.add_node(node_coord, type='facility', facility_idx=idx)
    
    # Add centroid nodes
    centroid_nodes = set()
    for i, (idx, point) in enumerate(zip(centroids_gdf.index, centroids_utm.geometry)):
        node_coord = (point.x, point.y)
        centroid_nodes.add(node_coord)
        G.add_node(node_coord, type='pop_centroid', centroid_idx=idx)
    
    print(f"Added {len(facility_nodes)} facility nodes and {len(centroid_nodes)} centroid nodes")
    
    # Add edges from grid lines
    print(f"Adding {len(split_lines)} grid edges...")
    for line in split_lines:
        coords = list(line.coords)
        G.add_edge(coords[0], coords[-1], weight=line.length, edge_type='grid_infrastructure')
    
    # OPTIMIZATION 2: Use spatial index for finding nearest grid nodes
    print("Building spatial index for grid nodes...")
    from scipy.spatial import cKDTree
    
    # Convert grid nodes to array for KDTree
    grid_node_list = list(grid_nodes)
    grid_node_array = np.array(grid_node_list)
    
    # Build KDTree for fast nearest neighbor search
    grid_tree = cKDTree(grid_node_array)
    
    # OPTIMIZATION 3: Vectorized connection of facilities to grid
    print(f"Connecting {len(facilities_utm)} facilities to nearest grid nodes...")
    connect_start = time.time()
    
    max_distance = 50000  # 50km threshold
    
    # Process facilities
    if len(facility_nodes) > 0:
        facility_coords = np.array(list(facility_nodes))
        
        # Find nearest grid node for each facility using KDTree
        distances, indices = grid_tree.query(facility_coords, k=1)
        
        # Add edges for facilities within threshold
        for i, (coord, dist, idx) in enumerate(zip(facility_nodes, distances, indices)):
            if dist <= max_distance:
                nearest_grid = grid_node_list[idx]
                G.add_edge(coord, nearest_grid, weight=dist, edge_type='grid_to_facility')
    
    print(f"  Facilities connected in {time.time() - connect_start:.2f}s")
    
    # OPTIMIZATION 4: Process centroids in batches for large datasets
    print(f"Connecting {len(centroids_utm)} centroids to nearest grid nodes...")
    connect_start = time.time()
    
    if len(centroid_nodes) > 10000:  # Large number of centroids
        # Process in batches to avoid memory issues
        batch_size = 5000
        centroid_list = list(centroid_nodes)
        
        for i in range(0, len(centroid_list), batch_size):
            batch_end = min(i + batch_size, len(centroid_list))
            batch_coords = np.array(centroid_list[i:batch_end])
            
            # Find nearest grid nodes for batch
            distances, indices = grid_tree.query(batch_coords, k=1)
            
            # Add edges for centroids within threshold
            for j, (coord, dist, idx) in enumerate(zip(centroid_list[i:batch_end], distances, indices)):
                if dist <= max_distance:
                    nearest_grid = grid_node_list[idx]
                    G.add_edge(coord, nearest_grid, weight=dist, edge_type='centroid_to_grid')
            
            if (i + batch_size) % 10000 == 0:
                print(f"    Processed {min(i + batch_size, len(centroid_list)):,}/{len(centroid_list):,} centroids...")
    else:
        # Process all at once for smaller datasets
        if len(centroid_nodes) > 0:
            centroid_coords = np.array(list(centroid_nodes))
            
            # Find nearest grid node for each centroid
            distances, indices = grid_tree.query(centroid_coords, k=1)
            
            # Add edges for centroids within threshold
            for coord, dist, idx in zip(centroid_nodes, distances, indices):
                if dist <= max_distance:
                    nearest_grid = grid_node_list[idx]
                    G.add_edge(coord, nearest_grid, weight=dist, edge_type='centroid_to_grid')
    
    print(f"  Centroids connected in {time.time() - connect_start:.2f}s")
    
    # Store UTM CRS for coordinate conversion
    G.graph['utm_crs'] = utm_crs
    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Stitch disconnected components within 10km (Jeju approach)
    print("Stitching network components...")
    stitch_start = time.time()
    G = stitch_network_components(G, max_distance_km=10)
    print(f"  Stitching complete in {time.time() - stitch_start:.2f}s")
    
    return G

def find_nearest_network_node(point_geom, network_graph):
    """
    Find the nearest network node to a given point geometry.
    
    Parameters:
    - point_geom: Shapely Point geometry
    - network_graph: NetworkX graph with node coordinates as tuples
    
    Returns:
    - Node ID of nearest network node, or None if no nodes found
    """
    if network_graph is None or len(network_graph.nodes) == 0:
        return None
    
    min_distance = float('inf')
    nearest_node = None
    
    # Get UTM coordinates for accurate distance calculation
    center_point = point_geom
    utm_crs = get_utm_crs(center_point.x, center_point.y)
    
    # Convert point to UTM
    point_utm = gpd.GeoSeries([point_geom], crs='EPSG:4326').to_crs(utm_crs).iloc[0]
    point_x, point_y = point_utm.x, point_utm.y
    
    for node in network_graph.nodes():
        # Node coordinates are stored as tuples (x, y) in UTM
        if isinstance(node, tuple) and len(node) == 2:
            node_x, node_y = node
            
            # Calculate Euclidean distance in UTM coordinates
            distance = ((point_x - node_x) ** 2 + (point_y - node_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
    
    return nearest_node

def find_available_facilities_within_radius(centroid_geom, facilities_gdf, utm_crs, radius_km=100):
    """
    Find facilities within a radius of a centroid using Euclidean distance for performance.
    
    Parameters:
    - centroid_geom: Shapely Point geometry in WGS84
    - facilities_gdf: GeoDataFrame of facilities in WGS84
    - utm_crs: UTM CRS for distance calculation
    - radius_km: Maximum radius in kilometers
    
    Returns:
    - GeoDataFrame of facilities within radius
    """
    if facilities_gdf.empty:
        return facilities_gdf.copy()
    
    # Convert to UTM for accurate distance calculation
    centroid_utm = gpd.GeoSeries([centroid_geom], crs='EPSG:4326').to_crs(utm_crs).iloc[0]
    facilities_utm = facilities_gdf.to_crs(utm_crs)
    
    # Calculate Euclidean distances
    distances = facilities_utm.geometry.distance(centroid_utm) / 1000  # Convert to km
    
    # Filter facilities within radius
    within_radius = facilities_gdf[distances <= radius_km].copy()
    
    return within_radius

def create_polyline_layer(active_connections, network_graph, country_iso3):
    """
    Creates a polyline layer showing the actual network paths for active supply connections.
    This visualizes exactly how electricity is flowing from a specific plant to a specific population center.
    """
    all_geometries = []
    all_attributes = []
    
    if not active_connections:
        return gpd.GeoDataFrame()
    
    utm_crs = network_graph.graph.get('utm_crs', 'EPSG:3857')

    for connection in active_connections:
        centroid_idx = connection.get('centroid_idx')
        facility_gem_id = connection.get('facility_gem_id')
        path_nodes = connection.get('path_nodes', [])
        
        try:
            if path_nodes and len(path_nodes) >= 2:
                path_points = []
                for node in path_nodes:
                    point_utm = gpd.GeoSeries([Point(node)], crs=utm_crs)
                    point_wgs84 = point_utm.to_crs(COMMON_CRS)
                    path_points.append((point_wgs84.iloc[0].x, point_wgs84.iloc[0].y))
                polyline_geom = LineString(path_points)
                all_geometries.append(polyline_geom)
                all_attributes.append({
                    'centroid_idx': centroid_idx,
                    'facility_gem_id': facility_gem_id,
                    'facility_type': connection.get('facility_type'),
                    'distance_km': connection.get('distance_km'),
                    'supply_mwh': connection.get('supply_mwh'),
                    'connection_id': f"C{centroid_idx}_F{facility_gem_id}",
                    'active_supply': 'Yes'
                })
        except Exception as e:
            print(f"Warning: Failed to create polyline for centroid {centroid_idx} to facility {facility_gem_id}: {e}")
    
    if all_geometries:
        polylines_layer = gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs=COMMON_CRS)
        polylines_layer['GID_0'] = country_iso3
        
        # Safe column selection to avoid KeyErrors
        preferred = ['geometry', 'GID_0', 'connection_id', 'centroid_idx', 'facility_gem_id', 'facility_type', 'distance_km', 'supply_mwh', 'active_supply']
        existing = [c for c in preferred if c in polylines_layer.columns]
        others = [c for c in polylines_layer.columns if c not in existing]
        polylines_layer = polylines_layer[existing + others]
        return polylines_layer
    
    return gpd.GeoDataFrame()

def calculate_network_distance_manual(network_graph, start_node, end_node):
    """
    Calculates the shortest path distance between two nodes in the network graph using Dijkstra's algorithm.
    Results are cached to avoid redundant calculations, which provides a significant performance boost.
    """
    # Create cache key (sort nodes to ensure bidirectional caching)
    cache_key = tuple(sorted([start_node, end_node]))
    
    # Check cache first
    if cache_key in path_cache:
        return path_cache[cache_key].copy() if path_cache[cache_key] else None
    
    try:
        # Original path calculation
        path_nodes = nx.shortest_path(network_graph, start_node, end_node, weight='weight')
        
        # Calculate distances as before
        total_distance = 0
        path_segments = []
        
        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            # Get edge data
            edge_data = network_graph[current_node][next_node]
            segment_weight = edge_data.get('weight', 0)
            edge_type = edge_data.get('edge_type', 'unknown')
            
            total_distance += segment_weight
            path_segments.append({
                'from_node': current_node,
                'to_node': next_node,
                'weight': segment_weight,
                'edge_type': edge_type
            })
        
        result = {
            'distance_km': total_distance / 1000.0,
            'path_nodes': path_nodes,
            'path_segments': path_segments,
            'total_segments': len(path_segments)
        }
        
        # Cache the result
        path_cache[cache_key] = result
        return result
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Cache negative result too
        path_cache[cache_key] = None
        return None

def process_centroid_distances_batch(centroid_batch, network_graph, facilities_gdf, centroid_mapping, facility_mapping, utm_crs):
    """
    Processes a batch of centroids for distance calculations. Designed to be run in parallel.
    For each centroid, it finds nearby facilities and calculates the shortest network path to them.
    """
    batch_results = []
    
    for centroid_idx, centroid in centroid_batch.iterrows():
        network_centroid = centroid_mapping.get(centroid_idx)
        distances = []
        
        if network_centroid is not None:
            available_facilities = find_available_facilities_within_radius(
                centroid.geometry, facilities_gdf, utm_crs, radius_km=100
            )
            for facility_idx, facility in available_facilities.iterrows():
                network_facility = facility_mapping.get(facility_idx)
                if network_facility is not None:
                    distance_result = calculate_network_distance_manual(
                        network_graph, network_centroid, network_facility
                    )
                    if distance_result is not None:
                        euclidean_distance = centroid.geometry.distance(facility.geometry) * 111.32
                        
                        # Get GEM ID with better null handling
                        gem_id_raw = facility.get('GEM unit/phase ID', '')
                        gem_id = str(gem_id_raw) if pd.notna(gem_id_raw) and gem_id_raw != '' else ''
                        
                        distances.append({
                            'facility_idx': facility_idx,
                            'distance_km': distance_result['distance_km'],
                            'path_nodes': distance_result['path_nodes'],
                            'path_segments': distance_result['path_segments'],
                            'total_segments': distance_result['total_segments'],
                            'facility_type': facility.get('Grouped_Type', ''),
                            'facility_capacity': facility.get('Adjusted_Capacity_MW', 0),
                            'facility_lat': facility.geometry.y,
                            'facility_lon': facility.geometry.x,
                            'gem_id': gem_id,
                            'euclidean_distance_km': euclidean_distance,
                            'network_path': distance_result['path_nodes']
                        })
        distances.sort(key=lambda x: x['distance_km'])
        batch_results.append({'centroid_idx': centroid_idx, 'distances': distances})
    
    return batch_results

def calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph):
    """
    Calculates network distances for all centroids to nearby facilities using parallel processing.
    It adaptively changes the batch size based on the number of centroids to balance memory usage and parallelism.
    This is the most computationally expensive part of the script.
    """
    num_centroids = len(centroids_gdf)
    num_facilities = len(facilities_gdf)
    print(f"Calculating distances for {num_centroids:,} centroids to {num_facilities:,} facilities using {MAX_WORKERS} parallel workers...")
    
    # Create coordinate mappings
    centroid_mapping = {}
    facility_mapping = {}
    
    for node, data in network_graph.nodes(data=True):
        if data.get('type') == 'pop_centroid':
            centroid_mapping[data.get('centroid_idx')] = node
        elif data.get('type') == 'facility':
            facility_mapping[data.get('facility_idx')] = node
    
    # Get UTM CRS for distance calculations
    center_point = centroids_gdf.geometry.union_all().centroid
    utm_crs = get_utm_crs(center_point.x, center_point.y)
    
    # Adaptive batch sizing based on dataset scale
    # Consider both centroids and facilities for complexity
    complexity_factor = num_centroids * min(num_facilities, 100) / 1000000  # Rough complexity estimate
    
    if num_centroids > 1000000:  # Over 1M centroids (like CHN with 4M)
        batch_size = 25  # Smaller batches for more parallelism (was 50)
    elif num_centroids > 100000:  # 100k-1M centroids
        batch_size = 50   # Was 100
    elif num_centroids > 50000:  # 50k-100k (like AFG with 71k)
        batch_size = 100  # Was 250
    else:  # Under 10k
        batch_size = max(50, num_centroids // MAX_WORKERS)
    
    # Create batches
    centroid_batches = []
    for i in range(0, num_centroids, batch_size):
        batch = centroids_gdf.iloc[i:min(i+batch_size, num_centroids)]
        centroid_batches.append(batch)
    
    print(f"Processing {len(centroid_batches)} batches of ~{batch_size} centroids each")
    
    # Progress reporting frequency
    if num_centroids > 100000:
        report_interval = max(10, len(centroid_batches) // 20)  # Report ~20 times
    else:
        report_interval = max(1, len(centroid_batches) // 10)  # Report ~10 times
    
    # Process batches in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed arguments
        process_func = partial(
            process_centroid_distances_batch,
            network_graph=network_graph,
            facilities_gdf=facilities_gdf,
            centroid_mapping=centroid_mapping,
            facility_mapping=facility_mapping,
            utm_crs=utm_crs
        )
        
        # Submit all batches
        future_to_batch = {executor.submit(process_func, batch): i for i, batch in enumerate(centroid_batches)}
        
        # Collect results
        for future in future_to_batch:
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                # Progress reporting
                if (batch_idx + 1) % report_interval == 0:
                    progress_pct = (batch_idx + 1) / len(centroid_batches) * 100
                    print(f"  Progress: {batch_idx + 1}/{len(centroid_batches)} batches ({progress_pct:.1f}%)")
                    
            except Exception as exc:
                print(f"  Batch {batch_idx} generated an exception: {exc}")
    
    print(f"Distance calculation completed for {len(all_results)} centroids")
    return all_results

def create_grid_lines_layer(grid_lines_gdf, network_graph, active_connections):
    """Creates the final grid lines layer, including original infrastructure and newly created connection lines (e.g., facility-to-grid, stitches)."""
    all_geometries = []
    all_attributes = []
    
    # Add original grid segments
    if not grid_lines_gdf.empty:
        for idx, row in grid_lines_gdf.iterrows():
            geom = row.geometry
            center = geom.centroid
            utm = get_utm_crs(center.x, center.y)
            try:
                length_km = gpd.GeoSeries([geom], crs=COMMON_CRS).to_crs(utm).iloc[0].length / 1000.0
            except Exception:
                length_km = None
            all_geometries.append(geom)
            all_attributes.append({'line_type': 'grid_infrastructure', 'line_id': f'grid_{idx}', 'distance_km': length_km})
    
    # Add connection edges from graph
    if network_graph is not None:
        utm_crs = network_graph.graph.get('utm_crs', 'EPSG:3857')
        for n1, n2, ed in network_graph.edges(data=True):
            et = ed.get('edge_type', 'unknown')
            if et in ['centroid_to_grid', 'grid_to_facility', 'component_stitch']:
                try:
                    s = gpd.GeoSeries([Point(n1), Point(n2)], crs=utm_crs).to_crs(COMMON_CRS)
                    geom = LineString([(s.iloc[0].x, s.iloc[0].y), (s.iloc[1].x, s.iloc[1].y)])
                    dist_km = (ed.get('weight', 0) or 0) / 1000.0
                    all_geometries.append(geom)
                    all_attributes.append({'line_type': et, 'line_id': f'{et}_{len(all_attributes)}', 'distance_km': dist_km})
                except Exception:
                    continue
    
    if all_geometries:
        return gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs=COMMON_CRS)
    return gpd.GeoDataFrame()

def create_all_layers(centroids_gdf, facilities_gdf, grid_lines_gdf, network_graph, active_connections, country_iso3, facility_supplied=None, facility_remaining=None):
    """Assembles all the final GeoDataFrame layers for output."""
    layers = {}
    
    # Centroids layer - population centers with demand and supply allocation results
    centroid_columns = ['geometry', 'GID_0', 'Population_centroid',
                        'Population_2024_centroid', 'Population_2030_centroid', 'Population_2050_centroid',
                        'Total_Demand_2024_centroid', 'Total_Demand_2030_centroid', 'Total_Demand_2050_centroid',
                        'supplying_facility_distance', 'supplying_facility_type', 'supplying_facility_gem_id', 
                        'supply_received_mwh', 'supply_status']
    available_columns = [c for c in centroid_columns if c in centroids_gdf.columns]
    centroids_layer = centroids_gdf[available_columns].copy()
    
    # Add centroid_idx column for spatial analysis tracking
    centroids_layer['centroid_idx'] = centroids_layer.index
    
    layers['centroids'] = centroids_layer
    
    # Grid lines layer - electrical grid infrastructure and connections
    grid_lines_layer = create_grid_lines_layer(grid_lines_gdf, network_graph, active_connections)
    if not grid_lines_layer.empty:
        grid_lines_layer['GID_0'] = country_iso3
        cols = ['geometry', 'GID_0'] + [c for c in grid_lines_layer.columns if c not in ['geometry', 'GID_0']]
        layers['grid_lines'] = grid_lines_layer[cols]
    
    # Facilities layer - energy generation facilities with capacity, production, and allocation data
    if not facilities_gdf.empty:
        facilities_simplified = facilities_gdf[['geometry', 'GEM unit/phase ID', 'Grouped_Type', 'Latitude', 'Longitude', 'Adjusted_Capacity_MW', 'total_mwh']].copy()
        facilities_simplified['GID_0'] = country_iso3
        
        # total_mwh is now directly available from the facility data
        # No need to calculate using capacity factors
        
        # Add supply allocation tracking columns
        if facility_supplied is not None and facility_remaining is not None:
            facilities_simplified['supplied_mwh'] = facilities_simplified.index.map(lambda idx: facility_supplied.get(idx, 0.0))
            facilities_simplified['remaining_mwh'] = facilities_simplified.index.map(lambda idx: facility_remaining.get(idx, 0.0))
        else:
            # Default values if no allocation data available
            facilities_simplified['supplied_mwh'] = 0.0
            facilities_simplified['remaining_mwh'] = facilities_simplified['total_mwh']
        
        cols = ['geometry', 'GID_0'] + [c for c in facilities_simplified.columns if c not in ['geometry', 'GID_0']]
        layers['facilities'] = facilities_simplified[cols]
    
    # Polylines layer - active supply connections between centroids and facilities
    polylines_layer = create_polyline_layer(active_connections, network_graph, country_iso3)
    if not polylines_layer.empty:
        layers['polylines'] = polylines_layer
    
    return layers

def process_country_supply(country_iso3, output_dir="outputs_per_country", test_mode=False):
    """Main function to process supply analysis for a single country, orchestrating all steps."""
    # Clear path cache for each country
    global path_cache
    path_cache = {}
    
    print(f"\nProcessing {country_iso3}... (Mode: {'TEST' if test_mode else 'PRODUCTION'})")
    total_start = time.time()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load inputs with timing
        with timer("Load admin boundaries"):
            admin_boundaries = load_admin_boundaries(country_iso3)
            country_bbox = get_country_bbox(admin_boundaries)
        
        with timer("Load population centroids"):
            centroids_gdf = load_population_centroids(country_bbox, admin_boundaries)
        
        with timer("Load facilities"):
            facilities_gdf = load_energy_facilities(country_iso3, 2024)
        
        with timer("Load grid lines"):
            grid_lines_gdf = load_grid_lines(country_bbox, admin_boundaries)
        
        print(f"Loaded: {len(centroids_gdf)} centroids, {len(facilities_gdf)} facilities, {len(grid_lines_gdf)} grid lines")
        
        # Country tag
        centroids_gdf['GID_0'] = country_iso3
        
        # Demand projections
        with timer("Load population and demand projections"):
            centroids_gdf = load_population_and_demand_projections(centroids_gdf, country_iso3)
        
        # Default outputs
        network_graph = None
        active_connections = []
        facility_supplied = None
        facility_remaining = None
        
        # Process network and allocate supply if facilities and grid exist
        if not facilities_gdf.empty and not grid_lines_gdf.empty:
            # Create network graph
            with timer("Create network graph"):
                network_graph = create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf)
            
            # Calculate distances between centroids and facilities
            with timer("Calculate facility distances"):
                centroid_facility_distances = calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph)
            
            # Initialize facility remaining capacities (MWh/year) and track supplied amounts
            facility_remaining = {}
            facility_supplied = {}
            for idx, facility in facilities_gdf.iterrows():
                # Use total_mwh directly from facility data
                total_mwh = facility.get('total_mwh', 0) or 0
                facility_remaining[idx] = total_mwh
                facility_supplied[idx] = 0.0
            
            # Prepare centroid columns for supply allocation
            centroids_gdf['supplying_facility_distance'] = ''
            centroids_gdf['supplying_facility_type'] = ''
            centroids_gdf['supplying_facility_gem_id'] = ''
            centroids_gdf['supply_received_mwh'] = 0.0
            centroids_gdf['supply_status'] = 'Not Filled'
            
            demand_col = 'Total_Demand_2024_centroid'
            
            # VECTORIZED SUPPLY ALLOCATION
            with timer("Allocate supply (Vectorized)"):
                centroids_gdf, active_connections = allocate_supply_vectorized(
                    centroids_gdf, facilities_gdf, centroid_facility_distances,
                    facility_remaining, facility_supplied, demand_col
                )
            
            # Update supply status for all centroids (keeping original loop for safety)
            for centroid_idx in centroids_gdf.index:  # Use .index instead of iterrows
                demand = float(centroids_gdf.loc[centroid_idx, demand_col] if pd.notna(centroids_gdf.loc[centroid_idx, demand_col]) else 0)
                received = float(centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] if pd.notna(centroids_gdf.loc[centroid_idx, 'supply_received_mwh']) else 0)
                
                if demand <= 0:
                    centroids_gdf.loc[centroid_idx, 'supply_status'] = 'No Demand'
                elif received >= demand:
                    centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Filled'
                elif received > 0:
                    centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Partially Filled'
                else:
                    centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Not Filled'
        else:
            # Initialize supply columns if network processing is skipped
            centroids_gdf['supplying_facility_distance'] = np.nan
            centroids_gdf['supplying_facility_type'] = ''
            centroids_gdf['supplying_facility_gem_id'] = ''
            centroids_gdf['supply_received_mwh'] = 0.0
            centroids_gdf['supply_status'] = 'Not Filled'
        
        # Create all layers
        with timer("Create output layers"):
            layers = create_all_layers(centroids_gdf, facilities_gdf, grid_lines_gdf, network_graph, active_connections, country_iso3, facility_supplied, facility_remaining)
        
        # Write outputs based on mode
        with timer("Save outputs"):
            if test_mode:
                # Test mode: Full GPKG output for detailed analysis
                output_file = output_path / f"p1_{country_iso3}.gpkg"
                
                for layer_name, layer_data in layers.items():
                    layer_data.to_file(output_file, layer=layer_name, driver="GPKG")
                
                print(f"Test mode: Full GPKG saved to {output_file}")
                output_result = str(output_file)
            else:
                # Production mode: Lightweight Parquet files for global analysis
                parquet_dir = output_path / "parquet"
                parquet_dir.mkdir(exist_ok=True)
                
                # Define essential columns for each layer
                parquet_schemas = {
                    'centroids': ['geometry', 'GID_0', 'supplying_facility_type', 'supply_status'],
                    'facilities': ['geometry', 'GID_0', 'Grouped_Type', 'total_mwh'],
                    'grid_lines': ['geometry', 'GID_0', 'line_type'],
                    'polylines': ['geometry', 'GID_0', 'facility_type']
                }
                
                output_files = []
                for layer_name, layer_data in layers.items():
                    if layer_name in parquet_schemas and not layer_data.empty:
                        # Select only essential columns
                        essential_columns = parquet_schemas[layer_name]
                        available_columns = [col for col in essential_columns if col in layer_data.columns]
                        
                        if available_columns:
                            layer_essential = layer_data[available_columns].copy()
                            parquet_file = parquet_dir / f"{layer_name}_{country_iso3}.parquet"
                            layer_essential.to_parquet(parquet_file)
                            output_files.append(str(parquet_file))
                            print(f"  Saved {layer_name}: {len(layer_essential)} records → {parquet_file.name}")
                
                print(f"Production mode: {len(output_files)} Parquet files saved to {parquet_dir}")
                output_result = str(parquet_dir)
        
        # Generate summary statistics for 2024
        print(f"\n{'='*60}")
        print(f"SUPPLY ANALYSIS SUMMARY FOR {country_iso3} (2024)")
        print(f"{'='*60}")
        
        # Calculate demand statistics
        demand_col_2024 = 'Total_Demand_2024_centroid'
        total_needed_mwh = centroids_gdf[demand_col_2024].sum()
        print(f"Total needed energy (demand): {total_needed_mwh:,.0f} MWh")
        
        # Calculate supply statistics from facilities
        if facility_supplied is not None and facility_remaining is not None:
            total_supplied_mwh = sum(facility_supplied.values())
            total_remaining_mwh = sum(facility_remaining.values())
            total_facility_capacity = total_supplied_mwh + total_remaining_mwh
            
            supplied_pct = (total_supplied_mwh / total_facility_capacity * 100) if total_facility_capacity > 0 else 0
            remaining_pct = (total_remaining_mwh / total_facility_capacity * 100) if total_facility_capacity > 0 else 0
            
            print(f"Total supplied energy: {total_supplied_mwh:,.0f} MWh ({supplied_pct:.1f}% of facility capacity)")
            print(f"Total remaining capacity: {total_remaining_mwh:,.0f} MWh ({remaining_pct:.1f}% of facility capacity)")
            
            # Calculate additional energy needed
            total_additionally_needed_mwh = max(0, total_needed_mwh - total_supplied_mwh)
            print(f"Total additionally needed: {total_additionally_needed_mwh:,.0f} MWh")
        else:
            print("Total supplied energy: 0 MWh (no facilities processed)")
            print("Total remaining capacity: 0 MWh (no facilities processed)")
            print(f"Total additionally needed: {total_needed_mwh:,.0f} MWh")
        
        # Calculate centroid status statistics
        status_counts = centroids_gdf['supply_status'].value_counts()
        total_centroids = len(centroids_gdf)
        
        print(f"\nCentroid supply status:")
        for status in ['Filled', 'Partially Filled', 'Not Filled', 'No Demand']:
            count = status_counts.get(status, 0)
            pct = (count / total_centroids * 100) if total_centroids > 0 else 0
            print(f"  {status}: {count:,} centroids ({pct:.1f}%)")
        
        print(f"{'='*60}")
        
        print(f"Results saved to {output_result}")
        print(f"Processing completed using {MAX_WORKERS} parallel workers")
        return output_result
    except Exception as e:
        print(f"Error processing {country_iso3}: {e}")
        return None

def allocate_supply_vectorized(centroids_gdf, facilities_gdf, centroid_facility_distances, 
                               facility_remaining, facility_supplied, demand_col):
    """
    Core supply allocation logic. This function matches supply (facilities) to demand (centroids).
    The logic is as follows:
    1. Iterate through facilities, starting with the one with the largest capacity.
    2. For the current facility, find all population centroids that need power and are connected to it.
    3. Sort these centroids by the network distance to the facility (closest first).
    4. Allocate the facility's power to these centroids one by one until the facility's capacity is used up.
    5. Repeat for the next largest facility.
    This version is optimized for performance using array operations where possible.
    """
    import numpy as np
    
    # Reset index to ensure we have sequential indices for array operations
    centroids_gdf_reset = centroids_gdf.reset_index(drop=True)
    
    # Create index mapping from original to reset indices
    original_to_reset = {orig_idx: reset_idx for reset_idx, orig_idx in enumerate(centroids_gdf.index)}
    
    # Pre-compute demand and initialize received arrays
    num_centroids = len(centroids_gdf_reset)
    centroid_demands = centroids_gdf_reset[demand_col].fillna(0).values.astype(np.float64)
    centroid_received = np.zeros(num_centroids, dtype=np.float64)
    
    # Initialize tracking lists for facility info (for comma-separated strings)
    supplying_distances = [[] for _ in range(num_centroids)]
    supplying_types = [[] for _ in range(num_centroids)]
    supplying_gem_ids = [[] for _ in range(num_centroids)]
    active_connections = []
    
    # Create facility list sorted by capacity (preserving original logic)
    facility_capacities = []
    for idx, facility in facilities_gdf.iterrows():
        total_capacity = facility_remaining.get(idx, 0)
        if total_capacity > 0:
            facility_capacities.append({
                'facility_idx': idx,
                'total_capacity': total_capacity,
                'gem_id': str(facility.get('GEM unit/phase ID', '')) if pd.notna(facility.get('GEM unit/phase ID', '')) else '',
                'facility_type': facility.get('Grouped_Type', ''),
                'geometry': facility.geometry
            })
    
    # Sort by capacity (largest first) - matches original behavior
    facility_capacities.sort(key=lambda x: x['total_capacity'], reverse=True)
    print(f"Processing {len(facility_capacities)} facilities by capacity (largest first)...")
    
    # Process each facility (preserving original allocation logic)
    for facility_info in facility_capacities:
        facility_idx = facility_info['facility_idx']
        remaining_capacity = facility_remaining.get(facility_idx, 0)
        
        if remaining_capacity <= 0:
            continue
        
        # Find centroids that can reach this facility
        facility_centroid_distances = []
        for item in centroid_facility_distances:
            original_centroid_idx = item['centroid_idx']
            
            # Map to reset index for array operations
            if original_centroid_idx not in original_to_reset:
                continue  # Skip if centroid was filtered out
                
            reset_centroid_idx = original_to_reset[original_centroid_idx]
            
            # Find this facility in the centroid's distance list
            for distance_info in item['distances']:
                if distance_info.get('facility_idx') == facility_idx:
                    remaining_demand = centroid_demands[reset_centroid_idx] - centroid_received[reset_centroid_idx]
                    
                    if remaining_demand > 0:
                        # Get nearest facility distance (for sorting)
                        nearest_facility_distance = item['distances'][0]['distance_km'] if item['distances'] else float('inf')
                        
                        facility_centroid_distances.append({
                            'centroid_idx': reset_centroid_idx,  # Use reset index
                            'original_idx': original_centroid_idx,  # Keep original for reference
                            'distance_to_this_facility': distance_info.get('distance_km'),
                            'nearest_facility_distance': nearest_facility_distance,
                            'remaining_demand': remaining_demand,
                            'path_nodes': distance_info.get('path_nodes', [])
                        })
                    break
        
        # Sort by distance to this facility (preserving original sort logic)
        facility_centroid_distances.sort(key=lambda x: x['distance_to_this_facility'])
        
        # Allocate to centroids
        for centroid_info in facility_centroid_distances:
            if remaining_capacity <= 0:
                break
            
            reset_centroid_idx = centroid_info['centroid_idx']
            remaining_demand = centroid_info['remaining_demand']
            distance_km = centroid_info['distance_to_this_facility']
            
            # Allocate supply
            allocated = min(remaining_demand, remaining_capacity)
            centroid_received[reset_centroid_idx] += allocated
            facility_remaining[facility_idx] -= allocated
            facility_supplied[facility_idx] += allocated
            remaining_capacity -= allocated
            
            # Track facility info for this centroid
            supplying_distances[reset_centroid_idx].append(f"{distance_km:.2f}")
            supplying_types[reset_centroid_idx].append(facility_info['facility_type'])
            supplying_gem_ids[reset_centroid_idx].append(facility_info['gem_id'])
            
            # Track active connection (use original index for connection tracking)
            centroid_geom = centroids_gdf_reset.iloc[reset_centroid_idx].geometry
            active_connections.append({
                'centroid_idx': centroid_info['original_idx'],  # Use original index for output
                'facility_gem_id': facility_info['gem_id'],
                'centroid_lat': centroid_geom.y,
                'centroid_lon': centroid_geom.x,
                'facility_lat': facility_info['geometry'].y,
                'facility_lon': facility_info['geometry'].x,
                'network_path': centroid_info['path_nodes'],
                'supply_mwh': allocated,
                'distance_km': distance_km,
                'facility_type': facility_info['facility_type'],
                'path_nodes': centroid_info['path_nodes']
            })
    
    # Update original centroids_gdf with vectorized operations
    # Map values back to original indices
    for orig_idx, reset_idx in original_to_reset.items():
        centroids_gdf.loc[orig_idx, 'supply_received_mwh'] = centroid_received[reset_idx]
        centroids_gdf.loc[orig_idx, 'supplying_facility_distance'] = ', '.join(supplying_distances[reset_idx]) if supplying_distances[reset_idx] else ''
        centroids_gdf.loc[orig_idx, 'supplying_facility_type'] = ', '.join(supplying_types[reset_idx]) if supplying_types[reset_idx] else ''
        centroids_gdf.loc[orig_idx, 'supplying_facility_gem_id'] = ', '.join(supplying_gem_ids[reset_idx]) if supplying_gem_ids[reset_idx] else ''
    
    return centroids_gdf, active_connections

def main():
    """Parses command-line arguments and runs the main processing function."""
    parser = argparse.ArgumentParser(description='Process supply analysis for a country')
    parser.add_argument('country_iso3', help='ISO3 country code')
    parser.add_argument('--output-dir', default='outputs_per_country', help='Output directory')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: generate full GPKG with all layers for detailed analysis')
    args = parser.parse_args()
    result = process_country_supply(args.country_iso3, args.output_dir, test_mode=args.test)
    if result:
        print(f"Successfully processed {args.country_iso3}")
        return 0
    else:
        print(f"Failed to process {args.country_iso3}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
