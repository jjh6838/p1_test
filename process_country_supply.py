#!/usr/bin/env python3
"""
Clean supply analysis per country with network-based grid lines
Produces 4 layers: centroids, facilities, grid_lines, polylines
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
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# Suppress warnings
# warnings.filterwarnings("ignore")

# Constants
COMMON_CRS = "EPSG:4326"  # WGS84 for input/output
YEARS = [2024, 2030, 2050]
SUPPLY_TYPES = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]

# Get optimal number of workers based on available CPUs
MAX_WORKERS = min(72, max(1, os.cpu_count() or 1))
print(f"Parallel processing configured for {MAX_WORKERS} workers")

def get_utm_crs(lon, lat):
    """Get appropriate UTM CRS for given coordinates"""
    utm_zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone}" if lat >= 0 else f"EPSG:{32700 + utm_zone}"

def get_country_bbox(admin_boundaries, buffer=0.1):
    """Get bounding box for a country with buffer"""
    bounds = admin_boundaries.total_bounds
    return [bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer]

def load_admin_boundaries(country_iso3):
    """Load administrative boundaries for country"""
    admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410-levels.gpkg', layer="ADM_0")
    country_data = admin_boundaries[admin_boundaries['GID_0'] == country_iso3]
    
    if country_data.empty:
        raise ValueError(f"No boundaries found for country {country_iso3}")
    
    return country_data

def load_population_centroids(country_bbox, admin_boundaries):
    """Load and process population centroids from raster with parallel coordinate processing"""
    minx, miny, maxx, maxy = country_bbox
    
    with rasterio.open('bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif') as src:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        pop_data = src.read(1, window=window)
        windowed_transform = rasterio.windows.transform(window, src.transform)
        
        rows, cols = pop_data.shape
        total_pixels = rows * cols
        
        print(f"Processing {total_pixels:,} population pixels using parallel workers...")
        
        # Create coordinate pairs in parallel
        def process_pixel_batch(start_idx, end_idx):
            batch_x, batch_y = [], []
            for idx in range(start_idx, end_idx):
                row = idx // cols
                col = idx % cols
                x, y = rasterio.transform.xy(windowed_transform, row, col)
                batch_x.append(x)
                batch_y.append(y)
            return batch_x, batch_y
        
        # Split work into batches
        batch_size = max(1, total_pixels // MAX_WORKERS)
        centroids_x, centroids_y = [], []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, total_pixels, batch_size):
                end_idx = min(i + batch_size, total_pixels)
                futures.append(executor.submit(process_pixel_batch, i, end_idx))
            
            for future in futures:
                batch_x, batch_y = future.result()
                centroids_x.extend(batch_x)
                centroids_y.extend(batch_y)
        
        print(f"Coordinate processing completed for {len(centroids_x):,} pixels")
        
        centroids_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(centroids_x, centroids_y),
            crs=COMMON_CRS
        )
        centroids_gdf["Population_centroid"] = pop_data.flatten()
        
        # Filter to country boundaries and remove zero population
        centroids_gdf = gpd.sjoin(centroids_gdf, admin_boundaries, how='inner', predicate='within')
        centroids_gdf = centroids_gdf[centroids_gdf["Population_centroid"] > 0].copy()
        
        return centroids_gdf

def load_population_and_demand_projections(centroids_gdf, country_iso3):
    """Load country-level population projections and calculate demand for multiple years"""
    print("Loading country-level population projections...")
    
    # Calculate baseline 2023 population from spatial data
    total_country_population_2023 = centroids_gdf["Population_centroid"].sum()
    print(f"Baseline population from spatial data (2023): {total_country_population_2023:,.0f}")
    
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
    
    return centroids_filtered

def load_conversion_rates(country_iso3):
    """Load country-specific capacity factors for each energy type.
    Excel columns contain annual generation hours (CF * 8760). We divide by 8760 to get capacity factors.
    """
    try:
        # Load conversion rates data
        conversion_df = pd.read_excel("outputs_processed_data/p1_a_ember_gem_2024.xlsx", 
                                    sheet_name="Grouped_cur")
        
        # Filter for the specified country
        country_conversion = conversion_df[conversion_df['Country Code'] == country_iso3]
        
        if country_conversion.empty:
            print(f"Warning: No conversion rates found for {country_iso3}, using default capacity factors")
            # Default capacity factors if country not found
            return {
                'Solar': 0.25,
                'Wind': 0.35,
                'Hydro': 0.45,
                'Other Renewables': 0.30,
                'Nuclear': 0.90,
                'Fossil': 0.50
            }
        
        # Extract conversion rates for each energy type
        country_data = country_conversion.iloc[0]
        conversion_rates = {}
        
        energy_types = ['Solar', 'Wind', 'Hydro', 'Other Renewables', 'Nuclear', 'Fossil']
        print(f"Converting annual generation hours to capacity factors for {country_iso3}:")
        
        for energy_type in energy_types:
            conv_col = f"{energy_type}_ConvRate"
            if conv_col in country_data:
                hours_per_year = country_data[conv_col]
                if pd.notna(hours_per_year) and hours_per_year > 0:
                    # Convert from annual generation hours to capacity factor
                    capacity_factor = hours_per_year / 8760.0
                    
                    # Clamp capacity factor to reasonable range (0-1)
                    # Some data might be erroneous (>8760 hours/year is impossible)
                    if capacity_factor > 1.0:
                        print(f"  Warning: {energy_type} has {hours_per_year:.1f} hours/year (impossible >8760). "
                              f"Clamping capacity factor from {capacity_factor:.3f} to 1.000")
                        capacity_factor = 1.0
                    elif capacity_factor < 0.001:
                        print(f"  Warning: {energy_type} has very low {hours_per_year:.3f} hours/year. "
                              f"Capacity factor: {capacity_factor:.6f} (might be already a decimal CF?)")
                    
                    conversion_rates[energy_type] = capacity_factor
                    print(f"  {energy_type}: {hours_per_year:.3f} hours/year → {capacity_factor:.3f} capacity factor ({capacity_factor*100:.1f}%)")
                else:
                    # Default rates if missing or invalid
                    default_rates = {
                        'Solar': 0.25,
                        'Wind': 0.35,
                        'Hydro': 0.45,
                        'Other Renewables': 0.30,
                        'Nuclear': 0.90,
                        'Fossil': 0.50
                    }
                    conversion_rates[energy_type] = default_rates.get(energy_type, 0.30)
                    print(f"  Warning: Using default capacity factor for {energy_type}: {conversion_rates[energy_type]:.3f}")
            else:
                print(f"  Warning: Column {conv_col} not found, using default")
                conversion_rates[energy_type] = 0.30  # Default fallback
        
        return conversion_rates
        
    except Exception as e:
        print(f"Error loading conversion rates for {country_iso3}: {e}")
        # Return default capacity factors
        return {
            'Solar': 0.25,
            'Wind': 0.35,
            'Hydro': 0.45,
            'Other Renewables': 0.30,
            'Nuclear': 0.90,
            'Fossil': 0.50
        }

def load_energy_facilities(country_iso3, year=2024):
    """Load energy facilities for country and year"""
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
    """Load and clip grid lines for country"""
    try:
        grid_lines = gpd.read_file('bigdata_gridfinder/grid.gpkg')
        minx, miny, maxx, maxy = country_bbox
        
        # Clip to bounding box then country boundaries
        from shapely.geometry import Polygon
        bbox_geom = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        bbox_poly = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=COMMON_CRS)
        
        grid_clipped = gpd.clip(grid_lines, bbox_poly)
        if not grid_clipped.empty:
            grid_country = gpd.clip(grid_clipped, admin_boundaries)
            print(f"Loaded {len(grid_country)} grid line segments")
            return grid_country
        
    except Exception as e:
        print(f"Error loading grid data: {e}")
    
    return gpd.GeoDataFrame()

def split_intersecting_edges(lines):
    """Split lines at intersections with parallel processing for large datasets"""
    if len(lines) > 1000:  # Use parallel processing for large datasets
        print(f"Processing {len(lines)} grid lines using parallel workers...")
        
        # Split lines into batches for parallel processing
        batch_size = max(1, len(lines) // MAX_WORKERS)
        batches = [lines[i:i+batch_size] for i in range(0, len(lines), batch_size)]
        
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
    """Create network graph from facilities, grid lines, and centroids"""
    # Get UTM CRS for accurate distance calculations
    center_lon = facilities_gdf.geometry.unary_union.centroid.x if not facilities_gdf.empty else grid_lines_gdf.geometry.unary_union.centroid.x
    center_lat = facilities_gdf.geometry.unary_union.centroid.y if not facilities_gdf.empty else grid_lines_gdf.geometry.unary_union.centroid.y
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
    
    split_lines = split_intersecting_edges(single_lines)
    print(f"Grid lines: {len(single_lines)} -> {len(split_lines)} after splitting")
    
    # Create nodes from line endpoints
    grid_nodes = set()
    for line in split_lines:
        coords = list(line.coords)
        grid_nodes.add(coords[0])
        grid_nodes.add(coords[-1])
    
    # Add nodes to graph
    for node in grid_nodes:
        G.add_node(node, type='grid_line')
    
    facility_nodes = set((point.x, point.y) for point in facilities_utm.geometry)
    for i, (idx, point) in enumerate(zip(facilities_gdf.index, facilities_utm.geometry)):
        node_coord = (point.x, point.y)
        G.add_node(node_coord, type='facility', facility_idx=idx)
    
    centroid_nodes = set((point.x, point.y) for point in centroids_utm.geometry)
    for i, (idx, point) in enumerate(zip(centroids_gdf.index, centroids_utm.geometry)):
        node_coord = (point.x, point.y)
        G.add_node(node_coord, type='pop_centroid', centroid_idx=idx)
    
    # Add edges from grid lines
    for line in split_lines:
        coords = list(line.coords)
        G.add_edge(coords[0], coords[-1], weight=line.length, edge_type='grid_infrastructure')
    
    # Connect facilities and centroids to nearest grid nodes
    grid_node_list = list(grid_nodes)
    
    for point_gdf, point_type in [(facilities_utm, 'facility'), (centroids_utm, 'pop_centroid')]:
        for i, point in enumerate(point_gdf.geometry):
            point_coord = (point.x, point.y)
            
            if grid_node_list:
                nearest_grid = min(grid_node_list, key=lambda n: Point(n).distance(point))
                distance = Point(nearest_grid).distance(point)
                
                # Connect with distance threshold
                max_distance = 50000  # 50km for both facilities and centroids
                if distance <= max_distance:
                    edge_type = 'centroid_to_grid' if point_type == 'pop_centroid' else 'grid_to_facility'
                    G.add_edge(point_coord, nearest_grid, weight=distance, edge_type=edge_type)
                    
    # Store UTM CRS for coordinate conversion
    G.graph['utm_crs'] = utm_crs
    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Stitch disconnected components within 10km (Jeju approach)
    G = stitch_network_components(G, max_distance_km=10)
    
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
    """Create polyline layer showing actual network paths for active supply connections only"""
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
    Calculate shortest path distance between two nodes in the network graph
    
    Parameters:
    - network_graph: NetworkX graph with weighted edges
    - start_node: Starting node coordinates (tuple)
    - end_node: Ending node coordinates (tuple)
    
    Returns:
    - Dictionary with distance info and path details, or None if no path exists
    """
    try:
        # Use NetworkX to get the path nodes (but we'll calculate distance manually)
        path_nodes = nx.shortest_path(network_graph, start_node, end_node, weight='weight')
        
        # Manually sum the weights of each edge in the path
        total_distance = 0
        path_segments = []
        
        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            # Get edge data
            edge_data = network_graph[current_node][next_node]
            segment_weight = edge_data.get('weight', 0)  # Should be in meters
            edge_type = edge_data.get('edge_type', 'unknown')
            
            total_distance += segment_weight
            path_segments.append({
                'from_node': current_node,
                'to_node': next_node,
                'weight': segment_weight,
                'edge_type': edge_type
            })
        
        return {
            'distance_km': total_distance / 1000.0,
            'path_nodes': path_nodes,
            'path_segments': path_segments,
            'total_segments': len(path_segments)
        }
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def process_centroid_distances_batch(centroid_batch, network_graph, facilities_gdf, centroid_mapping, facility_mapping, utm_crs):
    """
    Process a batch of centroids for distance calculations in parallel
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
    Calculate network distances from all centroids to nearby facilities using parallel processing
    
    Parameters:
    - centroids_gdf: GeoDataFrame of population centroids
    - facilities_gdf: GeoDataFrame of energy facilities
    - network_graph: NetworkX graph with grid infrastructure
    
    Returns:
    - List of dictionaries with centroid indices and their facility distances
    """
    print(f"Calculating distances for {len(centroids_gdf)} centroids using {MAX_WORKERS} parallel workers...")
    
    # Create coordinate mappings
    centroid_mapping = {}
    facility_mapping = {}
    
    for node, data in network_graph.nodes(data=True):
        if data.get('type') == 'pop_centroid':
            centroid_mapping[data.get('centroid_idx')] = node
        elif data.get('type') == 'facility':
            facility_mapping[data.get('facility_idx')] = node
    
    # Get UTM CRS for distance calculations
    center_point = centroids_gdf.geometry.unary_union.centroid
    utm_crs = get_utm_crs(center_point.x, center_point.y)
    
    # Split centroids into batches for parallel processing
    batch_size = max(1, len(centroids_gdf) // MAX_WORKERS)
    centroid_batches = []
    
    for i in range(0, len(centroids_gdf), batch_size):
        batch = centroids_gdf.iloc[i:i+batch_size]
        centroid_batches.append(batch)
    
    print(f"Processing {len(centroid_batches)} batches with average size {batch_size}")
    
    # Process batches in parallel using ThreadPoolExecutor (better for I/O bound operations)
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
                print(f"  Completed batch {batch_idx + 1}/{len(centroid_batches)}")
            except Exception as exc:
                print(f"  Batch {batch_idx} generated an exception: {exc}")
    
    print(f"Distance calculation completed for {len(all_results)} centroids")
    return all_results

def create_grid_lines_layer(grid_lines_gdf, network_graph, active_connections):
    """Create grid lines layer with grid_infrastructure, centroid_to_grid, grid_to_facility, component_stitch."""
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

def create_all_layers(centroids_gdf, facilities_gdf, grid_lines_gdf, network_graph, active_connections, country_iso3, capacity_factors=None, facility_supplied=None, facility_remaining=None):
    """Create all output layers for the GPKG file: centroids, facilities, grid_lines, and polylines"""
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
        facilities_simplified = facilities_gdf[['geometry', 'GEM unit/phase ID', 'Grouped_Type', 'Latitude', 'Longitude', 'Adjusted_Capacity_MW']].copy()
        facilities_simplified['GID_0'] = country_iso3
        
        # Use provided capacity factors or load them if not provided
        if capacity_factors is None:
            capacity_factors = load_conversion_rates(country_iso3)
        
        # Calculate annual energy production potential
        facilities_simplified['total_mwh'] = facilities_simplified.apply(
            lambda r: (r.get('Adjusted_Capacity_MW', 0) or 0) * 8760 * capacity_factors.get(r.get('Grouped_Type', ''), 0), axis=1
        )
        
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
    """
    Main function to process supply analysis for a country
    
    Args:
        country_iso3: ISO3 country code
        output_dir: Output directory for results
        test_mode: If True, generates full GPKG with all layers for testing.
                  If False, generates lightweight Parquet files for global analysis.
    
    Test mode creates 4 layers: centroids, facilities, grid_lines, polylines (full data)
    Production mode creates 4 parquet files with essential columns for global analysis:
    - centroids: geometry, GID_0, supplying_facility_type, supply_status
    - facilities: geometry, GID_0, Grouped_Type, total_mwh
    - grid_lines: geometry, GID_0, line_type
    - polylines: geometry, GID_0, facility_type
    """
    print(f"Processing {country_iso3}... (Mode: {'TEST' if test_mode else 'PRODUCTION'})")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load inputs
        admin_boundaries = load_admin_boundaries(country_iso3)
        country_bbox = get_country_bbox(admin_boundaries)
        centroids_gdf = load_population_centroids(country_bbox, admin_boundaries)
        facilities_gdf = load_energy_facilities(country_iso3, 2024)
        grid_lines_gdf = load_grid_lines(country_bbox, admin_boundaries)
        print(f"Loaded: {len(centroids_gdf)} centroids, {len(facilities_gdf)} facilities, {len(grid_lines_gdf)} grid lines")
        
        # Country tag
        centroids_gdf['GID_0'] = country_iso3
        
        # Demand projections
        centroids_gdf = load_population_and_demand_projections(centroids_gdf, country_iso3)
        
        # Default outputs
        network_graph = None
        active_connections = []
        capacity_factors = None
        facility_supplied = None
        facility_remaining = None
        
        # Process network and allocate supply if facilities and grid exist
        if not facilities_gdf.empty and not grid_lines_gdf.empty:
            # Load conversion rates once for reuse
            capacity_factors = load_conversion_rates(country_iso3)
            
            # Create network graph
            network_graph = create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf)
            
            # Calculate distances between centroids and facilities
            centroid_facility_distances = calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph)
            
            # Initialize facility remaining capacities (MWh/year) and track supplied amounts
            facility_remaining = {}
            facility_supplied = {}
            for idx, facility in facilities_gdf.iterrows():
                capacity_mw = facility.get('Adjusted_Capacity_MW', 0) or 0
                energy_type = facility.get('Grouped_Type', '') or ''
                cf = capacity_factors.get(energy_type, 0.30)
                total_mwh = capacity_mw * 8760 * cf if capacity_mw > 0 else 0
                facility_remaining[idx] = total_mwh
                facility_supplied[idx] = 0.0
            
            # Prepare centroid columns for supply allocation
            if 'supplying_facility_distance' not in centroids_gdf.columns:
                centroids_gdf['supplying_facility_distance'] = np.nan
            if 'supplying_facility_type' not in centroids_gdf.columns:
                centroids_gdf['supplying_facility_type'] = ''
            if 'supplying_facility_gem_id' not in centroids_gdf.columns:
                centroids_gdf['supplying_facility_gem_id'] = ''
            centroids_gdf['supply_received_mwh'] = 0.0
            centroids_gdf['supply_status'] = 'Not Filled'
            
            demand_col = 'Total_Demand_2024_centroid'
            
            # Initialize centroid tracking columns (supporting multiple facilities)
            centroids_gdf['supplying_facility_distance'] = ''
            centroids_gdf['supplying_facility_type'] = ''
            centroids_gdf['supplying_facility_gem_id'] = ''
            centroids_gdf['supply_received_mwh'] = 0.0
            centroids_gdf['supply_status'] = 'Not Filled'
            
            # Create facility-to-centroids mapping sorted by capacity (largest first)
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
            
            # Sort facilities by total capacity (largest first)
            facility_capacities.sort(key=lambda x: x['total_capacity'], reverse=True)
            print(f"Processing {len(facility_capacities)} facilities by capacity (largest first)...")
            
            # Process each facility and allocate to nearest centroids
            for facility_info in facility_capacities:
                facility_idx = facility_info['facility_idx']
                remaining_capacity = facility_remaining.get(facility_idx, 0)
                
                if remaining_capacity <= 0:
                    continue
                
                # Find centroids that can reach this facility
                facility_centroid_distances = []
                for item in centroid_facility_distances:
                    centroid_idx = item['centroid_idx']
                    
                    # Find this facility in the centroid's distance list
                    facility_distance_info = None
                    for distance_info in item['distances']:
                        if distance_info.get('facility_idx') == facility_idx:
                            facility_distance_info = distance_info
                            break
                    
                    if facility_distance_info:
                        centroid_demand = float(centroids_gdf.loc[centroid_idx, demand_col] or 0)
                        centroid_received = float(centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] or 0)
                        remaining_demand = centroid_demand - centroid_received
                        
                        if remaining_demand > 0:  # Only consider centroids with unmet demand
                            # Get the centroid's nearest facility distance (to any facility)
                            nearest_facility_distance = item['distances'][0]['distance_km'] if item['distances'] else float('inf')
                            
                            facility_centroid_distances.append({
                                'centroid_idx': centroid_idx,
                                'distance_to_this_facility': facility_distance_info.get('distance_km'),
                                'nearest_facility_distance': nearest_facility_distance,
                                'remaining_demand': remaining_demand,
                                'path_nodes': facility_distance_info.get('path_nodes', []),
                                'centroid_geom': centroids_gdf.loc[centroid_idx, 'geometry']
                            })
                
                # Sort centroids by their distance to this specific facility
                facility_centroid_distances.sort(key=lambda x: x['distance_to_this_facility'])
                
                # Allocate capacity to centroids prioritized by nearest facility distance
                for centroid_info in facility_centroid_distances:
                    if remaining_capacity <= 0:
                        break
                    
                    centroid_idx = centroid_info['centroid_idx']
                    remaining_demand = centroid_info['remaining_demand']
                    distance_km = centroid_info['distance_to_this_facility']
                    
                    # Allocate supply
                    allocated = min(remaining_demand, remaining_capacity)
                    centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] += allocated
                    facility_remaining[facility_idx] -= allocated
                    facility_supplied[facility_idx] += allocated
                    remaining_capacity -= allocated
                    
                    # Update centroid facility tracking (comma-separated for multiple facilities)
                    current_distances = centroids_gdf.loc[centroid_idx, 'supplying_facility_distance']
                    current_types = centroids_gdf.loc[centroid_idx, 'supplying_facility_type']
                    current_gem_ids = centroids_gdf.loc[centroid_idx, 'supplying_facility_gem_id']
                    
                    # Append new facility info
                    new_distance = f"{distance_km:.2f}"
                    new_type = facility_info['facility_type']
                    new_gem_id = facility_info['gem_id']
                    
                    if current_distances == '':
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_distance'] = new_distance
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_type'] = new_type
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_gem_id'] = new_gem_id
                    else:
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_distance'] = f"{current_distances}, {new_distance}"
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_type'] = f"{current_types}, {new_type}"
                        centroids_gdf.loc[centroid_idx, 'supplying_facility_gem_id'] = f"{current_gem_ids}, {new_gem_id}"
                    
                    # Track active connection for polylines
                    active_connections.append({
                        'centroid_idx': centroid_idx,
                        'facility_gem_id': facility_info['gem_id'],
                        'centroid_lat': centroid_info['centroid_geom'].y,
                        'centroid_lon': centroid_info['centroid_geom'].x,
                        'facility_lat': facility_info['geometry'].y,
                        'facility_lon': facility_info['geometry'].x,
                        'network_path': centroid_info['path_nodes'],
                        'supply_mwh': allocated,
                        'distance_km': distance_km,
                        'facility_type': facility_info['facility_type'],
                        'path_nodes': centroid_info['path_nodes']
                    })
            
            # Update supply status for all centroids
            for centroid_idx, centroid in centroids_gdf.iterrows():
                demand = float(centroid.get(demand_col, 0) or 0)
                received = float(centroid.get('supply_received_mwh', 0) or 0)
                
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
        layers = create_all_layers(centroids_gdf, facilities_gdf, grid_lines_gdf, network_graph, active_connections, country_iso3, capacity_factors, facility_supplied, facility_remaining)
        
        # Write outputs based on mode
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

def main():
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
