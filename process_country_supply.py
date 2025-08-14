#!/usr/bin/env python3
"""
Clean supply analysis per country with network-based grid lines
Produces exactly 3 layers: centroids, facilities, grid_lines
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
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Suppress warnings
# warnings.filterwarnings("ignore")

# Constants
COMMON_CRS = "EPSG:4326"  # WGS84 for input/output
YEARS = [2024, 2030, 2050]
SUPPLY_TYPES = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]

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
    """Load and process population centroids from raster"""
    minx, miny, maxx, maxy = country_bbox
    
    with rasterio.open('bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif') as src:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        pop_data = src.read(1, window=window)
        windowed_transform = rasterio.windows.transform(window, src.transform)
        
        rows, cols = pop_data.shape
        centroids_x, centroids_y = [], []
        
        for row in range(rows):
            for col in range(cols):
                x, y = rasterio.transform.xy(windowed_transform, row, col)
                centroids_x.append(x)
                centroids_y.append(y)
        
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
    """Split lines at intersections"""
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

def stitch_network_components(network_graph, max_distance_km=50):
    """
    Connect disconnected network components by adding edges between closest nodes
    of different components that are within max_distance_km of each other.
    
    Parameters:
    - network_graph: NetworkX graph
    - max_distance_km: Maximum distance in kilometers to connect components
    
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
    
    # For each pair of significant components, find the closest nodes and connect if within distance threshold
    for i in range(len(significant_components)):
        for j in range(i + 1, len(significant_components)):
            component1 = significant_components[i]
            component2 = significant_components[j]
            
            min_distance = float('inf')
            best_connection = None
            
            # Find the closest pair of nodes between the two components
            for node1 in component1:
                for node2 in component2:
                    if isinstance(node1, tuple) and isinstance(node2, tuple) and len(node1) == 2 and len(node2) == 2:
                        # Calculate Euclidean distance between nodes (already in UTM)
                        distance = ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_connection = (node1, node2)
            
            # Add edge if within distance threshold
            if best_connection and min_distance <= max_distance_m:
                node1, node2 = best_connection
                network_graph.add_edge(node1, node2, weight=min_distance, edge_type='component_stitch')
                connections_added += 1
                print(f"  Connected components {i+1} and {j+1}: {min_distance/1000:.2f}km")
    
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
                max_distance = 50000  # 50km for facilities, unlimited for centroids
                if point_type == 'pop_centroid' or distance <= max_distance:
                    edge_type = 'centroid_to_grid' if point_type == 'pop_centroid' else 'grid_to_facility'
                    G.add_edge(point_coord, nearest_grid, weight=distance, edge_type=edge_type)
    
    # Store UTM CRS for coordinate conversion
    G.graph['utm_crs'] = utm_crs
    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Stitch disconnected components within 50km
    G = stitch_network_components(G, max_distance_km=50)
    
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

def calculate_centroid_facility_distances_manual(centroids_gdf, facilities_gdf, network_graph):
    """
    Calculate network distances from centroids to facilities using manual distance calculation.
    
    Parameters:
    - centroids_gdf: GeoDataFrame of centroids
    - facilities_gdf: GeoDataFrame of facilities  
    - network_graph: NetworkX graph with grid infrastructure
    
    Returns:
    - Dictionary with centroid indices as keys and distance data as values
    """
    if facilities_gdf.empty or network_graph is None:
        return {}
    
    # Get UTM CRS for accurate distance calculations
    center_point = centroids_gdf.geometry.unary_union.centroid
    utm_crs = get_utm_crs(center_point.x, center_point.y)
    
    distances = {}
    
    for centroid_idx, centroid_row in centroids_gdf.iterrows():
        # Find facilities within 100km radius for performance
        nearby_facilities = find_available_facilities_within_radius(
            centroid_row.geometry, facilities_gdf, utm_crs, radius_km=100
        )
        
        if nearby_facilities.empty:
            distances[centroid_idx] = {}
            continue
        
        # Find nearest network node to centroid
        centroid_node = find_nearest_network_node(centroid_row.geometry, network_graph)
        if centroid_node is None:
            distances[centroid_idx] = {}
            continue
        
        facility_distances = {}
        
        for facility_idx, facility_row in nearby_facilities.iterrows():
            # Find nearest network node to facility
            facility_node = find_nearest_network_node(facility_row.geometry, network_graph)
            if facility_node is None:
                continue
            
            try:
                # Use manual distance calculation
                distance_result = calculate_network_distance_manual(network_graph, centroid_node, facility_node)
                
                if distance_result is not None:
                    facility_distances[facility_idx] = {
                        'distance_km': distance_result['distance_km'],
                        'path_nodes': distance_result['path_nodes'],
                        'path_segments': distance_result['path_segments'],
                        'total_segments': distance_result['total_segments'],
                        'facility_type': facility_row.get('Grouped_Type', 'Unknown'),
                        'capacity_mw': facility_row.get('Adjusted_Capacity_MW', 0),
                        'total_mwh': facility_row.get('total_mwh', 0),
                        'centroid_node': centroid_node,
                        'facility_node': facility_node
                    }
            except Exception as e:
                print(f"Error calculating distance from centroid {centroid_idx} to facility {facility_idx}: {e}")
                continue
        
        distances[centroid_idx] = facility_distances
    
    return distances

def create_candidate_polyline_layer(centroid_facility_distances, network_graph, country_iso3):
    """Deprecated: not used. Returns empty GeoDataFrame."""
    return gpd.GeoDataFrame()

def create_polyline_layer(active_connections, network_graph, country_iso3):
    """Create polyline layer showing actual network paths for active supply connections only"""
    all_geometries = []
    all_attributes = []
    
    if not active_connections:
        return gpd.GeoDataFrame()
    
    utm_crs = network_graph.graph.get('utm_crs', 'EPSG:3857')
    
    # Reduced verbosity: no count print
    
    for connection in active_connections:
        centroid_idx = connection.get('centroid_idx')
        facility_idx = connection.get('facility_idx')
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
                    'facility_idx': facility_idx,
                    'facility_type': connection.get('facility_type'),
                    'distance_km': connection.get('distance_km'),
                    'supply_mwh': connection.get('supply_mwh'),
                    'connection_id': f"C{centroid_idx}_F{facility_idx}",
                    'active_supply': 'Yes'
                })
        except Exception as e:
            print(f"Warning: Failed to create polyline for centroid {centroid_idx} to facility {facility_idx}: {e}")
    
    if all_geometries:
        polylines_layer = gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs=COMMON_CRS)
        polylines_layer['GID_0'] = country_iso3
        
        # Safe column selection to avoid KeyErrors
        preferred = ['geometry', 'GID_0', 'connection_id', 'centroid_idx', 'facility_idx', 'facility_type', 'distance_km', 'supply_mwh', 'active_supply']
        existing = [c for c in preferred if c in polylines_layer.columns]
        others = [c for c in polylines_layer.columns if c not in existing]
        polylines_layer = polylines_layer[existing + others]
        return polylines_layer
    
    return gpd.GeoDataFrame()

def calculate_network_distance_manual(network_graph, start_node, end_node):
    """Calculate distance by manually finding and summing network segments"""
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

def calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph):
    """Calculate distances from centroids to facilities using manual network calculation"""
    centroid_facility_distances = []
    
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
    
    # Removed verbose progress print
    # print(f"Processing {len(centroids_gdf)} centroids for distance calculation...")
    
    for centroid_idx, centroid in centroids_gdf.iterrows():
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
                            'gem_id': facility.get('GEM unit/phase ID', ''),
                            'euclidean_distance_km': euclidean_distance,
                            'network_path': distance_result['path_nodes']
                        })
        distances.sort(key=lambda x: x['distance_km'])
        centroid_facility_distances.append({'centroid_idx': centroid_idx, 'distances': distances})
    return centroid_facility_distances

def allocate_supply(centroids_gdf, facilities_gdf, centroid_facility_distances, country_iso3):
    """Allocate supply to centroids based on distance-only priority and facility capacity."""
    capacity_factors = load_conversion_rates(country_iso3)
    
    # Initialize facility remaining capacities (MWh/year)
    facility_remaining = {}
    for idx, facility in facilities_gdf.iterrows():
        capacity_mw = facility.get('Adjusted_Capacity_MW', 0) or 0
        energy_type = facility.get('Grouped_Type', '') or ''
        cf = capacity_factors.get(energy_type, 0.30)
        facility_remaining[idx] = capacity_mw * 8760 * cf if capacity_mw > 0 else 0
    
    # Prepare centroid columns
    if 'nearest_facility_distance' not in centroids_gdf.columns:
        centroids_gdf['nearest_facility_distance'] = np.nan
    if 'nearest_facility_type' not in centroids_gdf.columns:
        centroids_gdf['nearest_facility_type'] = ''
    centroids_gdf['supply_received_mwh'] = 0.0
    centroids_gdf['supply_status'] = 'Not Filled'
    
    active_connections = []
    demand_col = 'Total_Demand_2024_centroid'
    
    # Sort by demand descending
    for _, centroid in centroids_gdf.sort_values(demand_col, ascending=False).iterrows():
        centroid_idx = centroid.name
        remaining_demand = float(centroid.get(demand_col, 0) or 0)
        if remaining_demand <= 0:
            continue
        
        # Find precomputed distances for this centroid
        centroid_distances = None
        for item in centroid_facility_distances:
            if item['centroid_idx'] == centroid_idx:
                centroid_distances = item['distances']
                break
        if not centroid_distances:
            continue
        
        # Filter facilities with remaining capacity and sort by shortest distance
        available = []
        for fi in centroid_distances:
            fidx = fi.get('facility_idx')
            if facility_remaining.get(fidx, 0) > 0:
                available.append(fi)
        available.sort(key=lambda x: x.get('distance_km', float('inf')))
        if not available:
            continue
        
        # Record nearest facility info
        nearest = available[0]
        centroids_gdf.loc[centroid_idx, 'nearest_facility_distance'] = nearest.get('distance_km', np.nan)
        centroids_gdf.loc[centroid_idx, 'nearest_facility_type'] = nearest.get('facility_type', '')
        
        # Allocate from closest facilities until demand is met or supply exhausted
        for fi in available:
            if remaining_demand <= 0:
                break
            fidx = fi.get('facility_idx')
            avail_supply = float(facility_remaining.get(fidx, 0) or 0)
            if avail_supply <= 0:
                continue
            allocated = min(remaining_demand, avail_supply)
            centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] += allocated
            facility_remaining[fidx] = avail_supply - allocated
            remaining_demand -= allocated
            
            # Track active connection for polylines
            active_connections.append({
                'centroid_idx': centroid_idx,
                'facility_idx': fidx,
                'centroid_lat': centroid.geometry.y,
                'centroid_lon': centroid.geometry.x,
                'facility_lat': fi.get('facility_lat'),
                'facility_lon': fi.get('facility_lon'),
                'network_path': fi.get('network_path'),
                'supply_mwh': allocated,
                'distance_km': fi.get('distance_km'),
                'facility_type': fi.get('facility_type'),
                'path_nodes': fi.get('path_nodes', [])
            })
        
        # Update supply status
        if remaining_demand <= 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Filled'
        elif centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] > 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Partially Filled'
    
    return centroids_gdf, active_connections

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

def process_country_supply(country_iso3, output_dir="outputs_per_country"):
    """Main function to process supply analysis for a country"""
    print(f"Processing {country_iso3}...")
    
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
        polylines_layer = gpd.GeoDataFrame()
        active_connections = []
        
        if not facilities_gdf.empty and not grid_lines_gdf.empty:
            network_graph = create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf)
            centroid_facility_distances = calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph)
            centroids_gdf, active_connections = allocate_supply(centroids_gdf, facilities_gdf, centroid_facility_distances, country_iso3)
            polylines_layer = create_polyline_layer(active_connections, network_graph, country_iso3)
        else:
            # Initialize supply columns if network is skipped
            centroids_gdf['nearest_facility_distance'] = np.nan
            centroids_gdf['nearest_facility_type'] = ''
            centroids_gdf['supply_received_mwh'] = 0.0
            centroids_gdf['supply_status'] = 'Not Filled'
        
        # Write outputs
        output_file = output_path / f"supply_analysis_{country_iso3}.gpkg"
        
        # Centroids layer
        centroid_columns = ['geometry', 'GID_0', 'Population_centroid',
                            'Population_2024_centroid', 'Population_2030_centroid', 'Population_2050_centroid',
                            'Total_Demand_2024_centroid', 'Total_Demand_2030_centroid', 'Total_Demand_2050_centroid',
                            'nearest_facility_distance', 'nearest_facility_type', 'supply_received_mwh', 'supply_status']
        available_columns = [c for c in centroid_columns if c in centroids_gdf.columns]
        centroids_gdf[available_columns].to_file(output_file, layer="centroids", driver="GPKG")
        
        # Grid lines layer
        grid_lines_layer = create_grid_lines_layer(grid_lines_gdf, network_graph, active_connections)
        if not grid_lines_layer.empty:
            grid_lines_layer['GID_0'] = country_iso3
            cols = ['geometry', 'GID_0'] + [c for c in grid_lines_layer.columns if c not in ['geometry', 'GID_0']]
            grid_lines_layer = grid_lines_layer[cols]
            grid_lines_layer.to_file(output_file, layer="grid_lines", driver="GPKG")
        
        # Facilities layer
        if not facilities_gdf.empty:
            facilities_simplified = facilities_gdf[['geometry', 'GEM unit/phase ID', 'Grouped_Type', 'Latitude', 'Longitude', 'Adjusted_Capacity_MW']].copy()
            facilities_simplified['GID_0'] = country_iso3
            cf_map = load_conversion_rates(country_iso3)
            facilities_simplified['total_mwh'] = facilities_simplified.apply(
                lambda r: (r.get('Adjusted_Capacity_MW', 0) or 0) * 8760 * cf_map.get(r.get('Grouped_Type', ''), 0), axis=1
            )
            cols = ['geometry', 'GID_0'] + [c for c in facilities_simplified.columns if c not in ['geometry', 'GID_0']]
            facilities_simplified = facilities_simplified[cols]
            facilities_simplified.to_file(output_file, layer="facilities", driver="GPKG")
        
        # Polylines layer
        if not polylines_layer.empty:
            polylines_layer.to_file(output_file, layer="polylines", driver="GPKG")
        
        print(f"Results saved to {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"Error processing {country_iso3}: {e}")
        return None

def main():
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
