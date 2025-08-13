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
warnings.filterwarnings("ignore")

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
    
    country_name = country_data['NAME_0'].iloc[0] if 'NAME_0' in country_data.columns else country_iso3
    return country_data, country_name

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
        bbox_poly = gpd.GeoDataFrame([1], geometry=[
            gpd.GeoSeries([gpd.points_from_xy([minx, maxx, maxx, minx, minx], 
                                             [miny, miny, maxy, maxy, miny])]).apply(lambda x: x.convex_hull)[0]
        ], crs=COMMON_CRS)
        
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
    for i, node in enumerate(facility_nodes):
        G.add_node(node, type='facility', facility_idx=facilities_gdf.index[i])
    
    centroid_nodes = set((point.x, point.y) for point in centroids_utm.geometry)
    for i, node in enumerate(centroid_nodes):
        G.add_node(node, type='pop_centroid', centroid_idx=centroids_gdf.index[i])
    
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
    
    return G

def calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph):
    """Calculate distances from centroids to facilities using network paths"""
    centroid_facility_distances = []
    
    # Create coordinate mappings
    centroid_mapping = {}
    facility_mapping = {}
    
    for node, data in network_graph.nodes(data=True):
        if data.get('type') == 'pop_centroid':
            centroid_mapping[data.get('centroid_idx')] = node
        elif data.get('type') == 'facility':
            facility_mapping[data.get('facility_idx')] = node
    
    for centroid_idx, centroid in centroids_gdf.iterrows():
        network_centroid = centroid_mapping.get(centroid_idx)
        distances = []
        
        if network_centroid is not None:
            for facility_idx, facility in facilities_gdf.iterrows():
                network_facility = facility_mapping.get(facility_idx)
                
                if network_facility is not None:
                    try:
                        # Calculate network path distance
                        path_length = nx.shortest_path_length(
                            network_graph, network_centroid, network_facility, weight='weight'
                        )
                        path = nx.shortest_path(
                            network_graph, network_centroid, network_facility, weight='weight'
                        )
                        
                        distances.append({
                            'facility_idx': facility_idx,
                            'distance_km': path_length / 1000.0,
                            'network_path': path,
                            'facility_type': facility.get('Grouped_Type', ''),
                            'facility_capacity': facility.get('Adjusted_Capacity_MW', 0),
                            'facility_lat': facility.geometry.y,
                            'facility_lon': facility.geometry.x
                        })
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        
        distances.sort(key=lambda x: x['distance_km'])
        centroid_facility_distances.append({
            'centroid_idx': centroid_idx,
            'distances': distances
        })
    
    return centroid_facility_distances

def allocate_supply(centroids_gdf, facilities_gdf, centroid_facility_distances, country_iso3):
    """Allocate supply to centroids based on distance and capacity"""
    # Load capacity factors
    default_factors = {'Solar': 0.25, 'Wind': 0.35, 'Hydro': 0.45, 'Other Renewables': 0.30, 'Nuclear': 0.90, 'Fossil': 0.50}
    
    # Initialize facility capacities
    facility_remaining = {}
    for idx, facility in facilities_gdf.iterrows():
        capacity_mw = facility.get('Adjusted_Capacity_MW', 0)
        energy_type = facility.get('Grouped_Type', '')
        capacity_factor = default_factors.get(energy_type, 0.30)
        annual_mwh = capacity_mw * 8760 * capacity_factor if capacity_mw > 0 else 0
        facility_remaining[idx] = annual_mwh
    
    # Initialize centroid columns
    centroids_gdf['nearest_facility_distance'] = np.nan
    centroids_gdf['nearest_facility_type'] = ''
    centroids_gdf['supply_received_mwh'] = 0.0
    centroids_gdf['supply_status'] = 'Not Filled'
    
    connection_lines = []
    
    # Process centroids by demand (highest first)
    demand_col = 'Total_Demand_2024_centroid'
    centroids_by_demand = centroids_gdf.sort_values(demand_col, ascending=False)
    
    for _, centroid in centroids_by_demand.iterrows():
        centroid_idx = centroid.name
        centroid_demand = centroid.get(demand_col, 0)
        
        if centroid_demand <= 0:
            continue
        
        # Find distances for this centroid
        centroid_distances = None
        for dist_info in centroid_facility_distances:
            if dist_info['centroid_idx'] == centroid_idx:
                centroid_distances = dist_info['distances']
                break
        
        if not centroid_distances:
            continue
        
        # Set nearest facility info
        nearest = centroid_distances[0]
        centroids_gdf.loc[centroid_idx, 'nearest_facility_distance'] = nearest['distance_km']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_type'] = nearest['facility_type']
        
        # Allocate supply from nearest available facilities
        remaining_demand = centroid_demand
        
        for facility_info in centroid_distances:
            if remaining_demand <= 0:
                break
            
            facility_idx = facility_info['facility_idx']
            available_supply = facility_remaining.get(facility_idx, 0)
            
            if available_supply > 0:
                allocated = min(remaining_demand, available_supply)
                centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] += allocated
                facility_remaining[facility_idx] -= allocated
                remaining_demand -= allocated
                
                # Create connection line
                connection_lines.append({
                    'centroid_idx': centroid_idx,
                    'facility_idx': facility_idx,
                    'centroid_lat': centroid.geometry.y,
                    'centroid_lon': centroid.geometry.x,
                    'facility_lat': facility_info['facility_lat'],
                    'facility_lon': facility_info['facility_lon'],
                    'network_path': facility_info.get('network_path'),
                    'supply_mwh': allocated
                })
        
        # Update status
        if remaining_demand <= 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Filled'
        elif centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] > 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Partially Filled'
    
    return centroids_gdf, connection_lines

def create_grid_lines_layer(grid_lines_gdf, network_graph, connection_lines):
    """Create comprehensive grid lines layer with three types"""
    all_geometries = []
    all_attributes = []
    
    # 1. GRID_INFRASTRUCTURE: Original grid lines
    if not grid_lines_gdf.empty:
        for idx, grid_line in grid_lines_gdf.iterrows():
            all_geometries.append(grid_line.geometry)
            all_attributes.append({
                'line_type': 'grid_infrastructure',
                'line_id': f"grid_{idx}",
                'from_type': 'grid',
                'to_type': 'grid',
                'description': 'Original grid infrastructure'
            })
    
    # 2. CENTROID_TO_GRID and GRID_TO_FACILITY from network graph
    if network_graph is not None:
        utm_crs = network_graph.graph.get('utm_crs', 'EPSG:3857')
        
        for edge in network_graph.edges(data=True):
            node1, node2, edge_data = edge
            edge_type = edge_data.get('edge_type', 'unknown')
            
            if edge_type in ['centroid_to_grid', 'grid_to_facility']:
                try:
                    # Convert UTM coordinates to WGS84
                    coords_utm = gpd.GeoSeries([Point(node1), Point(node2)], crs=utm_crs)
                    coords_wgs84 = coords_utm.to_crs(COMMON_CRS)
                    
                    line_geom = LineString([(coords_wgs84.iloc[0].x, coords_wgs84.iloc[0].y),
                                          (coords_wgs84.iloc[1].x, coords_wgs84.iloc[1].y)])
                    
                    all_geometries.append(line_geom)
                    all_attributes.append({
                        'line_type': edge_type,
                        'line_id': f"{edge_type}_{len(all_attributes)}",
                        'from_type': 'centroid' if edge_type == 'centroid_to_grid' else 'grid',
                        'to_type': 'grid' if edge_type == 'centroid_to_grid' else 'facility',
                        'description': f"Network connection: {edge_type.replace('_', ' to ')}"
                    })
                except Exception as e:
                    print(f"Warning: Failed to create {edge_type} line: {e}")
    
    # Create grid lines GeoDataFrame
    if all_geometries:
        grid_lines_layer = gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs=COMMON_CRS)
        
        # Print summary
        type_counts = grid_lines_layer['line_type'].value_counts()
        print(f"Grid lines created: {len(grid_lines_layer)} total")
        for line_type, count in type_counts.items():
            print(f"  {line_type}: {count}")
        
        return grid_lines_layer
    
    return gpd.GeoDataFrame()

def process_country_supply(country_iso3, output_dir="outputs_per_country"):
    """Main function to process supply analysis for a country"""
    print(f"Processing {country_iso3}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load data
        admin_boundaries, country_name = load_admin_boundaries(country_iso3)
        country_bbox = get_country_bbox(admin_boundaries)
        
        centroids_gdf = load_population_centroids(country_bbox, admin_boundaries)
        facilities_gdf = load_energy_facilities(country_iso3, 2024)
        grid_lines_gdf = load_grid_lines(country_bbox, admin_boundaries)
        
        print(f"Loaded: {len(centroids_gdf)} centroids, {len(facilities_gdf)} facilities, {len(grid_lines_gdf)} grid lines")
        
        # Add country identifiers
        centroids_gdf['GID_0'] = country_iso3
        centroids_gdf['NAME_0'] = country_name
        
        # Calculate demand (placeholder - using population as proxy)
        total_pop = centroids_gdf['Population_centroid'].sum()
        per_capita_demand = 3.2  # MWh per person per year
        total_demand = total_pop * per_capita_demand
        centroids_gdf['Total_Demand_2024_centroid'] = (centroids_gdf['Population_centroid'] / total_pop) * total_demand
        
        connection_lines = []
        
        # Network analysis if we have facilities and grid
        if not facilities_gdf.empty and not grid_lines_gdf.empty:
            print("Creating network graph...")
            network_graph = create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf)
            
            print("Calculating facility distances...")
            centroid_facility_distances = calculate_facility_distances(centroids_gdf, facilities_gdf, network_graph)
            
            print("Allocating supply...")
            centroids_gdf, connection_lines = allocate_supply(centroids_gdf, facilities_gdf, centroid_facility_distances, country_iso3)
        else:
            print("Skipping network analysis - missing facilities or grid data")
            network_graph = None
        
        # Create output layers
        output_file = output_path / f"supply_analysis_{country_iso3}.gpkg"
        
        # Layer 1: Centroids
        centroids_simplified = centroids_gdf[['geometry', 'GID_0', 'NAME_0', 'Population_centroid', 
                                            'Total_Demand_2024_centroid', 'nearest_facility_distance',
                                            'nearest_facility_type', 'supply_received_mwh', 'supply_status']].copy()
        centroids_simplified.to_file(output_file, layer="centroids", driver="GPKG")
        
        # Layer 2: Grid Lines (three types)
        grid_lines_layer = create_grid_lines_layer(grid_lines_gdf, network_graph, connection_lines)
        if not grid_lines_layer.empty:
            grid_lines_layer.to_file(output_file, layer="grid_lines", driver="GPKG")
        
        # Layer 3: Facilities
        if not facilities_gdf.empty:
            facilities_simplified = facilities_gdf[['geometry', 'GEM unit/phase ID', 'Grouped_Type', 
                                                  'Latitude', 'Longitude', 'Adjusted_Capacity_MW']].copy()
            facilities_simplified['GID_0'] = country_iso3
            facilities_simplified['NAME_0'] = country_name
            facilities_simplified.to_file(output_file, layer="facilities", driver="GPKG")
        
        print(f"Results saved to {output_file}")
        print(f"GPKG contains 3 layers: centroids, grid_lines, facilities")
        
        # Print summary
        if 'supply_status' in centroids_gdf.columns:
            status_counts = centroids_gdf['supply_status'].value_counts()
            print(f"Supply status: {dict(status_counts)}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"Error processing {country_iso3}: {e}")
        return None

def main():
    """Command line interface"""
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
