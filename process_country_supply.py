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
from pyproj import Transformer
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

# Enhanced fallback connection function removed - using only Strategy 1 (connect_disconnected_components)
# for optimal performance. Strategy 1 proactively bridges disconnected components during network creation.

def calculate_facility_distances_chunk(centroid_chunk, facilities_gdf, network_graph=None):
    """Calculate distances from centroids to ALL facilities (not just nearest) for supply allocation"""
    results = []
    
    # If we have a network graph, we need to map the original coordinates to the UTM network coordinates
    centroid_to_network_mapping = {}
    facility_to_network_mapping = {}
    
    if network_graph is not None:
        # Create mapping from original coordinates to network coordinates
        for node, data in network_graph.nodes(data=True):
            if data.get('type') == 'pop_centroid':
                original_lat = data.get('original_lat')
                original_lon = data.get('original_lon')
                if original_lat is not None and original_lon is not None:
                    centroid_to_network_mapping[(original_lon, original_lat)] = node
            elif data.get('type') == 'facility':
                original_lat = data.get('original_lat')
                original_lon = data.get('original_lon')
                if original_lat is not None and original_lon is not None:
                    facility_to_network_mapping[(original_lon, original_lat)] = node
    
    for idx, centroid in centroid_chunk.iterrows():
        centroid_distances = []
        
        if network_graph is not None:
            # Use network-based distance calculation
            centroid_coord = (centroid.geometry.x, centroid.geometry.y)
            network_centroid = centroid_to_network_mapping.get(centroid_coord)
            
            # Check if centroid is in the network graph
            if network_centroid is not None:
                # Calculate distance to ALL facilities in the network
                for fac_idx, facility in facilities_gdf.iterrows():
                    facility_coord = (facility.geometry.x, facility.geometry.y)
                    network_facility = facility_to_network_mapping.get(facility_coord)
                    
                    if network_facility is not None:
                                try:
                                    # Calculate shortest path distance using NetworkX (in meters)
                                    network_distance_meters = nx.shortest_path_length(
                                        network_graph, 
                                        network_centroid, 
                                        network_facility, 
                                        weight='weight'
                                    )
                                    
                                    # Get the actual path for visualization
                                    network_path = nx.shortest_path(
                                        network_graph,
                                        network_centroid,
                                        network_facility,
                                        weight='weight'
                                    )
                                    
                                    # Convert meters to kilometers
                                    network_distance_km = network_distance_meters / 1000.0
                                    
                                    centroid_distances.append({
                                        'centroid_idx': idx,
                                        'facility_idx': fac_idx,
                                        'distance': network_distance_km,  # Now in kilometers
                                        'type': facility.get('Grouped_Type', ''),
                                        'capacity': facility.get('Adjusted_Capacity_MW', np.nan),
                                        'gem_id': facility.get('GEM unit/phase ID', ''),
                                        'facility_lat': facility.get('Latitude', np.nan),
                                        'facility_lon': facility.get('Longitude', np.nan),
                                        'network_path': network_path,  # Store the actual path
                                        'connection_type': 'network_path'
                                    })
                                except (nx.NetworkXNoPath, nx.NodeNotFound):
                                    # No network path available - skip this facility
                                    # Note: Most cases should be handled by Strategy 1 (connect_disconnected_components)
                                    continue                # Sort by distance for allocation priority
                centroid_distances.sort(key=lambda x: x['distance'])
            else:
                # Centroid not in network
                centroid_distances = []
        else:
            # Fallback to Euclidean distance calculation if no network provided
            if not centroid_chunk.empty and not facilities_gdf.empty:
                # Get approximate center to determine appropriate UTM zone
                center_lon = centroid.geometry.x
                center_lat = centroid.geometry.y
                
                # Calculate UTM zone
                utm_zone = int((center_lon + 180) / 6) + 1
                utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
                
                # Project both datasets to UTM for accurate distance calculation
                centroid_utm = gpd.GeoSeries([centroid.geometry], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
                facilities_utm = facilities_gdf.to_crs(utm_crs)
                
                # Calculate distances to all facilities
                for fac_idx, facility in facilities_gdf.iterrows():
                    facility_utm = facilities_utm.loc[fac_idx]
                    distance_meters = facility_utm.geometry.distance(centroid_utm)
                    distance_km = distance_meters / 1000.0  # Convert to kilometers
                    
                    centroid_distances.append({
                        'centroid_idx': idx,
                        'facility_idx': fac_idx,
                        'distance': distance_km,  # Now in kilometers
                        'type': facility.get('Grouped_Type', ''),
                        'capacity': facility.get('Adjusted_Capacity_MW', np.nan),
                        'gem_id': facility.get('GEM unit/phase ID', ''),
                        'facility_lat': facility.get('Latitude', np.nan),
                        'facility_lon': facility.get('Longitude', np.nan)
                    })
                
                # Sort by distance for allocation priority
                centroid_distances.sort(key=lambda x: x['distance'])
        
        results.append({
            'centroid_idx': idx,
            'facility_distances': centroid_distances
        })
    
    return results

def load_conversion_rates(country_iso3):
    """Load country-specific conversion rates for each energy type"""
    try:
        # Load conversion rates data
        conversion_df = pd.read_excel("outputs_processed_data/p1_a_ember_gem_2024.xlsx", 
                                    sheet_name="Grouped_cur")
        
        # Filter for the specified country
        country_conversion = conversion_df[conversion_df['Country Code'] == country_iso3]
        
        if country_conversion.empty:
            print(f"Warning: No conversion rates found for {country_iso3}, using default rates")
            # Default conversion rates if country not found
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
        for energy_type in energy_types:
            conv_col = f"{energy_type}_ConvRate"
            if conv_col in country_data:
                rate = country_data[conv_col]
                if pd.notna(rate) and rate > 0:
                    conversion_rates[energy_type] = rate
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
                    print(f"Warning: Using default conversion rate for {energy_type}: {conversion_rates[energy_type]}")
            else:
                print(f"Warning: Conversion rate column {conv_col} not found")
                conversion_rates[energy_type] = 0.30  # Default fallback
        
        print(f"Loaded conversion rates for {country_iso3}:")
        for energy_type, rate in conversion_rates.items():
            print(f"  {energy_type}: {rate:.3f}")
        
        return conversion_rates
        
    except Exception as e:
        print(f"Error loading conversion rates for {country_iso3}: {e}")
        # Return default conversion rates
        return {
            'Solar': 0.25,
            'Wind': 0.35,
            'Hydro': 0.45,
            'Other Renewables': 0.30,
            'Nuclear': 0.90,
            'Fossil': 0.50
        }

def allocate_supply_to_centroids(centroids_gdf, facilities_gdf, centroid_facility_distances, country_iso3, network_graph=None, year=2024):
    """Allocate facility supply to centroids based on demand and distance priority"""
    print(f"Allocating supply for year {year}...")
    
    # Load country-specific conversion rates
    conversion_rates = load_conversion_rates(country_iso3)
    
    # Initialize facility remaining capacity tracking
    facility_remaining = {}
    for idx, facility in facilities_gdf.iterrows():
        # Get facility details
        capacity_mw = facility.get('Adjusted_Capacity_MW', 0)
        energy_type = facility.get('Grouped_Type', '')
        
        if pd.isna(capacity_mw) or capacity_mw <= 0:
            capacity_mw = 0
            annual_generation_mwh = 0
        else:
            # Use country-specific and energy-type-specific conversion rate
            conv_rate = conversion_rates.get(energy_type, 0.30)  # Default 30% if type not found
            annual_generation_mwh = capacity_mw * 8760 * conv_rate
        
        facility_remaining[idx] = annual_generation_mwh
    
    total_facility_capacity = sum(facility_remaining.values())
    print(f"Total facility capacity available: {total_facility_capacity:,.0f} MWh")
    
    # Print capacity breakdown by energy type
    capacity_by_type = {}
    for idx, facility in facilities_gdf.iterrows():
        energy_type = facility.get('Grouped_Type', 'Unknown')
        capacity = facility_remaining.get(idx, 0)
        if energy_type not in capacity_by_type:
            capacity_by_type[energy_type] = 0
        capacity_by_type[energy_type] += capacity
    
    print(f"Capacity breakdown by energy type:")
    for energy_type, capacity in sorted(capacity_by_type.items()):
        if capacity > 0:
            print(f"  {energy_type}: {capacity:,.0f} MWh")
    
    # Initialize centroid allocation tracking
    centroids_gdf['supply_status'] = 'Not Filled'
    centroids_gdf['energy_types_supplied'] = ''
    centroids_gdf['facilities_serving'] = ''
    centroids_gdf['supply_received_mwh'] = 0.0
    centroids_gdf['supply_shortage_mwh'] = 0.0
    centroids_gdf['nearest_facility_distance'] = np.nan
    centroids_gdf['nearest_facility_type'] = ''
    centroids_gdf['nearest_facility_capacity'] = np.nan
    centroids_gdf['nearest_facility_gem_id'] = ''
    centroids_gdf['nearest_facility_lat'] = np.nan
    centroids_gdf['nearest_facility_lon'] = np.nan
    
    demand_col = f"Total_Demand_{year}_centroid"
    if demand_col not in centroids_gdf.columns:
        print(f"Warning: Demand column {demand_col} not found, using zero demand")
        centroids_gdf[demand_col] = 0.0
    
    total_demand = centroids_gdf[demand_col].sum()
    print(f"Total demand to satisfy: {total_demand:,.0f} MWh")
    
    # Create connection lines for visualization
    connection_lines = []
    
    # Process each centroid in order of demand (highest first)
    centroids_by_demand = centroids_gdf.sort_values(demand_col, ascending=False)
    
    for _, centroid in centroids_by_demand.iterrows():
        centroid_idx = centroid.name
        centroid_demand = centroid[demand_col]
        
        if centroid_demand <= 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'No Demand'
            continue
        
        # Find facility distances for this centroid
        centroid_distances = None
        for dist_info in centroid_facility_distances:
            if dist_info['centroid_idx'] == centroid_idx:
                centroid_distances = dist_info['facility_distances']
                break
        
        if not centroid_distances:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'No Facilities Accessible'
            centroids_gdf.loc[centroid_idx, 'supply_shortage_mwh'] = centroid_demand
            continue
        
        # Set nearest facility info (first in sorted list)
        nearest_facility = centroid_distances[0]
        centroids_gdf.loc[centroid_idx, 'nearest_facility_distance'] = nearest_facility['distance']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_type'] = nearest_facility['type']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_capacity'] = nearest_facility['capacity']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_gem_id'] = nearest_facility['gem_id']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_lat'] = nearest_facility['facility_lat']
        centroids_gdf.loc[centroid_idx, 'nearest_facility_lon'] = nearest_facility['facility_lon']
        
        # Allocate supply from facilities in distance order
        remaining_demand = centroid_demand
        energy_types_used = []
        facilities_used = []
        
        for facility_info in centroid_distances:
            if remaining_demand <= 0:
                break
            
            facility_idx = facility_info['facility_idx']
            available_supply = facility_remaining.get(facility_idx, 0)
            
            if available_supply > 0:
                # Allocate supply (min of remaining demand and available supply)
                allocated_supply = min(remaining_demand, available_supply)
                
                # Update tracking
                centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] += allocated_supply
                facility_remaining[facility_idx] -= allocated_supply
                remaining_demand -= allocated_supply
                
                # Track energy types and facilities
                energy_type = facility_info['type']
                if energy_type and energy_type not in energy_types_used:
                    energy_types_used.append(energy_type)
                
                facility_id = facility_info['gem_id']
                if facility_id and facility_id not in facilities_used:
                    facilities_used.append(facility_id)
                
                # Create connection line for visualization following network path
                connection_info = {
                    'centroid_idx': centroid_idx,
                    'centroid_lat': centroid.geometry.y,
                    'centroid_lon': centroid.geometry.x,
                    'facility_idx': facility_idx,
                    'facility_lat': facility_info['facility_lat'],
                    'facility_lon': facility_info['facility_lon'],
                    'facility_type': facility_info['type'],
                    'facility_gem_id': facility_info['gem_id'],
                    'distance': facility_info['distance'],
                    'supply_allocated_mwh': allocated_supply
                }
                
                # Add network path if available
                if network_graph is not None:
                    connection_info['network_path'] = facility_info.get('network_path', None)
                
                connection_lines.append(connection_info)
        
        # Update centroid status
        centroids_gdf.loc[centroid_idx, 'energy_types_supplied'] = ', '.join(energy_types_used)
        centroids_gdf.loc[centroid_idx, 'facilities_serving'] = ', '.join(facilities_used)
        centroids_gdf.loc[centroid_idx, 'supply_shortage_mwh'] = remaining_demand
        
        if remaining_demand <= 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Filled'
        elif centroids_gdf.loc[centroid_idx, 'supply_received_mwh'] > 0:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Partially Filled'
        else:
            centroids_gdf.loc[centroid_idx, 'supply_status'] = 'Not Filled'
    
    # Print allocation summary
    status_counts = centroids_gdf['supply_status'].value_counts()
    print(f"\nSupply Allocation Summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} centroids")
    
    total_supplied = centroids_gdf['supply_received_mwh'].sum()
    total_shortage = centroids_gdf['supply_shortage_mwh'].sum()
    print(f"Total supply allocated: {total_supplied:,.0f} MWh")
    print(f"Total unmet demand: {total_shortage:,.0f} MWh")
    print(f"Supply fulfillment rate: {(total_supplied / total_demand * 100):.1f}%")
    
    return centroids_gdf, connection_lines

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
    """Connect a chunk of points to nearest grid nodes with enhanced connectivity"""
    connections = []
    for point in point_chunk:
        point_coord = (point.x, point.y)
        
        # Find nearest node in the grid network
        if grid_nodes:
            nearest_node = min(grid_nodes, 
                             key=lambda n: Point(n).distance(point))
            
            # Add edge if within distance threshold (50km for both centroids and facilities)
            distance = Point(nearest_node).distance(point)
            if point_type == 'pop_centroid' and distance > 50000:  # 50km for centroids
                continue
            if point_type == 'facility' and distance > 50000:  # 50km for facilities  
                continue
            connections.append((point_coord, nearest_node, distance))
    return connections

def create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf, n_threads=1):
    """Create network graph from facilities, grid lines, and population centroids"""
    # Project all data to appropriate UTM zone for accurate distance calculations
    if not facilities_gdf.empty:
        # Get approximate center to determine appropriate UTM zone
        center = facilities_gdf.geometry.unary_union.centroid
        center_lon = center.x
        center_lat = center.y
    else:
        # Fallback to grid center
        center = grid_lines_gdf.geometry.unary_union.centroid
        center_lon = center.x
        center_lat = center.y
    
    # Calculate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
    
    print(f"Projecting network data to {utm_crs} for accurate distance calculations...")
    
    # Project all datasets to UTM for accurate distance calculation
    facilities_utm = facilities_gdf.to_crs(utm_crs)
    grid_lines_utm = grid_lines_gdf.to_crs(utm_crs)
    centroids_utm = centroids_gdf.to_crs(utm_crs)
    
    # Initialize empty graph
    G = nx.Graph()
    
    # 1. Process grid lines and split at intersections
    single_lines = []
    for _, row in grid_lines_utm.iterrows():
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
    facility_nodes = set((point.x, point.y) for point in facilities_utm.geometry)
    pop_centroid_nodes = set((point.x, point.y) for point in centroids_utm.geometry)
    
    # 4. Add all nodes to the graph with their types, but store original lat/lon for reference
    for node in nodes:
        G.add_node(node, pos=node, type='grid_line')
    for i, (node, point) in enumerate(zip(facility_nodes, facilities_gdf.geometry)):
        G.add_node(node, pos=node, type='facility', 
                  original_lat=point.y, original_lon=point.x, facility_idx=facilities_gdf.index[i])
    for i, (node, point) in enumerate(zip(pop_centroid_nodes, centroids_gdf.geometry)):
        G.add_node(node, pos=node, type='pop_centroid', 
                  original_lat=point.y, original_lon=point.x, centroid_idx=centroids_gdf.index[i])
    
    # 5. Create edges from split lines (now in UTM, so line.length is in meters)
    for line in split_lines:
        coords = list(line.coords)
        G.add_edge(coords[0], coords[-1], weight=line.length)
    
    # 6. Connect facilities and centroids to nearest grid nodes
    grid_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'grid_line']
    
    if n_threads > 1 and (len(facilities_utm) + len(centroids_utm)) > 100:
        print(f"Using {n_threads} threads for network connections...")
        
        # Process facilities and centroids in parallel
        for point_gdf_utm, point_gdf_orig, point_type in [(facilities_utm, facilities_gdf, 'facility'), 
                                                          (centroids_utm, centroids_gdf, 'pop_centroid')]:
            if len(point_gdf_utm) > 0:
                # Split points into chunks
                points = list(point_gdf_utm.geometry)
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
        for point_gdf_utm, point_type in [(facilities_utm, 'facility'), (centroids_utm, 'pop_centroid')]:
            for point in point_gdf_utm.geometry:
                point_coord = (point.x, point.y)
                
                # Find nearest node in the grid network
                if grid_nodes:
                    nearest_node = min(grid_nodes, 
                                     key=lambda n: Point(n).distance(point))
                    
                    # Add edge if within distance threshold (50km for both centroids and facilities)
                    distance = Point(nearest_node).distance(point)
                    max_distance = 50000  # 50km in meters for both centroids and facilities
                    if distance > max_distance:
                        continue
                    G.add_edge(point_coord, nearest_node, weight=distance)
    
    # Before finalizing, connect disconnected components to improve network connectivity
    print("Checking for disconnected network components...")
    G = connect_disconnected_components(G)
    
    print(f"Network graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def connect_disconnected_components(G, max_gap_distance=50000):
    """
    STRATEGY 1 (ONLY): Connect disconnected network components by adding bridges between them.
    This is the primary and only strategy for network connectivity enhancement.
    
    Args:
        G: NetworkX graph
        max_gap_distance: Maximum distance in meters to bridge components (50km)
    
    Returns:
        Enhanced NetworkX graph with better connectivity
    """
    # Find all connected components
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        print("Network is fully connected - no disconnected components found")
        return G
    
    print(f"Found {len(components)} disconnected network components")
    
    # Calculate component sizes for prioritization
    component_info = []
    for i, component in enumerate(components):
        # Count different node types in each component
        grid_nodes = sum(1 for n in component if G.nodes[n].get('type') == 'grid_line')
        facility_nodes = sum(1 for n in component if G.nodes[n].get('type') == 'facility')
        centroid_nodes = sum(1 for n in component if G.nodes[n].get('type') == 'pop_centroid')
        
        component_info.append({
            'id': i,
            'nodes': component,
            'size': len(component),
            'grid_nodes': grid_nodes,
            'facility_nodes': facility_nodes,
            'centroid_nodes': centroid_nodes,
            'priority_score': facility_nodes * 10 + centroid_nodes * 5 + grid_nodes  # Prioritize components with facilities
        })
    
    # Sort components by priority (largest and most important first)
    component_info.sort(key=lambda x: x['priority_score'], reverse=True)
    
    print("Component analysis:")
    for comp in component_info:
        print(f"  Component {comp['id']}: {comp['size']} nodes "
              f"(Grid: {comp['grid_nodes']}, Facilities: {comp['facility_nodes']}, "
              f"Centroids: {comp['centroid_nodes']}, Priority: {comp['priority_score']})")
    
    # Connect smaller components to larger ones using single distance threshold
    bridges_added = 0
    
    print(f"\nAttempting connections with {max_gap_distance/1000:.1f}km threshold...")
    
    for i in range(1, len(component_info)):
        current_comp = component_info[i]
        
        # Find the best component to connect to
        best_target = None
        best_distance = float('inf')
        best_bridge = None
        
        # Check connection to all higher-priority components
        for j in range(i):
            target_comp = component_info[j]
            
            # Find shortest distance between components
            min_distance = float('inf')
            best_nodes = None
            
            for node1 in current_comp['nodes']:
                if G.nodes[node1].get('type') == 'grid_line':  # Only connect via grid nodes
                    for node2 in target_comp['nodes']:
                        if G.nodes[node2].get('type') == 'grid_line':  # Only connect via grid nodes
                            distance = Point(node1).distance(Point(node2))
                            if distance < min_distance:
                                min_distance = distance
                                best_nodes = (node1, node2)
            
            # Check if this is the best connection so far
            if min_distance < best_distance and min_distance <= max_gap_distance:
                best_distance = min_distance
                best_target = j
                best_bridge = best_nodes
        
        # Add bridge if we found a good connection
        if best_bridge and best_distance <= max_gap_distance:
            node1, node2 = best_bridge
            G.add_edge(node1, node2, weight=best_distance, 
                      bridge=True, bridge_type='component_connector')
            bridges_added += 1
            
            print(f"  Added bridge between components {current_comp['id']} and {best_target}: "
                  f"{best_distance/1000:.1f} km")
    
    print(f"\nAdded {bridges_added} component bridges to improve connectivity")
    
    # Final connectivity verification
    final_components = list(nx.connected_components(G))
    print(f"Final network has {len(final_components)} connected component(s)")
    
    if len(final_components) > 1:
        print("Warning: Some components remain disconnected. Consider increasing max_gap_distance.")
        for i, comp in enumerate(final_components):
            comp_size = len(comp)
            facility_count = sum(1 for n in comp if G.nodes[n].get('type') == 'facility')
            centroid_count = sum(1 for n in comp if G.nodes[n].get('type') == 'pop_centroid')
            print(f"  Isolated component {i}: {comp_size} nodes ({facility_count} facilities, {centroid_count} centroids)")
    
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
    admin_boundaries['geometry'] = admin_boundaries['geometry'].simplify(tolerance=0.001, preserve_topology=True)
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
            
            # Calculate facility distances and allocate supply
            if len(facilities_gdf) > 0:
                print(f"Calculating facility distances for supply allocation with {len(centroids_filtered)} centroids using {n_threads} threads...")
                
                # Calculate distances from all centroids to all facilities
                centroid_facility_distances = []
                
                if n_threads > 1 and len(centroids_filtered) > 100:
                    # Parallel processing for large datasets
                    chunk_size = max(10, len(centroids_filtered) // n_threads)
                    chunks = [centroids_filtered.iloc[i:i+chunk_size] for i in range(0, len(centroids_filtered), chunk_size)]
                    
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        chunk_results = list(executor.map(
                            partial(calculate_facility_distances_chunk, facilities_gdf=facilities_gdf, network_graph=network_graph),
                            chunks
                        ))
                    
                    # Combine results
                    for chunk_result in chunk_results:
                        centroid_facility_distances.extend(chunk_result)
                else:
                    # Serial processing for small datasets or single thread
                    # Create mapping from original coordinates to network coordinates
                    centroid_to_network_mapping = {}
                    facility_to_network_mapping = {}
                    
                    for node, data in network_graph.nodes(data=True):
                        if data.get('type') == 'pop_centroid':
                            original_lat = data.get('original_lat')
                            original_lon = data.get('original_lon')
                            if original_lat is not None and original_lon is not None:
                                centroid_to_network_mapping[(original_lon, original_lat)] = node
                        elif data.get('type') == 'facility':
                            original_lat = data.get('original_lat')
                            original_lon = data.get('original_lon')
                            if original_lat is not None and original_lon is not None:
                                facility_to_network_mapping[(original_lon, original_lat)] = node
                    
                    for idx, centroid in centroids_filtered.iterrows():
                        centroid_distances = []
                        centroid_coord = (centroid.geometry.x, centroid.geometry.y)
                        network_centroid = centroid_to_network_mapping.get(centroid_coord)
                        
                        # Check if centroid is in the network graph
                        if network_centroid is not None:
                            # Calculate distance to ALL facilities in the network
                            for fac_idx, facility in facilities_gdf.iterrows():
                                facility_coord = (facility.geometry.x, facility.geometry.y)
                                network_facility = facility_to_network_mapping.get(facility_coord)
                                
                                if network_facility is not None:
                                    try:
                                        # Calculate shortest path distance using NetworkX (in meters)
                                        network_distance_meters = nx.shortest_path_length(
                                            network_graph, 
                                            network_centroid, 
                                            network_facility, 
                                            weight='weight'
                                        )
                                        
                                        # Convert meters to kilometers
                                        network_distance_km = network_distance_meters / 1000.0
                                        
                                        centroid_distances.append({
                                            'centroid_idx': idx,
                                            'facility_idx': fac_idx,
                                            'distance': network_distance_km,  # Now in kilometers
                                            'type': facility.get('Grouped_Type', ''),
                                            'capacity': facility.get('Adjusted_Capacity_MW', np.nan),
                                            'gem_id': facility.get('GEM unit/phase ID', ''),
                                            'facility_lat': facility.get('Latitude', np.nan),
                                            'facility_lon': facility.get('Longitude', np.nan)
                                        })
                                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                                        # No path exists between centroid and this facility
                                        continue
                            
                            # Sort by distance for allocation priority
                            centroid_distances.sort(key=lambda x: x['distance'])
                        
                        centroid_facility_distances.append({
                            'centroid_idx': idx,
                            'facility_distances': centroid_distances
                        })
                
                # Allocate supply to centroids
                centroids_filtered, connection_lines = allocate_supply_to_centroids(
                    centroids_filtered, facilities_gdf, centroid_facility_distances, country_iso3, network_graph, year=2024
                )
                
                print(f"Added supply allocation information to centroids")
            
        except Exception as e:
            print(f"Warning: Network analysis failed: {e}")
            network_results = {}
            
            # Fallback: Calculate facility distances without network (Euclidean distance)
            if not facilities_gdf.empty:
                print("Falling back to Euclidean distance calculation for supply allocation...")
                
                # Calculate distances from all centroids to all facilities using Euclidean distance
                centroid_facility_distances = []
                
                if n_threads > 1 and len(centroids_filtered) > 100:
                    # Parallel processing for large datasets
                    chunk_size = max(10, len(centroids_filtered) // n_threads)
                    chunks = [centroids_filtered.iloc[i:i+chunk_size] for i in range(0, len(centroids_filtered), chunk_size)]
                    
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        chunk_results = list(executor.map(
                            partial(calculate_facility_distances_chunk, facilities_gdf=facilities_gdf, network_graph=None),
                            chunks
                        ))
                    
                    # Combine results
                    for chunk_result in chunk_results:
                        centroid_facility_distances.extend(chunk_result)
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
                        # Project both datasets to UTM for accurate distance calculation
                        centroids_utm = centroids_filtered.to_crs(utm_crs)
                        facilities_utm = facilities_gdf.to_crs(utm_crs)
                        
                        for idx, centroid in centroids_filtered.iterrows():
                            centroid_distances = []
                            centroid_utm = centroids_utm.loc[idx]
                            
                            # Calculate distances to all facilities
                            for fac_idx, facility in facilities_gdf.iterrows():
                                facility_utm = facilities_utm.loc[fac_idx]
                                distance_meters = facility_utm.geometry.distance(centroid_utm.geometry)
                                distance_km = distance_meters / 1000.0  # Convert to kilometers
                                
                                centroid_distances.append({
                                    'centroid_idx': idx,
                                    'facility_idx': fac_idx,
                                    'distance': distance_km,  # Now in kilometers
                                    'type': facility.get('Grouped_Type', ''),
                                    'capacity': facility.get('Adjusted_Capacity_MW', np.nan),
                                    'gem_id': facility.get('GEM unit/phase ID', ''),
                                    'facility_lat': facility.get('Latitude', np.nan),
                                    'facility_lon': facility.get('Longitude', np.nan)
                                })
                            
                            # Sort by distance for allocation priority
                            centroid_distances.sort(key=lambda x: x['distance'])
                            
                            centroid_facility_distances.append({
                                'centroid_idx': idx,
                                'facility_distances': centroid_distances
                            })
                
                # Allocate supply to centroids
                centroids_filtered, connection_lines = allocate_supply_to_centroids(
                    centroids_filtered, facilities_gdf, centroid_facility_distances, country_iso3, None, year=2024
                )
                
                print(f"Added Euclidean distance supply allocation information to centroids")
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
    
    # Keep the specified columns for simplified output - now including supply allocation results
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
        'nearest_facility_distance',    # Distance to nearest energy facility (km)
        'nearest_facility_type',        # Type of nearest facility (e.g., Solar, Wind, Fossil)
        'nearest_facility_capacity',    # Capacity of nearest facility in MW
        'nearest_facility_gem_id',      # GEM unit/phase ID of nearest facility
        'nearest_facility_lat',         # Latitude of nearest facility
        'nearest_facility_lon',         # Longitude of nearest facility
        'supply_status',                # Filled, Partially Filled, Not Filled
        'energy_types_supplied',        # Energy types serving this centroid (e.g., "Solar, Wind")
        'facilities_serving',           # Facility IDs serving this centroid
        'supply_received_mwh',          # Total supply received (MWh)
        'supply_shortage_mwh'           # Unmet demand (MWh)
        ]
    
    # Only keep columns that exist in the dataframe
    available_columns = [col for col in final_columns if col in centroids_filtered.columns]
    centroids_simplified = centroids_filtered[available_columns].copy()
    
    print(f"Final output columns: {available_columns}")
    
    # Save results in streamlined format
    output_file_gpkg = output_path / f"supply_analysis_{country_iso3}.gpkg"
    
    # Save as GPKG (primary output for ArcGIS with multiple layers: centroids, grid_lines, facilities)
    centroids_simplified.to_file(output_file_gpkg, driver="GPKG", layer="centroids")
    print(f"GPKG results saved to {output_file_gpkg}")
    
    # Save CSV for easy data analysis (without geometry)
    csv_file = output_path / f"supply_analysis_{country_iso3}.csv"
    csv_columns = [col for col in available_columns if col != 'geometry']
    centroids_simplified[csv_columns].to_csv(csv_file, index=False)
    print(f"CSV saved to {csv_file}")
    
    # Create and save enhanced grid lines layer and facilities layer
    if 'connection_lines' in locals() and connection_lines:
        print(f"Creating enhanced grid lines layer (original + segmented network connections)...")
        
        # 1. Load original grid lines for the country
        original_grid_lines = load_grid_lines(country_bbox, admin_boundaries)
        
        # 2. Create connection lines GeoDataFrame with segmented network paths
        # Break network paths into segments with proper attribution for detailed analysis
        connection_line_geometries = []
        connection_line_attributes = []
        
        # Check if network_graph is available from network analysis
        network_graph_available = 'network_graph' in locals() and network_graph is not None
        
        for connection in connection_lines:
            # Check if we have network path information
            if 'network_path' in connection and connection['network_path'] is not None:
                # Break network path into segments with attribution
                path_coords = connection['network_path']
                if len(path_coords) >= 2:
                    # Convert UTM coordinates back to WGS84 if needed
                    if network_graph_available:
                        # Check if coordinates look like UTM (large numbers)
                        center_x = sum(coord[0] for coord in path_coords) / len(path_coords)
                        center_y = sum(coord[1] for coord in path_coords) / len(path_coords)
                        
                        if abs(center_x) > 180 or abs(center_y) > 90:
                            # Convert UTM coordinates back to WGS84
                            center_lon = connection['centroid_lon']
                            center_lat = connection['centroid_lat']
                            utm_zone = int((center_lon + 180) / 6) + 1
                            utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
                            
                            # Convert all path coordinates to WGS84
                            path_utm_gdf = gpd.GeoSeries([Point(coord) for coord in path_coords], crs=utm_crs)
                            path_wgs84_gdf = path_utm_gdf.to_crs("EPSG:4326")
                            path_coords_wgs84 = [(point.x, point.y) for point in path_wgs84_gdf]
                        else:
                            # Already in WGS84
                            path_coords_wgs84 = path_coords
                    else:
                        path_coords_wgs84 = path_coords
                    
                    # Create segments from the network path
                    for i in range(len(path_coords_wgs84) - 1):
                        segment_geom = LineString([path_coords_wgs84[i], path_coords_wgs84[i + 1]])
                        
                        # Determine line type based on position and connection type
                        connection_type = connection.get('connection_type', 'network_path')
                        
                        if connection_type == 'network_via_bridge':
                            if i == 0:
                                line_type = 'centroid_to_grid'
                            elif i == len(path_coords_wgs84) - 2:
                                line_type = 'grid_to_facility'
                            else:
                                line_type = 'grid_via_bridge'
                        elif connection_type == 'direct_fallback':
                            line_type = 'direct_fallback'
                        else:
                            # Standard network path
                            if i == 0:
                                line_type = 'centroid_to_grid'
                            elif i == len(path_coords_wgs84) - 2:
                                line_type = 'grid_to_facility'
                            else:
                                line_type = 'grid_infrastructure'
                        
                        connection_line_geometries.append(segment_geom)
                        connection_line_attributes.append({
                            'line_type': line_type,
                            'centroid_idx': connection['centroid_idx'],
                            'facility_idx': connection['facility_idx'],
                            'facility_type': connection['facility_type'],
                            'facility_gem_id': connection['facility_gem_id'],
                            'distance_km': connection['distance'],
                            'supply_allocated_mwh': connection['supply_allocated_mwh'],
                            'path_type': 'network_path',
                            'segment': i + 1
                        })
                else:
                    # Fallback to direct line if path is too short
                    line_geom = LineString([
                        (connection['centroid_lon'], connection['centroid_lat']),
                        (connection['facility_lon'], connection['facility_lat'])
                    ])
                    connection_line_geometries.append(line_geom)
                    connection_line_attributes.append({
                        'line_type': 'direct_connection',
                        'centroid_idx': connection['centroid_idx'],
                        'facility_idx': connection['facility_idx'],
                        'facility_type': connection['facility_type'],
                        'facility_gem_id': connection['facility_gem_id'],
                        'distance_km': connection['distance'],
                        'supply_allocated_mwh': connection['supply_allocated_mwh'],
                        'path_type': 'direct_path',
                        'segment': 1
                    })
            else:
                # Fallback: Create direct line from centroid to facility
                line_geom = LineString([
                    (connection['centroid_lon'], connection['centroid_lat']),
                    (connection['facility_lon'], connection['facility_lat'])
                ])
                connection_line_geometries.append(line_geom)
                connection_line_attributes.append({
                    'line_type': 'direct_connection',
                    'centroid_idx': connection['centroid_idx'],
                    'facility_idx': connection['facility_idx'],
                    'facility_type': connection['facility_type'],
                    'facility_gem_id': connection['facility_gem_id'],
                    'distance_km': connection['distance'],
                    'supply_allocated_mwh': connection['supply_allocated_mwh'],
                    'path_type': 'direct_path',
                    'segment': 1
                })
        
        connections_gdf = gpd.GeoDataFrame(
            connection_line_attributes, 
            geometry=connection_line_geometries, 
            crs="EPSG:4326"
        )
        
        # 3. Prepare original grid lines with consistent attributes
        if not original_grid_lines.empty:
            # Add line_type and fill missing columns with appropriate values
            original_grid_lines = original_grid_lines.copy()
            original_grid_lines['line_type'] = 'grid_infrastructure'
            original_grid_lines['centroid_idx'] = None
            original_grid_lines['facility_idx'] = None
            original_grid_lines['facility_type'] = None
            original_grid_lines['facility_gem_id'] = None
            original_grid_lines['distance_km'] = None
            original_grid_lines['supply_allocated_mwh'] = None
            original_grid_lines['path_type'] = 'existing_infrastructure'
            original_grid_lines['segment'] = None
            
            # Keep only the columns we need for consistency
            grid_columns = ['geometry', 'line_type', 'centroid_idx', 'facility_idx', 'facility_type', 
                           'facility_gem_id', 'distance_km', 'supply_allocated_mwh', 'path_type', 'segment']
            original_grid_lines = original_grid_lines[grid_columns]
            
            # 4. Combine original grid lines with segmented network connection lines
            enhanced_grid_gdf = pd.concat([original_grid_lines, connections_gdf], ignore_index=True)
            print(f"Enhanced grid layer: {len(original_grid_lines)} original infrastructure lines + {len(connections_gdf)} network connection segments = {len(enhanced_grid_gdf)} total")
        else:
            # If no original grid lines, use only network connections
            enhanced_grid_gdf = connections_gdf
            print(f"Enhanced grid layer: {len(connections_gdf)} network connection segments only (no original grid infrastructure found)")
        
        # Save enhanced grid lines to GPKG
        enhanced_grid_gdf.to_file(output_file_gpkg, driver="GPKG", layer="grid_lines")
        print(f"Enhanced grid lines added to GPKG as 'grid_lines' layer")
    else:
        print(f"No connection lines available for enhanced grid layer")
    
    # Create and save facilities layer with capacity and usage information
    if not facilities_gdf.empty:
        print(f"Creating facilities layer with capacity and usage data...")
        
        # Create facilities layer with enhanced attributes
        facilities_enhanced = facilities_gdf.copy()
        
        # Add capacity and usage information
        facilities_enhanced['capacity_mw'] = facilities_enhanced.get('Adjusted_Capacity_MW', 0)
        
        # Calculate total generation potential using conversion rates
        conversion_rates = load_conversion_rates(country_iso3)
        facilities_enhanced['total_generation_mwh'] = 0.0
        facilities_enhanced['used_mwh'] = 0.0
        facilities_enhanced['remaining_mwh'] = 0.0
        
        for idx, facility in facilities_enhanced.iterrows():
            capacity_mw = facility.get('Adjusted_Capacity_MW', 0)
            energy_type = facility.get('Grouped_Type', '')
            
            if pd.isna(capacity_mw) or capacity_mw <= 0:
                total_generation_mwh = 0
            else:
                conv_rate = conversion_rates.get(energy_type, 0.30)
                total_generation_mwh = capacity_mw * 8760 * conv_rate
            
            facilities_enhanced.loc[idx, 'total_generation_mwh'] = total_generation_mwh
            
            # Calculate used capacity from connection lines
            if 'connection_lines' in locals() and connection_lines:
                used_mwh = sum([conn['supply_allocated_mwh'] for conn in connection_lines 
                               if conn['facility_idx'] == idx])
                facilities_enhanced.loc[idx, 'used_mwh'] = used_mwh
                facilities_enhanced.loc[idx, 'remaining_mwh'] = max(0, total_generation_mwh - used_mwh)
            else:
                facilities_enhanced.loc[idx, 'used_mwh'] = 0
                facilities_enhanced.loc[idx, 'remaining_mwh'] = total_generation_mwh
        
        # Add utilization percentage
        facilities_enhanced['utilization_percent'] = np.where(
            facilities_enhanced['total_generation_mwh'] > 0,
            (facilities_enhanced['used_mwh'] / facilities_enhanced['total_generation_mwh']) * 100,
            0
        )
        
        # Select relevant columns for the facilities layer
        facilities_columns = [
            'geometry',
            'GID_0', 'NAME_0',  # Country info if available
            'GEM unit/phase ID',
            'Grouped_Type',
            'Latitude', 'Longitude',
            'capacity_mw',
            'total_generation_mwh',
            'used_mwh',
            'remaining_mwh',
            'utilization_percent'
        ]
        
        # Only keep columns that exist
        available_facilities_columns = [col for col in facilities_columns if col in facilities_enhanced.columns]
        facilities_simplified = facilities_enhanced[available_facilities_columns].copy()
        
        # Add country information if not already present
        if 'GID_0' not in facilities_simplified.columns:
            facilities_simplified['GID_0'] = country_iso3
        if 'NAME_0' not in facilities_simplified.columns:
            facilities_simplified['NAME_0'] = country_name
        
        # Save facilities to GPKG
        facilities_simplified.to_file(output_file_gpkg, driver="GPKG", layer="facilities")
        print(f"Facilities layer added to GPKG as 'facilities' layer")
        print(f"Facilities summary:")
        print(f"  Total facilities: {len(facilities_simplified)}")
        print(f"  Total capacity: {facilities_simplified['capacity_mw'].sum():,.0f} MW")
        print(f"  Total generation potential: {facilities_simplified['total_generation_mwh'].sum():,.0f} MWh")
        print(f"  Total used: {facilities_simplified['used_mwh'].sum():,.0f} MWh")
        print(f"  Total remaining: {facilities_simplified['remaining_mwh'].sum():,.0f} MWh")
        print(f"  Average utilization: {facilities_simplified['utilization_percent'].mean():.1f}%")
        
        # Print capacity by energy type
        if 'Grouped_Type' in facilities_simplified.columns:
            capacity_by_type = facilities_simplified.groupby('Grouped_Type').agg({
                'capacity_mw': 'sum',
                'total_generation_mwh': 'sum',
                'used_mwh': 'sum',
                'utilization_percent': 'mean'
            }).round(1)
            print(f"\nCapacity by energy type:")
            for energy_type, data in capacity_by_type.iterrows():
                print(f"  {energy_type}: {data['capacity_mw']:,.0f} MW, "
                      f"{data['total_generation_mwh']:,.0f} MWh potential, "
                      f"{data['used_mwh']:,.0f} MWh used, "
                      f"{data['utilization_percent']:.1f}% utilization")
    else:
        print(f"No facilities available for facilities layer")
    
    # Update layer count in summary
    layer_count = 1  # centroids layer
    if 'enhanced_grid_gdf' in locals():
        layer_count += 1  # grid_lines layer
    if not facilities_gdf.empty:
        layer_count += 1  # facilities layer
    
    print(f"GPKG now contains {layer_count} layers: 'centroids'", end="")
    if 'enhanced_grid_gdf' in locals():
        print(f", 'grid_lines'", end="")
    if not facilities_gdf.empty:
        print(f", 'facilities'", end="")
    print()
    
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
    
    # Network analysis and supply allocation summary
    if 'nearest_facility_distance' in centroids_simplified.columns:
        facility_data = centroids_simplified.dropna(subset=['nearest_facility_distance'])
        if len(facility_data) > 0:
            print(f"\nSupply Allocation Analysis Results:")
            print(f"Centroids with facility access: {len(facility_data):,} out of {len(centroids_simplified):,}")
            
            # Supply allocation status summary
            if 'supply_status' in centroids_simplified.columns:
                status_counts = centroids_simplified['supply_status'].value_counts()
                print(f"\nSupply Status Distribution:")
                for status, count in status_counts.items():
                    percentage = (count / len(centroids_simplified)) * 100
                    print(f"  {status}: {count:,} centroids ({percentage:.1f}%)")
            
            # Energy supply summary
            if 'supply_received_mwh' in centroids_simplified.columns:
                total_supply = centroids_simplified['supply_received_mwh'].sum()
                total_shortage = centroids_simplified['supply_shortage_mwh'].sum() if 'supply_shortage_mwh' in centroids_simplified.columns else 0
                total_demand = total_supply + total_shortage
                
                print(f"\nEnergy Supply Summary:")
                print(f"Total demand: {total_demand:,.0f} MWh")
                print(f"Total supply allocated: {total_supply:,.0f} MWh")
                print(f"Total shortage: {total_shortage:,.0f} MWh")
                if total_demand > 0:
                    fulfillment_rate = (total_supply / total_demand) * 100
                    print(f"Supply fulfillment rate: {fulfillment_rate:.1f}%")
            
            # Check if network-based calculation was used
            network_path_count = len(centroids_simplified[
                (centroids_simplified['nearest_facility_type'] == 'NO_NETWORK_PATH') |
                (centroids_simplified['nearest_facility_type'] == 'NOT_IN_NETWORK')
            ])
            
            if network_path_count > 0:
                print(f"\nNetwork Distance Calculation Status:")
                print(f"  Centroids with network path: {len(facility_data) - network_path_count:,}")
                print(f"  Centroids without network path: {network_path_count:,}")
                
                # Calculate statistics for network-connected centroids only
                network_connected = facility_data[
                    ~facility_data['nearest_facility_type'].isin(['NO_NETWORK_PATH', 'NOT_IN_NETWORK'])
                ]
                if len(network_connected) > 0:
                    print(f"Average network distance to nearest facility: {network_connected['nearest_facility_distance'].mean():.2f} km")
                    print(f"Median network distance to nearest facility: {network_connected['nearest_facility_distance'].median():.2f} km")
                else:
                    print("No centroids connected via network paths")
            else:
                # All distances calculated successfully (either network or Euclidean)
                print(f"\nDistance Statistics:")
                print(f"Average distance to nearest facility: {facility_data['nearest_facility_distance'].mean():.2f} km")
                print(f"Median distance to nearest facility: {facility_data['nearest_facility_distance'].median():.2f} km")
            
            # Energy type breakdown
            if 'energy_types_supplied' in centroids_simplified.columns:
                supplied_centroids = centroids_simplified[centroids_simplified['energy_types_supplied'] != '']
                if len(supplied_centroids) > 0:
                    print(f"\nEnergy Types Serving Centroids:")
                    # Count unique energy types
                    all_energy_types = []
                    for types_str in supplied_centroids['energy_types_supplied']:
                        if types_str and isinstance(types_str, str):
                            types_list = [t.strip() for t in types_str.split(',')]
                            all_energy_types.extend(types_list)
                    
                    from collections import Counter
                    energy_type_counts = Counter(all_energy_types)
                    for energy_type, count in energy_type_counts.most_common():
                        print(f"  {energy_type}: {count} centroids")
            
            # Facility type breakdown (nearest facility)
            if 'nearest_facility_type' in centroids_simplified.columns:
                facility_types = facility_data['nearest_facility_type'].value_counts()
                print(f"\nNearest Facility Types:")
                for ftype, count in facility_types.items():
                    if ftype and ftype not in ['NO_NETWORK_PATH', 'NOT_IN_NETWORK']:  # Skip empty strings and network status indicators
                        print(f"  {ftype}: {count} centroids")
                
                # Report network connectivity issues if any
                network_issues = facility_types.get('NO_NETWORK_PATH', 0) + facility_types.get('NOT_IN_NETWORK', 0)
                if network_issues > 0:
                    print(f"\nNetwork connectivity issues: {network_issues} centroids")
                    if facility_types.get('NO_NETWORK_PATH', 0) > 0:
                        print(f"  No network path to facilities: {facility_types['NO_NETWORK_PATH']} centroids")
                    if facility_types.get('NOT_IN_NETWORK', 0) > 0:
                        print(f"  Not connected to network: {facility_types['NOT_IN_NETWORK']} centroids")
    
    if network_results:
        print(f"\nNetwork Statistics:")
        print(f"Total network nodes: {network_results.get('total_nodes', 'N/A')}")
        print(f"Total network edges: {network_results.get('total_edges', 'N/A')}")
        print(f"Energy facilities in network: {network_results.get('facility_nodes', 'N/A')}")
    
    return str(output_file_gpkg)  # Return GPKG file path as primary output

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
