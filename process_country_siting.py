#!/usr/bin/env python3
"""
Siting Analysis for Remote Settlement Electrification

This script analyzes remote settlements that are not adequately served by existing grid 
infrastructure and determines optimal locations for new grid extensions or off-grid solutions.

Key Steps:
1. Filter Settlements: Identifies settlements not fully served by removing those with population < 100
2. Cluster Analysis: Groups underserved settlements using K-means (number based on facilities with remaining capacity)
3. Grid Proximity: Calculates distances from cluster centers to nearest grid lines
4. Network Design: For remote clusters, designs minimum spanning tree style networks
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse
import sys
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
COMMON_CRS = "EPSG:4326"
ANALYSIS_YEAR = 2030
SUPPLY_FACTOR = 1.0
CLUSTER_RADIUS_KM = 50
CLUSTER_MIN_SAMPLES = 1
GRID_DISTANCE_THRESHOLD_KM = 50
DROP_PERCENTAGE = 0.01


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_centroids(country_iso3, scenario_suffix, output_dir="outputs_per_country"):
    """Load centroids data from parquet file."""
    parquet_path = Path(output_dir) / "parquet" / scenario_suffix / f"centroids_{country_iso3}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Centroids file not found: {parquet_path}")
    
    centroids_gdf = gpd.read_parquet(parquet_path)
    print(f"Loaded {len(centroids_gdf)} centroids from {parquet_path}")
    return centroids_gdf


def load_grid_lines(country_iso3, scenario_suffix, output_dir="outputs_per_country"):
    """Load grid lines data from parquet file."""
    parquet_path = Path(output_dir) / "parquet" / scenario_suffix / f"grid_lines_{country_iso3}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Grid lines file not found: {parquet_path}")
    
    grid_lines_gdf = gpd.read_parquet(parquet_path)
    print(f"Loaded {len(grid_lines_gdf)} grid lines from {parquet_path}")
    return grid_lines_gdf


def load_facilities(country_iso3, scenario_suffix, output_dir="outputs_per_country"):
    """Load facilities data from parquet file."""
    parquet_path = Path(output_dir) / "parquet" / scenario_suffix / f"facilities_{country_iso3}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Facilities file not found: {parquet_path}")
    
    facilities_gdf = gpd.read_parquet(parquet_path)
    print(f"Loaded {len(facilities_gdf)} facilities from {parquet_path}")
    return facilities_gdf


def calculate_num_clusters(facilities_gdf, settlements_gdf):
    """Calculate number of clusters based on remaining capacity by energy type (capacity-driven approach)."""
    print("\n" + "="*60)
    print("CALCULATING CLUSTERS FROM REMAINING CAPACITY (CAPACITY-DRIVEN)")
    print("="*60)
    
    # Filter facilities with remaining capacity
    facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy()
    
    print(f"\nFacilities with remaining capacity: {len(facilities_with_capacity)}")
    
    if len(facilities_with_capacity) == 0:
        print("No facilities with remaining capacity found!")
        return 1, {}, 0, 0
    
    # Calculate remaining supply by energy type - THIS IS THE PRIMARY DRIVER
    remaining_by_type = facilities_with_capacity.groupby('Grouped_Type')['remaining_mwh'].sum().to_dict()
    total_remaining_supply = sum(remaining_by_type.values())
    
    # Calculate demand gap for settlements
    demand_col = f'Total_Demand_{ANALYSIS_YEAR}_centroid'
    if demand_col in settlements_gdf.columns:
        settlements_gdf['demand_gap_mwh'] = settlements_gdf[demand_col] - settlements_gdf['supply_received_mwh']
    else:
        settlements_gdf['demand_gap_mwh'] = settlements_gdf['supply_received_mwh']
    
    settlements_gdf['demand_gap_mwh'] = settlements_gdf['demand_gap_mwh'].clip(lower=0)
    total_demand_gap = settlements_gdf['demand_gap_mwh'].sum()
    
    print(f"\nRemaining capacity by energy type (PRIMARY DRIVER):")
    for energy_type, remaining_mwh in sorted(remaining_by_type.items()):
        print(f"  {energy_type}: {remaining_mwh:,.2f} MWh")
    
    print(f"\nTotal remaining supply: {total_remaining_supply:,.2f} MWh")
    print(f"Total demand gap: {total_demand_gap:,.2f} MWh")
    print(f"Supply/Demand ratio: {total_remaining_supply/total_demand_gap:.2f}x" if total_demand_gap > 0 else "Supply/Demand ratio: N/A (no demand gap)")
    
    # Calculate average total_mwh by energy type for the entire facilities dataset
    avg_total_mwh_by_type = facilities_gdf.groupby('Grouped_Type')['total_mwh'].mean().to_dict()
    
    print(f"\nAverage total_mwh by energy type (for cluster range calculation):")
    for energy_type, avg_mwh in sorted(avg_total_mwh_by_type.items()):
        print(f"  {energy_type}: {avg_mwh:,.2f} MWh")
    
    # Calculate clusters per energy type with min/max range
    # min = 1 cluster per facility (baseline)
    # max = remaining_mwh / avg_total_mwh if remaining_mwh > avg_total_mwh
    clusters_per_type = {}
    total_clusters = 0
    
    print(f"\nCluster allocation by energy type (with range based on capacity):")
    for energy_type in remaining_by_type.keys():
        # Get facilities of this type with remaining capacity
        type_facilities = facilities_with_capacity[facilities_with_capacity['Grouped_Type'] == energy_type]
        
        min_clusters_for_type = len(type_facilities)  # 1 per facility (baseline)
        
        # Calculate max clusters: if any facility has remaining_mwh > avg_total_mwh, allow splitting
        avg_total_mwh = avg_total_mwh_by_type.get(energy_type, 0)
        max_clusters_for_type = min_clusters_for_type
        
        if avg_total_mwh > 0:
            for _, facility in type_facilities.iterrows():
                facility_remaining = facility['remaining_mwh']
                if facility_remaining > avg_total_mwh:
                    # This facility can be split into multiple clusters
                    additional_clusters = int(facility_remaining / avg_total_mwh) - 1
                    max_clusters_for_type += additional_clusters
        
        clusters_per_type[energy_type] = max_clusters_for_type
        total_clusters += max_clusters_for_type
        
        remaining_mwh = remaining_by_type[energy_type]
        capacity_per_cluster = remaining_mwh / max_clusters_for_type if max_clusters_for_type > 0 else 0
        print(f"  {energy_type}: {len(type_facilities)} facilities → {min_clusters_for_type}-{max_clusters_for_type} clusters ({remaining_mwh:,.2f} MWh → {capacity_per_cluster:,.2f} MWh per cluster)")
    
    # Ensure we don't exceed number of available settlements
    n_settlements = len(settlements_gdf)
    if total_clusters > n_settlements:
        print(f"\nWarning: Calculated {total_clusters} clusters exceeds {n_settlements} settlements.")
        print(f"  Scaling down clusters proportionally...")
        scale_factor = n_settlements / total_clusters
        for energy_type in clusters_per_type:
            clusters_per_type[energy_type] = max(1, int(clusters_per_type[energy_type] * scale_factor))
        total_clusters = sum(clusters_per_type.values())
        print(f"  Adjusted total clusters: {total_clusters}")
    
    print(f"\nFinal cluster configuration:")
    print(f"  Total clusters across all types: {total_clusters}")
    for energy_type, n_clusters in sorted(clusters_per_type.items()):
        capacity_per_cluster = remaining_by_type[energy_type] / n_clusters
        print(f"  {energy_type}: {n_clusters} clusters × {capacity_per_cluster:,.2f} MWh = {remaining_by_type[energy_type]:,.2f} MWh")
    
    return total_clusters, clusters_per_type, total_remaining_supply, total_demand_gap


def filter_settlements(centroids_gdf):
    """Step 1: Filter settlements by removing 'Filled' and low population, then dropping bottom 1% by demand."""
    print("\n" + "="*60)
    print("STEP 1: FILTER SETTLEMENTS")
    print("="*60)
    
    status_counts = centroids_gdf['supply_status'].value_counts()
    print(f"\nInitial settlement status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")
    
    # Check what unique status values exist
    print(f"\nUnique status values in data: {centroids_gdf['supply_status'].unique().tolist()}")
    
    # Keep only "Partially Filled" and "Not Filled" settlements
    unfilled = centroids_gdf[centroids_gdf['supply_status'].isin(['Partially Filled', 'Not Filled'])].copy()
    print(f"\nAfter keeping only 'Partially Filled' and 'Not Filled': {len(unfilled):,} settlements")
    
    if len(unfilled) > 0:
        unfilled_status_counts = unfilled['supply_status'].value_counts()
        print(f"Breakdown:")
        for status, count in unfilled_status_counts.items():
            print(f"  {status}: {count:,}")
    
    # Remove settlements with population < 100
    pop_col = 'Population_centroid'
    if pop_col in unfilled.columns:
        unfilled = unfilled[unfilled[pop_col] >= 100].copy()
        print(f"After removing settlements with population < 100: {len(unfilled):,} settlements")
    else:
        print(f"Warning: '{pop_col}' column not found, skipping population filter")
    
    if len(unfilled) == 0:
        print("No unfilled settlements found!")
    
    return unfilled


def cluster_settlements(settlements_gdf, n_clusters):
    """Cluster settlements using weighted K-means based on demand_gap_mwh."""
    print("\n" + "="*60)
    print("STEP 2: CLUSTER SETTLEMENTS (WEIGHTED K-MEANS)")
    print("="*60)
    
    if len(settlements_gdf) == 0:
        print("No settlements to cluster!")
        return settlements_gdf
    
    if n_clusters <= 0:
        print(f"Invalid number of clusters: {n_clusters}. Using 1 cluster as fallback.")
        n_clusters = 1
    
    # Ensure n_clusters doesn't exceed number of settlements
    n_clusters = min(n_clusters, len(settlements_gdf))
    
    print(f"\nClustering parameters:")
    print(f"  Algorithm: Weighted K-means")
    print(f"  Weight metric: demand_gap_mwh")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Total settlements: {len(settlements_gdf)}")
    
    coords = np.column_stack([
        settlements_gdf.geometry.x,
        settlements_gdf.geometry.y
    ])
    
    # Use demand_gap_mwh as sample weights for clustering
    weights = settlements_gdf['demand_gap_mwh'].values
    if weights.sum() > 0:
        weights = weights / weights.sum()  # Normalize
    else:
        weights = np.ones(len(weights)) / len(weights)  # Equal weights if no demand gap
    
    print(f"\nDemand gap statistics:")
    print(f"  Total demand gap: {settlements_gdf['demand_gap_mwh'].sum():,.2f} MWh")
    print(f"  Average demand gap: {settlements_gdf['demand_gap_mwh'].mean():,.2f} MWh")
    print(f"  Max demand gap: {settlements_gdf['demand_gap_mwh'].max():,.2f} MWh")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords, sample_weight=weights)
    
    settlements_gdf['cluster_id'] = cluster_labels
    
    print(f"\nClustering results:")
    cluster_sizes = settlements_gdf.groupby('cluster_id').size().sort_values(ascending=False)
    print(f"  Largest cluster: {cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Smallest cluster: {cluster_sizes.iloc[-1] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Average cluster size: {cluster_sizes.mean():.1f} settlements")
    print(f"  Median cluster size: {cluster_sizes.median():.1f} settlements")
    
    # Report total demand gap per cluster
    cluster_demand = settlements_gdf.groupby('cluster_id')['demand_gap_mwh'].sum().sort_values(ascending=False)
    print(f"\nDemand gap by cluster:")
    print(f"  Highest demand cluster: {cluster_demand.iloc[0]:,.2f} MWh")
    print(f"  Lowest demand cluster: {cluster_demand.iloc[-1]:,.2f} MWh")
    print(f"  Average per cluster: {cluster_demand.mean():,.2f} MWh")
    
    return settlements_gdf


def calculate_cluster_centers(settlements_gdf, facilities_gdf, clusters_per_type, remaining_by_type):
    """Match settlements to facilities such that cluster demand_gap_mwh equals facility remaining_mwh."""
    
    if len(settlements_gdf) == 0:
        return gpd.GeoDataFrame(columns=['cluster_id', 'geometry', 'num_settlements', 
                                        'Grouped_Type', 'remaining_mwh', 'center_lon', 'center_lat'],
                               crs=COMMON_CRS)
    
    # Get facilities with remaining capacity
    facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy().reset_index(drop=True)
    
    print(f"\nMatching settlements to facilities (demand_gap_mwh = remaining_mwh):")
    print(f"  Total settlements: {len(settlements_gdf)}")
    print(f"  Total facilities with capacity: {len(facilities_with_capacity)}")
    print(f"  Total demand gap: {settlements_gdf['demand_gap_mwh'].sum():,.2f} MWh")
    print(f"  Total remaining capacity: {facilities_with_capacity['remaining_mwh'].sum():,.2f} MWh")
    
    # Create clusters by iteratively assigning settlements to facilities
    # Strategy: For each facility, find settlements that match its remaining_mwh
    
    settlements_remaining = settlements_gdf.copy()
    settlements_remaining['assigned_cluster'] = -1
    
    cluster_info = []
    cluster_id = 0
    
    # Sort facilities by remaining capacity (largest first) for better matching
    facilities_sorted = facilities_with_capacity.sort_values('remaining_mwh', ascending=False)
    
    for fac_idx, facility in facilities_sorted.iterrows():
        if len(settlements_remaining[settlements_remaining['assigned_cluster'] == -1]) == 0:
            break
        
        facility_capacity = facility['remaining_mwh']
        facility_type = facility['Grouped_Type']
        facility_geom = facility.geometry
        
        # Find unassigned settlements
        unassigned = settlements_remaining[settlements_remaining['assigned_cluster'] == -1].copy()
        
        if len(unassigned) == 0:
            continue
        
        # Calculate distance from facility to each unassigned settlement
        unassigned['distance_to_facility'] = unassigned.geometry.apply(lambda g: g.distance(facility_geom))
        
        # Sort by distance (nearest first)
        unassigned = unassigned.sort_values('distance_to_facility')
        
        # Greedily assign settlements until we match the facility's capacity
        cumulative_demand = 0
        selected_settlements = []
        
        for idx, settlement in unassigned.iterrows():
            settlement_demand = settlement['demand_gap_mwh']
            
            # Check if adding this settlement would exceed capacity
            if cumulative_demand + settlement_demand <= facility_capacity * 1.1:  # Allow 10% tolerance
                selected_settlements.append(idx)
                cumulative_demand += settlement_demand
                
                # Stop if we've matched the capacity (within tolerance)
                if abs(cumulative_demand - facility_capacity) / facility_capacity < 0.05:  # Within 5%
                    break
        
        # If no settlements selected but unassigned exist, take at least one
        if len(selected_settlements) == 0 and len(unassigned) > 0:
            selected_settlements.append(unassigned.index[0])
            cumulative_demand = unassigned.iloc[0]['demand_gap_mwh']
        
        # Assign these settlements to this cluster
        settlements_remaining.loc[selected_settlements, 'assigned_cluster'] = cluster_id
        
        # Calculate weighted cluster center
        cluster_settlements = settlements_remaining.loc[selected_settlements]
        weights = cluster_settlements['demand_gap_mwh'].values
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        center_lon = np.average(cluster_settlements.geometry.x, weights=weights)
        center_lat = np.average(cluster_settlements.geometry.y, weights=weights)
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'geometry': Point(center_lon, center_lat),
            'num_settlements': len(selected_settlements),
            'demand_gap_mwh': cumulative_demand,
            'Grouped_Type': facility_type,
            'remaining_mwh': facility_capacity,  # Use facility's actual remaining capacity
            'center_lon': center_lon,
            'center_lat': center_lat,
            'matched_facility_id': fac_idx,
            'capacity_match_ratio': cumulative_demand / facility_capacity if facility_capacity > 0 else 0
        })
        
        cluster_id += 1
    
    # Handle any remaining unassigned settlements
    unassigned_final = settlements_remaining[settlements_remaining['assigned_cluster'] == -1]
    if len(unassigned_final) > 0:
        print(f"\nWarning: {len(unassigned_final)} settlements could not be matched to facilities")
        print(f"  Unassigned demand: {unassigned_final['demand_gap_mwh'].sum():,.2f} MWh")
    
    # Update settlements_gdf with cluster assignments
    settlements_gdf['cluster_id'] = settlements_remaining['assigned_cluster'].values
    
    cluster_centers_gdf = gpd.GeoDataFrame(cluster_info, crs=COMMON_CRS)
    
    print(f"\nCluster matching results:")
    print(f"  Total clusters created: {len(cluster_centers_gdf)}")
    print(f"  Total demand in clusters: {cluster_centers_gdf['demand_gap_mwh'].sum():,.2f} MWh")
    print(f"  Total capacity allocated: {cluster_centers_gdf['remaining_mwh'].sum():,.2f} MWh")
    print(f"  Average capacity match ratio: {cluster_centers_gdf['capacity_match_ratio'].mean():.2%}")
    
    print(f"\nCapacity matching by energy type:")
    for energy_type in sorted(cluster_centers_gdf['Grouped_Type'].unique()):
        type_clusters = cluster_centers_gdf[cluster_centers_gdf['Grouped_Type'] == energy_type]
        type_demand = type_clusters['demand_gap_mwh'].sum()
        type_capacity = type_clusters['remaining_mwh'].sum()
        match_ratio = type_demand / type_capacity if type_capacity > 0 else 0
        print(f"  {energy_type}: {len(type_clusters)} clusters, demand: {type_demand:,.2f} MWh, capacity: {type_capacity:,.2f} MWh (ratio: {match_ratio:.2%})")
    
    return cluster_centers_gdf


def compute_grid_distances(cluster_centers_gdf, grid_lines_gdf):
    """Step 3: Compute distance from cluster centers to nearest grid lines."""
    print("\n" + "="*60)
    print("STEP 3: COMPUTE GRID DISTANCES")
    print("="*60)
    
    if len(cluster_centers_gdf) == 0:
        print("No cluster centers to analyze!")
        return cluster_centers_gdf
    
    if len(grid_lines_gdf) == 0:
        print("Warning: No grid lines available!")
        cluster_centers_gdf['distance_to_grid_km'] = np.inf
        cluster_centers_gdf['nearest_grid_lon'] = np.nan
        cluster_centers_gdf['nearest_grid_lat'] = np.nan
        cluster_centers_gdf['is_remote'] = True
        return cluster_centers_gdf
    
    grid_union = grid_lines_gdf.geometry.unary_union
    
    distances = []
    nearest_points_list = []
    
    print(f"\nCalculating distances for {len(cluster_centers_gdf)} cluster centers...")
    
    for idx, cluster in cluster_centers_gdf.iterrows():
        cluster_point = cluster.geometry
        
        nearest_geoms = nearest_points(cluster_point, grid_union)
        nearest_grid_point = nearest_geoms[1]
        
        distance_km = haversine_distance_km(
            cluster_point.y, cluster_point.x,
            nearest_grid_point.y, nearest_grid_point.x
        )
        
        distances.append(distance_km)
        nearest_points_list.append(nearest_grid_point)
    
    cluster_centers_gdf['distance_to_grid_km'] = distances
    cluster_centers_gdf['nearest_grid_lon'] = [pt.x for pt in nearest_points_list]
    cluster_centers_gdf['nearest_grid_lat'] = [pt.y for pt in nearest_points_list]
    cluster_centers_gdf['is_remote'] = cluster_centers_gdf['distance_to_grid_km'] > GRID_DISTANCE_THRESHOLD_KM
    
    print(f"\nGrid distance statistics:")
    print(f"  Mean distance: {cluster_centers_gdf['distance_to_grid_km'].mean():.2f} km")
    print(f"  Median distance: {cluster_centers_gdf['distance_to_grid_km'].median():.2f} km")
    print(f"  Max distance: {cluster_centers_gdf['distance_to_grid_km'].max():.2f} km")
    print(f"  Min distance: {cluster_centers_gdf['distance_to_grid_km'].min():.2f} km")
    
    n_remote = cluster_centers_gdf['is_remote'].sum()
    n_near = len(cluster_centers_gdf) - n_remote
    print(f"\nCluster classification:")
    print(f"  Near grid (<{GRID_DISTANCE_THRESHOLD_KM}km): {n_near} clusters")
    print(f"  Remote (≥{GRID_DISTANCE_THRESHOLD_KM}km): {n_remote} clusters")
    
    return cluster_centers_gdf


def build_steiner_network(settlements_gdf, cluster_id):
    """Step 4: Build Steiner tree network for a remote cluster using approximation algorithm."""
    cluster_settlements = settlements_gdf[settlements_gdf['cluster_id'] == cluster_id]
    
    if len(cluster_settlements) <= 1:
        return []
    
    if len(cluster_settlements) == 2:
        # For 2 points, Steiner tree is just a direct line
        settlement_list = cluster_settlements.reset_index(drop=True)
        point1 = settlement_list.iloc[0].geometry
        point2 = settlement_list.iloc[1].geometry
        distance_km = haversine_distance_km(point1.y, point1.x, point2.y, point2.x)
        return [{
            'geometry': LineString([point1, point2]),
            'cluster_id': cluster_id,
            'distance_km': distance_km,
            'from_idx': settlement_list.index[0],
            'to_idx': settlement_list.index[1],
            'is_steiner_point': False
        }]
    
    # For 3+ points, use sparse distance computation for scalability
    settlement_list = cluster_settlements.reset_index(drop=True)
    n_points = len(cluster_settlements)
    
    coords = np.column_stack([
        cluster_settlements.geometry.x,
        cluster_settlements.geometry.y
    ])
    
    # For large clusters, use sparse k-nearest neighbors approach
    # For small clusters, use full distance matrix
    SPARSE_THRESHOLD = 300  # Switch to sparse computation above this size
    K_NEIGHBORS = min(15, n_points - 1)  # Number of nearest neighbors to consider
    
    if n_points > SPARSE_THRESHOLD:
        # Sparse approach: Build MST using k-nearest neighbors
        from scipy.spatial import KDTree
        
        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(coords)
        
        # For each point, find k nearest neighbors
        edges_candidates = []
        for i in range(n_points):
            point = coords[i]
            distances, indices = tree.query(point, k=K_NEIGHBORS + 1)  # +1 because it includes itself
            
            for j, neighbor_idx in enumerate(indices[1:]):  # Skip first (itself)
                if neighbor_idx < n_points:
                    # Calculate actual haversine distance
                    dist_km = haversine_distance_km(
                        settlement_list.iloc[i].geometry.y,
                        settlement_list.iloc[i].geometry.x,
                        settlement_list.iloc[neighbor_idx].geometry.y,
                        settlement_list.iloc[neighbor_idx].geometry.x
                    )
                    edges_candidates.append((i, neighbor_idx, dist_km))
        
        # Build sparse distance matrix from k-nearest neighbors
        from scipy.sparse import lil_matrix
        sparse_dist = lil_matrix((n_points, n_points))
        
        for i, j, dist in edges_candidates:
            sparse_dist[i, j] = dist
            sparse_dist[j, i] = dist
        
        # Compute MST on sparse graph
        mst = minimum_spanning_tree(sparse_dist)
        mst_array = mst.toarray()
        
    else:
        # Dense approach: Full distance matrix for small clusters
        dist_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = haversine_distance_km(
                    settlement_list.iloc[i].geometry.y,
                    settlement_list.iloc[i].geometry.x,
                    settlement_list.iloc[j].geometry.y,
                    settlement_list.iloc[j].geometry.x
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Compute MST
        mst = minimum_spanning_tree(dist_matrix)
        mst_array = mst.toarray()
    
    # Build adjacency list from MST
    adjacency = {i: [] for i in range(n_points)}
    mst_edges = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                adjacency[i].append(j)
                adjacency[j].append(i)
                dist_val = mst_array[i, j] if mst_array[i, j] > 0 else mst_array[j, i]
                mst_edges.append((i, j, dist_val))
    
    # Identify vertices with degree >= 3 (junction points for Steiner optimization)
    high_degree_vertices = [v for v in range(n_points) if len(adjacency[v]) >= 3]
    
    if not high_degree_vertices:
        # No junctions - return MST as-is
        edges = []
        for i, j, dist in mst_edges:
            point1 = settlement_list.iloc[i].geometry
            point2 = settlement_list.iloc[j].geometry
            edges.append({
                'geometry': LineString([point1, point2]),
                'cluster_id': cluster_id,
                'distance_km': dist,
                'from_idx': settlement_list.index[i],
                'to_idx': settlement_list.index[j],
                'is_steiner_point': False
            })
        return edges
    
    # Add Steiner points at high-degree junctions and rebuild connections
    # Strategy: Replace each high-degree vertex with a Steiner point at optimal location
    edges = []
    steiner_point_map = {}  # Maps original vertex -> steiner point info
    steiner_node_offset = 1000000  # Offset to distinguish Steiner node IDs from settlement IDs
    
    for vertex in high_degree_vertices:
        neighbors = adjacency[vertex]
        
        # Calculate Steiner point location (centroid of vertex + neighbors)
        all_junction_points = [coords[vertex]] + [coords[n] for n in neighbors]
        steiner_x = np.mean([p[0] for p in all_junction_points])
        steiner_y = np.mean([p[1] for p in all_junction_points])
        
        steiner_point_map[vertex] = {
            'geometry': Point(steiner_x, steiner_y),
            'node_id': steiner_node_offset + vertex  # Unique integer ID for Steiner point
        }
    
    # Now rebuild edges: 
    # - If edge connects to high-degree vertex, route through its Steiner point
    # - Otherwise, keep original edge
    for i, j, dist in mst_edges:
        point_i = settlement_list.iloc[i].geometry
        point_j = settlement_list.iloc[j].geometry
        idx_i = settlement_list.index[i]
        idx_j = settlement_list.index[j]
        
        # Check if either endpoint is a high-degree vertex (has Steiner point)
        i_has_steiner = i in steiner_point_map
        j_has_steiner = j in steiner_point_map
        
        if i_has_steiner and j_has_steiner:
            # Both ends are junctions - connect their Steiner points
            steiner_i = steiner_point_map[i]['geometry']
            steiner_j = steiner_point_map[j]['geometry']
            steiner_id_i = steiner_point_map[i]['node_id']
            steiner_id_j = steiner_point_map[j]['node_id']
            dist_steiner = haversine_distance_km(steiner_i.y, steiner_i.x, steiner_j.y, steiner_j.x)
            edges.append({
                'geometry': LineString([steiner_i, steiner_j]),
                'cluster_id': cluster_id,
                'distance_km': dist_steiner,
                'from_idx': steiner_id_i,
                'to_idx': steiner_id_j,
                'is_steiner_point': True
            })
        elif i_has_steiner:
            # i is junction - connect j to i's Steiner point
            steiner_i = steiner_point_map[i]['geometry']
            steiner_id_i = steiner_point_map[i]['node_id']
            dist_to_steiner = haversine_distance_km(point_j.y, point_j.x, steiner_i.y, steiner_i.x)
            edges.append({
                'geometry': LineString([point_j, steiner_i]),
                'cluster_id': cluster_id,
                'distance_km': dist_to_steiner,
                'from_idx': idx_j,
                'to_idx': steiner_id_i,
                'is_steiner_point': True
            })
        elif j_has_steiner:
            # j is junction - connect i to j's Steiner point
            steiner_j = steiner_point_map[j]['geometry']
            steiner_id_j = steiner_point_map[j]['node_id']
            dist_to_steiner = haversine_distance_km(point_i.y, point_i.x, steiner_j.y, steiner_j.x)
            edges.append({
                'geometry': LineString([point_i, steiner_j]),
                'cluster_id': cluster_id,
                'distance_km': dist_to_steiner,
                'from_idx': idx_i,
                'to_idx': steiner_id_j,
                'is_steiner_point': True
            })
        else:
            # Neither is junction - keep original MST edge
            edges.append({
                'geometry': LineString([point_i, point_j]),
                'cluster_id': cluster_id,
                'distance_km': dist,
                'from_idx': idx_i,
                'to_idx': idx_j,
                'is_steiner_point': False
            })
    
    return edges


def build_remote_networks(settlements_gdf, cluster_centers_gdf):
    """Build MST networks for all remote clusters."""
    print("\n" + "="*60)
    print("STEP 4: BUILD STEINER NETWORKS FOR REMOTE CLUSTERS")
    print("="*60)
    
    remote_clusters = cluster_centers_gdf[cluster_centers_gdf['is_remote']]['cluster_id'].values
    
    if len(remote_clusters) == 0:
        print("No remote clusters found!")
        return gpd.GeoDataFrame(columns=['geometry', 'cluster_id', 'distance_km', 
                                        'from_idx', 'to_idx'], crs=COMMON_CRS)
    
    print(f"\nBuilding networks for {len(remote_clusters)} remote clusters...")
    
    all_edges = []
    for cluster_id in remote_clusters:
        edges = build_steiner_network(settlements_gdf, cluster_id)
        all_edges.extend(edges)
        
        if edges:
            total_length = sum(e['distance_km'] for e in edges)
            print(f"  Cluster {cluster_id}: {len(edges)} edges, {total_length:.2f} km total")
    
    if not all_edges:
        print("No network edges generated!")
        return gpd.GeoDataFrame(columns=['geometry', 'cluster_id', 'distance_km', 
                                        'from_idx', 'to_idx', 'is_steiner_point'], crs=COMMON_CRS)
    
    networks_gdf = gpd.GeoDataFrame(all_edges, crs=COMMON_CRS)
    
    # Count Steiner points
    steiner_edges = networks_gdf[networks_gdf['is_steiner_point'] == True]
    print(f"\nTotal network edges: {len(networks_gdf)}")
    print(f"  Edges with Steiner points: {len(steiner_edges)}")
    print(f"  Direct settlement edges: {len(networks_gdf) - len(steiner_edges)}")
    print(f"Total network length: {networks_gdf['distance_km'].sum():.2f} km")
    
    return networks_gdf


def save_outputs(settlements_gdf, cluster_centers_gdf, networks_gdf, country_iso3, 
                output_dir="outputs_per_country"):
    """Save all outputs to parquet files."""
    scenario_suffix = f"{ANALYSIS_YEAR}_supply_{int(SUPPLY_FACTOR*100)}%"
    output_path = Path(output_dir) / "parquet" / scenario_suffix
    output_path.mkdir(parents=True, exist_ok=True)
    
    settlements_file = output_path / f"siting_settlements_{country_iso3}.parquet"
    settlements_gdf.to_parquet(settlements_file, compression='snappy')
    print(f"\nSaved settlements: {settlements_file}")
    
    centers_file = output_path / f"siting_clusters_{country_iso3}.parquet"
    cluster_centers_gdf.to_parquet(centers_file, compression='snappy')
    print(f"Saved cluster centers: {centers_file}")
    
    if len(networks_gdf) > 0:
        networks_file = output_path / f"siting_networks_{country_iso3}.parquet"
        networks_gdf.to_parquet(networks_file, compression='snappy')
        print(f"Saved networks: {networks_file}")
    
    summary_data = {
        'Parameter': [
            'Country',
            'Analysis_Year',
            'Supply_Factor_Pct',
            'Cluster_Radius_km',
            'Grid_Distance_Threshold_km',
            'Drop_Percentage',
            'Total_Unfilled_Settlements',
            'Total_Clusters',
            'Remote_Clusters',
            'Near_Grid_Clusters',
            'Total_Network_Edges',
            'Total_Network_Length_km'
        ],
        'Value': [
            country_iso3,
            ANALYSIS_YEAR,
            SUPPLY_FACTOR * 100,
            CLUSTER_RADIUS_KM,
            GRID_DISTANCE_THRESHOLD_KM,
            DROP_PERCENTAGE * 100,
            len(settlements_gdf),
            len(cluster_centers_gdf),
            cluster_centers_gdf['is_remote'].sum() if len(cluster_centers_gdf) > 0 else 0,
            (~cluster_centers_gdf['is_remote']).sum() if len(cluster_centers_gdf) > 0 else 0,
            len(networks_gdf),
            networks_gdf['distance_km'].sum() if len(networks_gdf) > 0 else 0
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    summary_file = output_path / f"siting_summary_{country_iso3}.xlsx"
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        worksheet = writer.sheets['Summary']
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 30
    
    print(f"Saved summary: {summary_file}")


def process_country_siting(country_iso3, output_dir="outputs_per_country"):
    """Main function to process siting analysis for a single country."""
    print("\n" + "="*60)
    print(f"SITING ANALYSIS FOR {country_iso3}")
    print("="*60)
    
    scenario_suffix = f"{ANALYSIS_YEAR}_supply_{int(SUPPLY_FACTOR*100)}%"
    
    try:
        centroids_gdf = load_centroids(country_iso3, scenario_suffix, output_dir)
        grid_lines_gdf = load_grid_lines(country_iso3, scenario_suffix, output_dir)
        facilities_gdf = load_facilities(country_iso3, scenario_suffix, output_dir)
        
        settlements_gdf = filter_settlements(centroids_gdf)
        
        if len(settlements_gdf) == 0:
            print("\nNo settlements to process!")
            return None
        
        # Calculate clusters based on remaining capacity (capacity-driven approach)
        n_clusters, clusters_per_type, total_supply, total_demand = calculate_num_clusters(facilities_gdf, settlements_gdf)
        
        # Calculate remaining capacity by energy type for allocation
        facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy()
        remaining_by_type = facilities_with_capacity.groupby('Grouped_Type')['remaining_mwh'].sum().to_dict()
        
        # Cluster settlements using weighted K-means (weighted by demand_gap_mwh)
        settlements_gdf = cluster_settlements(settlements_gdf, n_clusters)
        
        # Match each cluster to a specific facility considering demand gap + energy type + remaining capacity
        cluster_centers_gdf = calculate_cluster_centers(settlements_gdf, facilities_gdf, clusters_per_type, remaining_by_type)
        cluster_centers_gdf = compute_grid_distances(cluster_centers_gdf, grid_lines_gdf)
        networks_gdf = build_remote_networks(settlements_gdf, cluster_centers_gdf)
        
        save_outputs(settlements_gdf, cluster_centers_gdf, networks_gdf, country_iso3, output_dir)
        
        print("\n" + "="*60)
        print(f"SITING ANALYSIS COMPLETE FOR {country_iso3}")
        print("="*60)
        
        return str(Path(output_dir) / "parquet" / scenario_suffix)
        
    except Exception as e:
        print(f"\nError processing {country_iso3}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Parse command-line arguments and run the main processing function."""
    parser = argparse.ArgumentParser(
        description='Process siting analysis for remote settlement electrification'
    )
    parser.add_argument('country_iso3', help='ISO3 country code')
    parser.add_argument('--output-dir', default='outputs_per_country', 
                       help='Output directory')
    parser.add_argument('--run-all-scenarios', action='store_true',
                       help='Run all supply scenarios: 100%%, 90%%, 80%%, 70%%, 60%%')
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.run_all_scenarios:
        supply_factors = [1.0, 0.9, 0.8, 0.7, 0.6]
        print("\n" + "="*60)
        print("RUNNING ALL SUPPLY SCENARIOS: 100%, 90%, 80%, 70%, 60%")
        print("="*60)
    else:
        supply_factors = [1.0]  # Default: 100% supply scenario
    
    global SUPPLY_FACTOR
    all_success = True
    
    for supply_factor in supply_factors:
        SUPPLY_FACTOR = supply_factor
        
        if len(supply_factors) > 1:
            print(f"\n\n{'#'*60}")
            print(f"# PROCESSING SUPPLY SCENARIO: {int(SUPPLY_FACTOR*100)}%")
            print(f"{'#'*60}\n")
        
        result = process_country_siting(args.country_iso3, args.output_dir)
        
        if result:
            print(f"\n✓ Successfully processed {args.country_iso3} at {int(SUPPLY_FACTOR*100)}% supply")
        else:
            print(f"\n✗ Failed to process {args.country_iso3} at {int(SUPPLY_FACTOR*100)}% supply")
            all_success = False
    
    if len(supply_factors) > 1:
        print("\n" + "="*60)
        print("ALL SCENARIOS COMPLETE")
        print("="*60)
        if all_success:
            print(f"✓ Successfully processed all {len(supply_factors)} scenarios for {args.country_iso3}")
        else:
            print(f"⚠ Some scenarios failed for {args.country_iso3}")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
