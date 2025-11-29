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
    """Calculate optimal number of clusters based on supply-demand matching."""
    print("\n" + "="*60)
    print("CALCULATING NUMBER OF CLUSTERS FROM SUPPLY-DEMAND BALANCE")
    print("="*60)
    
    # Filter facilities with remaining capacity
    facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy()
    
    print(f"\nFacilities with remaining capacity: {len(facilities_with_capacity)}")
    
    if len(facilities_with_capacity) == 0:
        print("No facilities with remaining capacity found!")
        return 1, 0, 0
    
    # Calculate total remaining supply
    total_remaining_supply = facilities_with_capacity['remaining_mwh'].sum()
    
    # Calculate demand gap for settlements (Total_Demand - supply_received_mwh)
    demand_col = f'Total_Demand_{ANALYSIS_YEAR}_centroid'
    if demand_col in settlements_gdf.columns:
        settlements_gdf['demand_gap_mwh'] = settlements_gdf[demand_col] - settlements_gdf['supply_received_mwh']
    else:
        # Fallback: use supply_received_mwh as proxy for unmet demand
        settlements_gdf['demand_gap_mwh'] = settlements_gdf['supply_received_mwh']
    
    # Ensure demand_gap is non-negative
    settlements_gdf['demand_gap_mwh'] = settlements_gdf['demand_gap_mwh'].clip(lower=0)
    total_demand_gap = settlements_gdf['demand_gap_mwh'].sum()
    
    print(f"\nSupply-Demand Analysis:")
    print(f"  Total remaining supply: {total_remaining_supply:,.2f} MWh")
    print(f"  Total demand gap: {total_demand_gap:,.2f} MWh")
    print(f"  Supply/Demand ratio: {total_remaining_supply/total_demand_gap:.2f}x" if total_demand_gap > 0 else "  Supply/Demand ratio: N/A (no demand gap)")
    
    # Count facilities by Grouped_Type
    cluster_counts = facilities_with_capacity.groupby('Grouped_Type').size().to_dict()
    
    print(f"\nFacilities by energy type:")
    total_facilities = 0
    for energy_type, count in sorted(cluster_counts.items()):
        print(f"  {energy_type}: {count} facilities")
        total_facilities += count
    
    # Calculate optimal number of clusters
    # Use number of facilities as base, but ensure it's reasonable given settlements
    n_settlements = len(settlements_gdf)
    n_clusters = min(total_facilities, n_settlements)
    
    # If we have more supply than demand, we might need fewer clusters
    # If we have less supply than demand, use all facilities
    if total_demand_gap > 0:
        supply_demand_ratio = total_remaining_supply / total_demand_gap
        if supply_demand_ratio > 2:
            # Excess supply - reduce clusters slightly
            n_clusters = max(1, int(n_clusters * 0.7))
    
    print(f"\nCluster calculation:")
    print(f"  Total facilities with capacity: {total_facilities}")
    print(f"  Total settlements to cluster: {n_settlements}")
    print(f"  Optimal number of clusters: {n_clusters}")
    
    return n_clusters, total_remaining_supply, total_demand_gap


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
    """Step 2: Cluster unfilled settlements using K-means based on spatial proximity."""
    print("\n" + "="*60)
    print("STEP 2: CLUSTER SETTLEMENTS")
    print("="*60)
    
    if len(settlements_gdf) == 0:
        print("No settlements to cluster!")
        return settlements_gdf
    
    if n_clusters <= 0:
        print(f"Invalid number of clusters: {n_clusters}. Using 1 cluster as fallback.")
        n_clusters = 1
    
    # Ensure n_clusters doesn't exceed number of settlements
    n_clusters = min(n_clusters, len(settlements_gdf))
    
    coords = np.column_stack([
        settlements_gdf.geometry.x,
        settlements_gdf.geometry.y
    ])
    
    print(f"\nClustering parameters:")
    print(f"  Algorithm: K-means")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Total settlements: {len(settlements_gdf)}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    settlements_gdf['cluster_id'] = cluster_labels
    
    print(f"\nClustering results:")
    print(f"  Total clusters: {n_clusters}")
    
    cluster_sizes = settlements_gdf.groupby('cluster_id').size().sort_values(ascending=False)
    print(f"\nCluster size distribution:")
    print(f"  Largest cluster: {cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Smallest cluster: {cluster_sizes.iloc[-1] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Average cluster size: {cluster_sizes.mean():.1f} settlements")
    print(f"  Median cluster size: {cluster_sizes.median():.1f} settlements")
    
    return settlements_gdf


def calculate_cluster_centers(settlements_gdf):
    """Calculate cluster center points weighted by demand gap."""
    cluster_centers = []
    
    for cluster_id in settlements_gdf['cluster_id'].unique():
        cluster_points = settlements_gdf[settlements_gdf['cluster_id'] == cluster_id]
        
        # Use demand_gap_mwh as the primary metric
        if 'demand_gap_mwh' in cluster_points.columns:
            total_demand = cluster_points['demand_gap_mwh'].sum()
            weights = cluster_points['demand_gap_mwh'].values
        else:
            demand_col = f'Total_Demand_{ANALYSIS_YEAR}_centroid'
            if demand_col in cluster_points.columns:
                total_demand = cluster_points[demand_col].sum()
                weights = cluster_points[demand_col].values
            else:
                total_demand = cluster_points['supply_received_mwh'].sum()
                weights = cluster_points['supply_received_mwh'].values
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        center_lon = np.average(cluster_points.geometry.x, weights=weights)
        center_lat = np.average(cluster_points.geometry.y, weights=weights)
        
        cluster_centers.append({
            'cluster_id': cluster_id,
            'geometry': Point(center_lon, center_lat),
            'num_settlements': len(cluster_points),
            'total_demand_gap_mwh': total_demand,
            'center_lon': center_lon,
            'center_lat': center_lat
        })
    
    if not cluster_centers:
        return gpd.GeoDataFrame(columns=['cluster_id', 'geometry', 'num_settlements', 
                                        'total_demand_gap_mwh', 'center_lon', 'center_lat'],
                               crs=COMMON_CRS)
    
    cluster_centers_gdf = gpd.GeoDataFrame(cluster_centers, crs=COMMON_CRS)
    
    print(f"\nCluster centers summary:")
    print(f"  Total clusters: {len(cluster_centers_gdf)}")
    print(f"  Total demand gap across clusters: {cluster_centers_gdf['total_demand_gap_mwh'].sum():,.2f} MWh")
    print(f"  Average demand gap per cluster: {cluster_centers_gdf['total_demand_gap_mwh'].mean():,.2f} MWh")
    
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
    """Step 4: Build MST-style Steiner-like network for a remote cluster."""
    cluster_settlements = settlements_gdf[settlements_gdf['cluster_id'] == cluster_id]
    
    if len(cluster_settlements) <= 1:
        return []
    
    coords = np.column_stack([
        cluster_settlements.geometry.y,
        cluster_settlements.geometry.x
    ])
    
    n_points = len(coords)
    dist_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = haversine_distance_km(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    mst = minimum_spanning_tree(dist_matrix)
    mst_array = mst.toarray()
    
    edges = []
    settlement_list = cluster_settlements.reset_index(drop=True)
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                point1 = settlement_list.iloc[i].geometry
                point2 = settlement_list.iloc[j].geometry
                distance_km = haversine_distance_km(point1.y, point1.x, point2.y, point2.x)
                
                edges.append({
                    'geometry': LineString([point1, point2]),
                    'cluster_id': cluster_id,
                    'distance_km': distance_km,
                    'from_idx': settlement_list.index[i],
                    'to_idx': settlement_list.index[j]
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
                                        'from_idx', 'to_idx'], crs=COMMON_CRS)
    
    networks_gdf = gpd.GeoDataFrame(all_edges, crs=COMMON_CRS)
    
    print(f"\nTotal network edges: {len(networks_gdf)}")
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
        
        # Calculate optimal number of clusters based on supply-demand balance
        n_clusters, total_supply, total_demand = calculate_num_clusters(facilities_gdf, settlements_gdf)
        
        settlements_gdf = cluster_settlements(settlements_gdf, n_clusters)
        cluster_centers_gdf = calculate_cluster_centers(settlements_gdf)
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
