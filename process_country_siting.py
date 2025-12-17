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
import os
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

# Suppress warnings
warnings.filterwarnings("ignore")

def get_bigdata_path(folder_name):
    """
    Get the correct path for bigdata folders.
    Checks local path first, then cluster path if not found.
    
    Args:
        folder_name: Name of the bigdata folder (e.g., 'bigdata_gadm')
    
    Returns:
        str: Path to the folder
    """
    local_path = folder_name
    cluster_path = f"/soge-home/projects/mistral/ji/{folder_name}"
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(cluster_path):
        return cluster_path
    else:
        # Return local path as default (will trigger appropriate error if needed)
        return local_path

# Constants
COMMON_CRS = "EPSG:4326"
ANALYSIS_YEAR = 2030
SUPPLY_FACTOR = 1.0
CLUSTER_RADIUS_KM = 50
CLUSTER_MIN_SAMPLES = 1
GRID_DISTANCE_THRESHOLD_KM = 50
DROP_PERCENTAGE = 0.01
DEMAND_TYPES = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]


def load_country_energy_mix(country_iso3, year=2030):
    """Load country-specific energy mix from Ember projection data.
    
    Args:
        country_iso3: ISO3 country code
        year: Analysis year (2030 or 2050)
    
    Returns:
        Dictionary with energy type proportions, or None if not found
    """
    try:
        ember_file = Path('outputs_processed_data/p1_b_ember_2024_30_50.xlsx')
        if not ember_file.exists():
            print(f"Warning: Ember data file not found: {ember_file}")
            return None
        
        df = pd.read_excel(ember_file)
        country_data = df[df['ISO3_code'] == country_iso3]
        
        if len(country_data) == 0:
            print(f"Warning: No energy mix data found for {country_iso3}")
            return None
        
        country_data = country_data.iloc[0]
        
        # Get MWh columns for the analysis year
        mwh_cols = {
            'Hydro': f'Hydro_{year}_MWh',
            'Solar': f'Solar_{year}_MWh',
            'Wind': f'Wind_{year}_MWh',
            'Other Renewables': f'Other Renewables_{year}_MWh',
            'Nuclear': f'Nuclear_{year}_MWh',
            'Fossil': f'Fossil_{year}_MWh'
        }
        
        # Calculate total generation
        total_mwh = sum([country_data[col] for col in mwh_cols.values() if pd.notna(country_data[col])])
        
        if total_mwh == 0:
            print(f"Warning: Total generation is 0 for {country_iso3}")
            return None
        
        # Calculate proportions
        energy_mix = {}
        for energy_type, col in mwh_cols.items():
            if pd.notna(country_data[col]):
                energy_mix[energy_type] = country_data[col] / total_mwh
            else:
                energy_mix[energy_type] = 0.0
        
        return energy_mix
        
    except Exception as e:
        print(f"Error loading energy mix for {country_iso3}: {e}")
        return None


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def identify_geographic_components(settlements_gdf, max_distance_km=50):
    """Identify separate geographic components (islands, separated territories) using DBSCAN.
    
    This ensures settlements on different islands or separated land masses are not 
    clustered together, which is crucial for non-contiguous countries.
    
    Args:
        settlements_gdf: GeoDataFrame with settlement points
        max_distance_km: Maximum distance (km) to consider settlements as same component
                        Default 100km handles most island separations
    
    Returns:
        settlements_gdf with 'geo_component' column added
    """
    if len(settlements_gdf) == 0:
        settlements_gdf['geo_component'] = -1
        return settlements_gdf
    
    # Extract coordinates
    coords = np.column_stack([
        settlements_gdf.geometry.y,  # latitude
        settlements_gdf.geometry.x   # longitude
    ])
    
    # Convert max_distance_km to approximate degrees
    # At equator: 1 degree ≈ 111 km
    # Use conservative estimate accounting for latitude
    avg_lat = np.abs(coords[:, 0].mean())
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(avg_lat))
    km_per_degree = (km_per_degree_lat + km_per_degree_lon) / 2
    
    epsilon_degrees = max_distance_km / km_per_degree
    
    # Use DBSCAN to identify geographic components
    # eps: maximum distance between points in same component
    # min_samples: 1 (even isolated settlements form component)
    dbscan = DBSCAN(eps=epsilon_degrees, min_samples=1, metric='euclidean')
    component_labels = dbscan.fit_predict(coords)
    
    settlements_gdf['geo_component'] = component_labels
    
    n_components = len(set(component_labels)) - (1 if -1 in component_labels else 0)
    
    print(f"\nGeographic component detection:")
    print(f"  Max distance threshold: {max_distance_km} km")
    print(f"  Identified components: {n_components}")
    
    if n_components > 1:
        component_sizes = settlements_gdf.groupby('geo_component').size().sort_values(ascending=False)
        print(f"  Component sizes:")
        for comp_id, size in component_sizes.head(10).items():
            if comp_id >= 0:  # Skip noise points (-1)
                comp_settlements = settlements_gdf[settlements_gdf['geo_component'] == comp_id]
                total_demand = comp_settlements['demand_gap_mwh'].sum()
                print(f"    Component {comp_id}: {size} settlements ({total_demand:,.2f} MWh demand)")
    
    return settlements_gdf


def load_admin_boundaries(country_iso3):
    """Load administrative boundaries for a specific country from the GADM dataset."""
    gadm_file = os.path.join(get_bigdata_path('bigdata_gadm'), 'gadm_410-levels.gpkg')
    admin_boundaries = gpd.read_file(gadm_file, layer="ADM_0")
    country_data = admin_boundaries[admin_boundaries['GID_0'] == country_iso3]
    
    if country_data.empty:
        raise ValueError(f"No boundaries found for country {country_iso3}")
    
    return country_data


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


def calculate_num_clusters(facilities_gdf, settlements_gdf, all_centroids_gdf=None):
    """Calculate number of clusters based on remaining capacity by energy type (capacity-driven approach).
    
    Args:
        facilities_gdf: Facilities dataframe
        settlements_gdf: Filtered settlements (unfilled only)
        all_centroids_gdf: All centroids (including filled) for calculating total supply received
    """
    print("\n" + "="*60)
    print("CALCULATING CLUSTERS FROM REMAINING CAPACITY (CAPACITY-DRIVEN)")
    print("="*60)
    
    # Calculate demand gap for settlements first
    demand_col = f'Total_Demand_{ANALYSIS_YEAR}_centroid'
    if demand_col in settlements_gdf.columns:
        settlements_gdf['demand_gap_mwh'] = settlements_gdf[demand_col] - settlements_gdf['supply_received_mwh']
    else:
        settlements_gdf['demand_gap_mwh'] = settlements_gdf['supply_received_mwh']
    
    settlements_gdf['demand_gap_mwh'] = settlements_gdf['demand_gap_mwh'].clip(lower=0)
    total_demand_gap = settlements_gdf['demand_gap_mwh'].sum()
    
    print(f"\nTotal unfilled demand across all settlements: {total_demand_gap:,.2f} MWh")
    
    # Calculate supply received by energy type from ALL centroids (including filled settlements)
    # to get accurate total supply picture
    centroids_for_supply = all_centroids_gdf if all_centroids_gdf is not None else settlements_gdf
    
    supply_received_by_type = {}
    for energy_type in DEMAND_TYPES:
        type_supply_col = f"{energy_type}_{ANALYSIS_YEAR}_centroid"
        if type_supply_col in centroids_for_supply.columns:
            # Sum up what was actually delivered to all settlements for this type
            supply_received_by_type[energy_type] = centroids_for_supply[type_supply_col].sum()
        else:
            supply_received_by_type[energy_type] = 0
    
    total_supply_received = sum(supply_received_by_type.values())
    
    # If energy type columns don't exist but supply_received_mwh does, use that for total
    if total_supply_received == 0 and 'supply_received_mwh' in centroids_for_supply.columns:
        total_supply_received = centroids_for_supply['supply_received_mwh'].sum()
        print(f"\nSupply received by all settlements: {total_supply_received:,.2f} MWh")
        print("  (Energy type breakdown not available in data)")
    else:
        print(f"\nSupply received by settlements (by energy type):")
        for energy_type, received_mwh in sorted(supply_received_by_type.items()):
            if received_mwh > 0:
                print(f"  {energy_type}: {received_mwh:,.2f} MWh")
    print(f"Total supply received: {total_supply_received:,.2f} MWh")
    
    # Calculate expected supply mix from country-specific Ember projections
    # This represents the country's planned/expected energy generation mix
    country_energy_mix = load_country_energy_mix(settlements_gdf['GID_0'].iloc[0], year=ANALYSIS_YEAR)
    
    # Define fallback global energy mix (only used if country data unavailable)
    fallback_global_mix = {
        "Fossil": 0.60,        # Coal, gas, oil
        "Solar": 0.15,         # Solar PV
        "Wind": 0.10,          # Wind
        "Hydro": 0.08,         # Hydro
        "Nuclear": 0.05,       # Nuclear
        "Other Renewables": 0.02  # Geothermal, biomass, etc.
    }
    
    if country_energy_mix is not None:
        # Use country-specific energy mix from Ember projections
        expected_mix = country_energy_mix
        print(f"\nExpected energy mix (from Ember {ANALYSIS_YEAR} projections):")
        for energy_type, proportion in sorted(expected_mix.items()):
            if proportion > 0:
                print(f"  {energy_type}: {proportion:.1%}")
    else:
        # Fallback to global default if country data not available
        expected_mix = fallback_global_mix.copy()
        print(f"\nNo country-specific data - using default global energy mix:")
        for energy_type, proportion in sorted(expected_mix.items()):
            print(f"  {energy_type}: {proportion:.1%}")
    
    # Calculate expected supply by type for TOTAL demand (apply mix to total first)
    # Then subtract what's already been supplied to find the shortfall
    # Use centroids_for_supply (all centroids) to get accurate total demand
    if demand_col in centroids_for_supply.columns:
        total_demand = centroids_for_supply[demand_col].sum()
    else:
        total_demand = total_supply_received + total_demand_gap
    
    print(f"\nDemand breakdown:")
    print(f"  Total demand: {total_demand:,.2f} MWh")
    print(f"  Already supplied: {total_supply_received:,.2f} MWh")
    print(f"  Remaining demand: {total_demand_gap:,.2f} MWh")
    
    # Step 1: Apply energy mix to DEMAND GAP (unfilled demand) to get expected synthetic capacity
    # We're only creating synthetic facilities for the unfilled portion
    expected_supply_by_type = {}
    for energy_type in DEMAND_TYPES:
        # Expected synthetic capacity = demand_gap × mix proportion
        expected_supply_by_type[energy_type] = total_demand_gap * expected_mix[energy_type]
    
    print(f"\nExpected synthetic capacity by energy type (from remaining {total_demand_gap:,.2f} MWh):")
    for energy_type, expected_mwh in sorted(expected_supply_by_type.items()):
        if expected_mwh > 0:
            print(f"  {energy_type}: {expected_mwh:,.2f} MWh")
    
    # Step 2: Calculate shortfall for synthetic facilities
    # Since we filtered to unfilled settlements, shortfall = expected capacity needed
    shortfall_by_type = {}
    for energy_type in DEMAND_TYPES:
        shortfall = expected_supply_by_type[energy_type]
        if shortfall > 0:
            shortfall_by_type[energy_type] = shortfall
    
    print(f"\nSynthetic facility capacity needed by energy type:")
    for energy_type, shortfall in sorted(shortfall_by_type.items()):
        if shortfall > 0:
            print(f"  {energy_type}: {shortfall:,.2f} MWh")
    
    # Filter facilities with remaining capacity (for reference/logging only)
    facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy()
    
    print(f"\nExisting facilities with remaining capacity: {len(facilities_with_capacity)}")
    
    # Calculate remaining supply by energy type from existing facilities (for reference only)
    remaining_by_type = facilities_with_capacity.groupby('Grouped_Type')['remaining_mwh'].sum().to_dict() if len(facilities_with_capacity) > 0 else {}
    
    print(f"\nExisting remaining capacity by energy type (cannot reach unfilled settlements):")
    for energy_type in DEMAND_TYPES:
        remaining = remaining_by_type.get(energy_type, 0)
        if remaining > 0:
            print(f"  {energy_type}: {remaining:,.2f} MWh (already used in cluster calculations)")
    
    # Create synthetic facilities for ALL shortfall (existing facilities can't reach unfilled settlements)
    # The existing remaining capacity is already factored into cluster number calculations
    # and physically cannot reach the unfilled settlements (that's why they remain unfilled)
    # 
    # Strategy: Create synthetic facilities per geographic component, ensuring capacity matches
    # component demand. Excess capacity is reallocated to other components.
    
    # First, identify geographic components for settlements (will be done later in clustering, 
    # but we need it now for component-aware synthetic facility placement)
    settlements_gdf_temp = identify_geographic_components(settlements_gdf.copy(), max_distance_km=50)
    
    # Calculate demand per component
    component_demand_gap = settlements_gdf_temp.groupby('geo_component')['demand_gap_mwh'].sum().to_dict()
    
    # Filter out small components (< 5 settlements) and get viable components
    component_sizes = settlements_gdf_temp.groupby('geo_component').size()
    viable_components = component_sizes[component_sizes >= 5].index.tolist()
    
    print(f"\nCreating synthetic facilities per geographic component:")
    print(f"  Viable components (≥5 settlements): {len(viable_components)}")
    
    synthetic_facilities = []
    
    # Sort energy types by total shortfall (smallest first)
    sorted_energy_types = sorted(shortfall_by_type.items(), key=lambda x: x[1])
    
    # Track which components have been assigned facilities
    components_with_facilities = set()
    
    for energy_type, total_shortfall in sorted_energy_types:
        if total_shortfall <= 0:
            continue
        
        print(f"\n  {energy_type}: {total_shortfall:,.2f} MWh total shortfall")
        
        # Try to find a component that doesn't have a facility yet and has demand matching this type
        assigned = False
        
        # Sort components by demand (largest first) to prioritize larger components
        component_demands = [(comp_id, component_demand_gap.get(comp_id, 0)) 
                            for comp_id in viable_components if comp_id >= 0]
        component_demands.sort(key=lambda x: x[1], reverse=True)
        
        for comp_id, comp_demand in component_demands:
            # Skip if this component already has a facility
            if comp_id in components_with_facilities:
                continue
            
            comp_settlements = settlements_gdf_temp[settlements_gdf_temp['geo_component'] == comp_id]
            
            # Check if component's demand is close to this energy type's shortfall
            # Use capacity = component's total demand (not proportional to energy mix)
            facility_capacity = comp_demand
            
            # Calculate optimal location: weighted centroid of settlements in this component
            settlements_with_demand = comp_settlements[comp_settlements['demand_gap_mwh'] > 0].copy()
            
            if len(settlements_with_demand) > 0:
                weights = settlements_with_demand['demand_gap_mwh'].values
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(weights)) / len(weights)
                
                optimal_lon = np.average(settlements_with_demand.geometry.x, weights=weights)
                optimal_lat = np.average(settlements_with_demand.geometry.y, weights=weights)
                optimal_location = Point(optimal_lon, optimal_lat)
            else:
                optimal_location = comp_settlements.geometry.centroid.iloc[0] if len(comp_settlements) > 0 else Point(0, 0)
            
            synthetic_facility = {
                'gem_id': f'SYNTHETIC_{energy_type}_C{comp_id}',
                'Grouped_Type': energy_type,
                'remaining_mwh': facility_capacity,
                'total_mwh': facility_capacity,
                'available_total_mwh': facility_capacity,
                'center_lat': optimal_location.y,
                'center_lon': optimal_location.x,
                'geometry': optimal_location,
                'geo_component': comp_id
            }
            synthetic_facilities.append(synthetic_facility)
            components_with_facilities.add(comp_id)
            assigned = True
            
            print(f"    Component {comp_id}: {facility_capacity:,.2f} MWh at ({optimal_location.y:.4f}, {optimal_location.x:.4f})")
            break
        
        # If no suitable component found, place in largest component
        if not assigned and len(viable_components) > 0:
            component_demands_dict = {comp_id: component_demand_gap.get(comp_id, 0) 
                                     for comp_id in viable_components if comp_id >= 0}
            largest_component_id = max(component_demands_dict.keys(), key=lambda k: component_demands_dict[k])
            largest_component_settlements = settlements_gdf_temp[settlements_gdf_temp['geo_component'] == largest_component_id]
            
            settlements_with_demand = largest_component_settlements[largest_component_settlements['demand_gap_mwh'] > 0].copy()
            
            if len(settlements_with_demand) > 0:
                weights = settlements_with_demand['demand_gap_mwh'].values
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(weights)) / len(weights)
                
                optimal_lon = np.average(settlements_with_demand.geometry.x, weights=weights)
                optimal_lat = np.average(settlements_with_demand.geometry.y, weights=weights)
                optimal_location = Point(optimal_lon, optimal_lat)
            else:
                optimal_location = largest_component_settlements.geometry.centroid.iloc[0] if len(largest_component_settlements) > 0 else Point(0, 0)
            
            synthetic_facility = {
                'gem_id': f'SYNTHETIC_{energy_type}_C{largest_component_id}',
                'Grouped_Type': energy_type,
                'remaining_mwh': total_shortfall,
                'total_mwh': total_shortfall,
                'available_total_mwh': total_shortfall,
                'center_lat': optimal_location.y,
                'center_lon': optimal_location.x,
                'geometry': optimal_location,
                'geo_component': largest_component_id
            }
            synthetic_facilities.append(synthetic_facility)
            
            print(f"    Component {largest_component_id} (fallback): {total_shortfall:,.2f} MWh at ({optimal_location.y:.4f}, {optimal_location.x:.4f})")
    
    # Track all synthetic capacity for cluster calculations
    for facility in synthetic_facilities:
        energy_type = facility['Grouped_Type']
        remaining_by_type[energy_type] = remaining_by_type.get(energy_type, 0) + facility['remaining_mwh']
    
    # Append synthetic facilities
    if synthetic_facilities:
        synthetic_gdf = gpd.GeoDataFrame(synthetic_facilities, crs=COMMON_CRS)
        facilities_with_capacity = pd.concat([facilities_with_capacity, synthetic_gdf], ignore_index=True)
        print(f"\n✓ Created {len(synthetic_facilities)} synthetic facilities")
    else:
        print(f"\n✓ Sufficient capacity available (no synthetic facilities needed)")
    
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
    
    total_remaining_supply = sum(remaining_by_type.values())
    
    return total_clusters, clusters_per_type, total_remaining_supply, total_demand_gap, facilities_with_capacity


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
    """Cluster settlements using weighted K-means, respecting geographic components."""
    print("\n" + "="*60)
    print("STEP 2: CLUSTER SETTLEMENTS (WITHIN GEOGRAPHIC COMPONENTS)")
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
    print(f"  Algorithm: Demand-weighted K-means with geographic validation")
    print(f"  Weight metric: demand_gap_mwh")
    print(f"  Total settlements: {len(settlements_gdf)}")
    print(f"  Desired total clusters: {n_clusters}")
    
    # STEP 1: Perform demand-weighted K-means clustering across all settlements
    print(f"\nSTEP 1: Initial demand-weighted clustering")
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
    
    if n_clusters == 1 or len(settlements_gdf) == 1:
        settlements_gdf['cluster_id'] = 0
        print(f"  Single cluster for all {len(settlements_gdf)} settlements")
    else:
        # K-means clustering with demand weighting
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        settlements_gdf['cluster_id'] = kmeans.fit_predict(coords, sample_weight=weights)
        print(f"  Created {n_clusters} demand-weighted clusters")
    
    # STEP 2: Identify geographic components and validate clusters
    print(f"\nSTEP 2: Geographic component validation (50km threshold)")
    settlements_gdf = identify_geographic_components(settlements_gdf, max_distance_km=50)
    
    n_components = len(set(settlements_gdf['geo_component'])) - (1 if -1 in settlements_gdf['geo_component'].values else 0)
    component_sizes = settlements_gdf.groupby('geo_component').size()
    
    print(f"  Identified {n_components} geographic components")
    
    # Check for clusters that span multiple geographic components
    cluster_component_violations = []
    for cluster_id in settlements_gdf['cluster_id'].unique():
        if cluster_id < 0:
            continue
        cluster_mask = settlements_gdf['cluster_id'] == cluster_id
        cluster_components = settlements_gdf[cluster_mask]['geo_component'].unique()
        
        if len(cluster_components) > 1:
            # This cluster spans multiple geographic components - need to split
            cluster_component_violations.append((cluster_id, cluster_components))
    
    if cluster_component_violations:
        print(f"\n  Found {len(cluster_component_violations)} clusters spanning multiple geographic components")
        print(f"  Splitting these clusters to respect geographic boundaries...")
        
        # Reassign cluster IDs to ensure each cluster is within a single component
        next_cluster_id = settlements_gdf['cluster_id'].max() + 1
        
        for cluster_id, components in cluster_component_violations:
            cluster_mask = settlements_gdf['cluster_id'] == cluster_id
            
            # For each component in this cluster, create a new sub-cluster
            for i, comp_id in enumerate(components):
                comp_cluster_mask = cluster_mask & (settlements_gdf['geo_component'] == comp_id)
                
                if i == 0:
                    # Keep first component with original cluster_id
                    continue
                else:
                    # Assign new cluster_id to other components
                    settlements_gdf.loc[comp_cluster_mask, 'cluster_id'] = next_cluster_id
                    next_cluster_id += 1
        
        final_n_clusters = len(settlements_gdf['cluster_id'].unique())
        print(f"  Split complete: {n_clusters} → {final_n_clusters} clusters")
    else:
        print(f"  ✓ All clusters respect geographic boundaries")
        final_n_clusters = n_clusters
    
    # STEP 3: Summary statistics
    print(f"\nSTEP 3: Clustering summary")
    cluster_sizes = settlements_gdf.groupby('cluster_id').size().sort_values(ascending=False)
    print(f"  Total clusters: {final_n_clusters}")
    print(f"  Largest cluster: {cluster_sizes.iloc[0]} settlements")
    print(f"  Smallest cluster: {cluster_sizes.iloc[-1]} settlements")
    print(f"  Average: {cluster_sizes.mean():.1f} settlements/cluster")
    print(f"  Median: {cluster_sizes.median():.1f} settlements/cluster")
    
    cluster_demand = settlements_gdf.groupby('cluster_id')['demand_gap_mwh'].sum().sort_values(ascending=False)
    print(f"\n  Demand distribution:")
    print(f"    Highest demand cluster: {cluster_demand.iloc[0]:,.2f} MWh")
    print(f"    Lowest demand cluster: {cluster_demand.iloc[-1]:,.2f} MWh")
    print(f"    Average: {cluster_demand.mean():,.2f} MWh/cluster")
    
    # Show cluster distribution by component
    cluster_component_summary = settlements_gdf.groupby(['geo_component', 'cluster_id']).size().reset_index(name='settlements')
    component_cluster_counts = cluster_component_summary.groupby('geo_component').size()
    
    print(f"\n  Clusters per geographic component:")
    for comp_id in sorted(component_cluster_counts.index):
        if comp_id >= 0:
            comp_clusters = component_cluster_counts[comp_id]
            comp_settlements = component_sizes[comp_id]
            comp_demand = settlements_gdf[settlements_gdf['geo_component'] == comp_id]['demand_gap_mwh'].sum()
            print(f"    Component {comp_id}: {comp_clusters} clusters, {comp_settlements} settlements ({comp_demand:,.2f} MWh)")
    
    return settlements_gdf


def calculate_cluster_centers(settlements_gdf, facilities_gdf, clusters_per_type, remaining_by_type):
    """Assign facilities to demand-weighted clusters. Clusters can receive multiple facilities as needed."""
    
    if len(settlements_gdf) == 0:
        return gpd.GeoDataFrame(columns=['cluster_id', 'geometry', 'num_settlements', 
                                        'Grouped_Type', 'remaining_mwh', 'center_lon', 'center_lat'],
                               crs=COMMON_CRS)
    
    # Get facilities with remaining capacity
    facilities_with_capacity = facilities_gdf[facilities_gdf['remaining_mwh'] > 0].copy().reset_index(drop=True)
    
    print(f"\nMatching facilities to demand-weighted clusters:")
    print(f"  Total settlements: {len(settlements_gdf)}")
    print(f"  Total facilities with capacity: {len(facilities_with_capacity)}")
    print(f"  Total demand gap: {settlements_gdf['demand_gap_mwh'].sum():,.2f} MWh")
    print(f"  Total remaining capacity: {facilities_with_capacity['remaining_mwh'].sum():,.2f} MWh")
    
    # Get existing clusters from settlements (created by demand-weighted clustering)
    if 'cluster_id' not in settlements_gdf.columns or settlements_gdf['cluster_id'].isna().all():
        print("\n⚠ Warning: No cluster_id found in settlements. Cannot match facilities.")
        return gpd.GeoDataFrame(crs=COMMON_CRS)
    
    # Calculate cluster properties
    cluster_summary = []
    for cluster_id in sorted(settlements_gdf['cluster_id'].unique()):
        if cluster_id < 0:
            continue
        
        cluster_settlements = settlements_gdf[settlements_gdf['cluster_id'] == cluster_id]
        total_demand = cluster_settlements['demand_gap_mwh'].sum()
        
        # Weighted center
        weights = cluster_settlements['demand_gap_mwh'].values
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        center_lon = np.average(cluster_settlements.geometry.x, weights=weights)
        center_lat = np.average(cluster_settlements.geometry.y, weights=weights)
        
        geo_component = cluster_settlements['geo_component'].mode()[0] if 'geo_component' in cluster_settlements.columns else None
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'num_settlements': len(cluster_settlements),
            'total_demand_mwh': total_demand,
            'center_lon': center_lon,
            'center_lat': center_lat,
            'geo_component': geo_component,
            'geometry': Point(center_lon, center_lat),
            'facilities_assigned': [],  # Will store list of (facility_type, capacity) tuples
            'remaining_demand': total_demand
        })
    
    clusters_df = pd.DataFrame(cluster_summary)
    
    print(f"\n  Found {len(clusters_df)} demand-weighted clusters")
    
    # Sort facilities by component (match component-specific facilities first) and capacity (smallest first)
    if 'geo_component' in facilities_with_capacity.columns:
        facilities_sorted = facilities_with_capacity.sort_values(
            by=['geo_component', 'remaining_mwh'],
            ascending=[True, True]
        )
    else:
        facilities_sorted = facilities_with_capacity.sort_values('remaining_mwh', ascending=True)

    # Assign facilities to clusters (prioritize clusters with no assignments, split large facilities if needed)
    for fac_idx, facility in facilities_sorted.iterrows():
        facility_type = facility['Grouped_Type']
        facility_capacity = facility['remaining_mwh']
        facility_component = facility.get('geo_component', None)

        remaining_capacity = facility_capacity

        while remaining_capacity > 0:
            # Filter clusters by component if facility is component-specific
            available_clusters = clusters_df
            if facility_component is not None and facility_component >= 0:
                available_clusters = available_clusters[available_clusters['geo_component'] == facility_component]

            available_clusters = available_clusters[available_clusters['remaining_demand'] > 0].copy()

            if len(available_clusters) == 0:
                break

            # Prefer clusters with no facilities; then highest remaining demand
            available_clusters['n_assigned'] = available_clusters['facilities_assigned'].apply(len)
            available_clusters = available_clusters.sort_values(
                by=['n_assigned', 'remaining_demand'],
                ascending=[True, False]
            )

            best_cluster_idx = available_clusters.index[0]
            best_cluster = clusters_df.loc[best_cluster_idx]

            allocation = min(remaining_capacity, best_cluster['remaining_demand'])
            if allocation <= 0:
                break

            # Add facility (or portion of it) to cluster
            clusters_df.at[best_cluster_idx, 'facilities_assigned'].append((facility_type, allocation, fac_idx))

            # Reduce remaining demand for cluster and capacity for facility
            new_remaining = max(0, best_cluster['remaining_demand'] - allocation)
            clusters_df.at[best_cluster_idx, 'remaining_demand'] = new_remaining
            remaining_capacity -= allocation

            print(f"  Assigned {facility_type} ({allocation:,.0f} MWh) → Cluster {best_cluster['cluster_id']} (cluster remaining: {new_remaining:,.0f} MWh; facility remaining: {remaining_capacity:,.0f} MWh)")

        if remaining_capacity > 0:
            print(f"  Note: {facility_type} facility {fac_idx} has {remaining_capacity:,.0f} MWh unassigned (no remaining demand within component filter)")
    
    # Create output clusters (one row per facility assignment)
    cluster_output = []
    output_cluster_id = 0
    
    for idx, cluster in clusters_df.iterrows():
        if len(cluster['facilities_assigned']) == 0:
            # Cluster with no facilities assigned
            print(f"\n⚠ Warning: Cluster {cluster['cluster_id']} has no facilities assigned ({cluster['total_demand_mwh']:,.0f} MWh demand)")
            continue
        
        for facility_type, facility_capacity, fac_idx in cluster['facilities_assigned']:
            cluster_output.append({
                'cluster_id': output_cluster_id,
                'original_cluster_id': cluster['cluster_id'],
                'geometry': cluster['geometry'],
                'num_settlements': cluster['num_settlements'],
                'demand_gap_mwh': cluster['total_demand_mwh'],  # Total cluster demand
                'Grouped_Type': facility_type,
                'remaining_mwh': facility_capacity,
                'center_lon': cluster['center_lon'],
                'center_lat': cluster['center_lat'],
                'matched_facility_id': fac_idx,
                'geo_component': cluster['geo_component']
            })
            output_cluster_id += 1
    
    if len(cluster_output) == 0:
        print("\n⚠ Warning: No facilities were matched to any clusters")
        return gpd.GeoDataFrame(crs=COMMON_CRS)
    
    cluster_centers_gdf = gpd.GeoDataFrame(cluster_output, crs=COMMON_CRS)
    
    print(f"\nCluster-facility matching results:")
    print(f"  Total facility assignments: {len(cluster_centers_gdf)}")
    print(f"  Unique clusters with facilities: {cluster_centers_gdf['original_cluster_id'].nunique()}")
    print(f"  Total cluster demand: {clusters_df['total_demand_mwh'].sum():,.2f} MWh")
    print(f"  Total capacity allocated: {cluster_centers_gdf['remaining_mwh'].sum():,.2f} MWh")
    
    print(f"\nFacilities assigned by energy type:")
    for energy_type in sorted(cluster_centers_gdf['Grouped_Type'].unique()):
        type_assignments = cluster_centers_gdf[cluster_centers_gdf['Grouped_Type'] == energy_type]
        type_capacity = type_assignments['remaining_mwh'].sum()
        print(f"  {energy_type}: {len(type_assignments)} assignments, capacity: {type_capacity:,.2f} MWh")
    
    # Show which clusters got multiple facilities
    multi_facility_clusters = cluster_centers_gdf.groupby('original_cluster_id').size()
    multi_facility_clusters = multi_facility_clusters[multi_facility_clusters > 1]
    if len(multi_facility_clusters) > 0:
        print(f"\nClusters with multiple facilities:")
        for orig_cluster_id, n_facilities in multi_facility_clusters.items():
            cluster_facilities = cluster_centers_gdf[cluster_centers_gdf['original_cluster_id'] == orig_cluster_id]
            facility_types = cluster_facilities['Grouped_Type'].tolist()
            total_capacity = cluster_facilities['remaining_mwh'].sum()
            print(f"  Cluster {orig_cluster_id}: {n_facilities} facilities ({', '.join(facility_types)}) = {total_capacity:,.0f} MWh")
    
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


def build_remote_networks(settlements_gdf, cluster_centers_gdf, grid_lines_gdf):
    """Build connections from remote cluster centers to nearest grid points.
    
    Only creates facility-to-grid connections for remote clusters (>50km from grid).
    Intra-cluster connections will be created naturally by process_country_supply.py
    during the supply allocation process (web-looking gridlines from facility to settlements).
    """
    print("\n" + "="*60)
    print("STEP 4: BUILD NETWORK CONNECTIONS FOR REMOTE CLUSTERS")
    print("="*60)
    
    remote_clusters = cluster_centers_gdf[cluster_centers_gdf['is_remote']]
    
    if len(remote_clusters) == 0:
        print("No remote clusters found!")
        return gpd.GeoDataFrame(columns=['geometry', 'cluster_id', 'distance_km', 
                                        'from_lon', 'from_lat', 'to_lon', 'to_lat'], crs=COMMON_CRS)
    
    print(f"\nCreating facility-to-grid connections for {len(remote_clusters)} remote clusters...")
    print("(Intra-cluster connections will be created during supply allocation)")
    
    all_connections = []
    for idx, cluster in remote_clusters.iterrows():
        cluster_center = Point(cluster['center_lon'], cluster['center_lat'])
        nearest_grid = Point(cluster['nearest_grid_lon'], cluster['nearest_grid_lat'])
        distance_km = cluster['distance_to_grid_km']
        
        connection = {
            'geometry': LineString([cluster_center, nearest_grid]),
            'cluster_id': cluster['cluster_id'],
            'distance_km': distance_km,
            'from_lon': cluster['center_lon'],
            'from_lat': cluster['center_lat'],
            'to_lon': cluster['nearest_grid_lon'],
            'to_lat': cluster['nearest_grid_lat'],
            'connection_type': 'facility_to_grid'
        }
        all_connections.append(connection)
        
        print(f"  Cluster {cluster['cluster_id']}: {distance_km:.2f} km to grid")
    
    if not all_connections:
        print("No network connections generated!")
        return gpd.GeoDataFrame(columns=['geometry', 'cluster_id', 'distance_km', 
                                        'from_lon', 'from_lat', 'to_lon', 'to_lat'], crs=COMMON_CRS)
    
    networks_gdf = gpd.GeoDataFrame(all_connections, crs=COMMON_CRS)
    
    print(f"\nTotal facility-to-grid connections: {len(networks_gdf)}")
    print(f"Total connection length: {networks_gdf['distance_km'].sum():.2f} km")
    print(f"Average connection distance: {networks_gdf['distance_km'].mean():.2f} km")
    
    return networks_gdf


def clip_networks_to_boundaries(networks_gdf, admin_boundaries):
    """Clip network edges to country boundaries to prevent extending beyond borders."""
    if networks_gdf.empty:
        return networks_gdf
    
    print("\nClipping networks to country boundaries...")
    
    # Ensure both are in same CRS
    admin_union = admin_boundaries.to_crs(COMMON_CRS).geometry.union_all()
    
    # Clip networks to boundaries
    clipped_networks = gpd.clip(networks_gdf, admin_boundaries)
    
    # Recalculate distances for clipped edges
    def get_utm_crs(lon, lat):
        """Get appropriate UTM CRS for given coordinates."""
        utm_zone = int((lon + 180) / 6) + 1
        return f"EPSG:{32600 + utm_zone}" if lat >= 0 else f"EPSG:{32700 + utm_zone}"
    
    if not clipped_networks.empty:
        for idx in clipped_networks.index:
            geom = clipped_networks.loc[idx, 'geometry']
            if geom is not None and not geom.is_empty:
                center = geom.centroid
                utm_crs = get_utm_crs(center.x, center.y)
                try:
                    geom_utm = gpd.GeoSeries([geom], crs=COMMON_CRS).to_crs(utm_crs).iloc[0]
                    clipped_networks.loc[idx, 'distance_km'] = geom_utm.length / 1000.0
                except Exception:
                    pass  # Keep original distance if conversion fails
    
    edges_removed = len(networks_gdf) - len(clipped_networks)
    if edges_removed > 0:
        print(f"  Removed {edges_removed} edges extending beyond boundaries")
    
    print(f"  Final network edges: {len(clipped_networks)}")
    print(f"  Final network length: {clipped_networks['distance_km'].sum():.2f} km")
    
    return clipped_networks


def clip_clusters_to_boundaries(cluster_centers_gdf, admin_boundaries):
    """Ensure cluster centers are within country boundaries."""
    if cluster_centers_gdf.empty:
        return cluster_centers_gdf
    
    print("\nValidating cluster centers within country boundaries...")
    
    # Ensure both are in same CRS
    admin_union = admin_boundaries.to_crs(COMMON_CRS).geometry.union_all()
    
    # Check which clusters are outside boundaries
    within_boundary = cluster_centers_gdf.geometry.within(admin_union)
    outside_count = (~within_boundary).sum()
    
    if outside_count > 0:
        print(f"  Warning: {outside_count} clusters found outside boundaries")
        
        # For clusters outside, find nearest point on boundary
        from shapely.ops import nearest_points
        
        for idx in cluster_centers_gdf[~within_boundary].index:
            cluster_point = cluster_centers_gdf.loc[idx, 'geometry']
            # Find nearest point on boundary
            nearest_pt = nearest_points(cluster_point, admin_union.boundary)[1]
            
            # Update geometry and coordinates
            cluster_centers_gdf.loc[idx, 'geometry'] = nearest_pt
            cluster_centers_gdf.loc[idx, 'center_lon'] = nearest_pt.x
            cluster_centers_gdf.loc[idx, 'center_lat'] = nearest_pt.y
        
        print(f"  Moved {outside_count} clusters to nearest boundary point")
    else:
        print(f"  All {len(cluster_centers_gdf)} clusters within boundaries ✓")
    
    return cluster_centers_gdf


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
        admin_boundaries = load_admin_boundaries(country_iso3)
        centroids_gdf = load_centroids(country_iso3, scenario_suffix, output_dir)
        grid_lines_gdf = load_grid_lines(country_iso3, scenario_suffix, output_dir)
        facilities_gdf = load_facilities(country_iso3, scenario_suffix, output_dir)
        
        settlements_gdf = filter_settlements(centroids_gdf)
        
        if len(settlements_gdf) == 0:
            print("\nNo settlements to process!")
            return None
        
        # Calculate clusters based on remaining capacity (capacity-driven approach)
        # Pass all centroids to correctly calculate total supply received (including filled settlements)
        # This now returns facilities_with_capacity which may include synthetic facilities
        n_clusters, clusters_per_type, total_supply, total_demand, facilities_with_capacity = calculate_num_clusters(
            facilities_gdf, settlements_gdf, all_centroids_gdf=centroids_gdf
        )
        
        # Calculate remaining capacity by energy type for allocation (already done in calculate_num_clusters)
        remaining_by_type = facilities_with_capacity.groupby('Grouped_Type')['remaining_mwh'].sum().to_dict()
        
        # Cluster settlements using weighted K-means (weighted by demand_gap_mwh)
        settlements_gdf = cluster_settlements(settlements_gdf, n_clusters)
        
        # Match each cluster to a specific facility considering demand gap + energy type + remaining capacity
        # Use facilities_with_capacity which now includes synthetic facilities for missing types
        cluster_centers_gdf = calculate_cluster_centers(settlements_gdf, facilities_with_capacity, clusters_per_type, remaining_by_type)
        
        # Ensure clusters are within country boundaries
        cluster_centers_gdf = clip_clusters_to_boundaries(cluster_centers_gdf, admin_boundaries)
        
        cluster_centers_gdf = compute_grid_distances(cluster_centers_gdf, grid_lines_gdf)
        networks_gdf = build_remote_networks(settlements_gdf, cluster_centers_gdf, grid_lines_gdf)
        
        # Clip networks to country boundaries
        networks_gdf = clip_networks_to_boundaries(networks_gdf, admin_boundaries)
        
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
