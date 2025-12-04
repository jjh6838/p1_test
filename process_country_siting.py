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
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

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


def identify_geographic_components(settlements_gdf, max_distance_km=100):
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
    admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410-levels.gpkg', layer="ADM_0")
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


def calculate_num_clusters(facilities_gdf, settlements_gdf):
    """Calculate number of clusters based on remaining capacity by energy type (capacity-driven approach)."""
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
    
    # Calculate supply received by energy type from settlements
    supply_received_by_type = {}
    for energy_type in DEMAND_TYPES:
        type_supply_col = f"{energy_type}_{ANALYSIS_YEAR}_centroid"
        if type_supply_col in settlements_gdf.columns:
            # Sum up what was actually delivered to all settlements for this type
            supply_received_by_type[energy_type] = settlements_gdf[type_supply_col].sum()
        else:
            supply_received_by_type[energy_type] = 0
    
    total_supply_received = sum(supply_received_by_type.values())
    
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
    if demand_col in settlements_gdf.columns:
        total_demand = settlements_gdf[demand_col].sum()
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
    settlements_gdf_temp = identify_geographic_components(settlements_gdf.copy(), max_distance_km=100)
    
    # Calculate demand per component
    component_demand_gap = settlements_gdf_temp.groupby('geo_component')['demand_gap_mwh'].sum().to_dict()
    
    # Filter out small components (< 5 settlements) and get viable components
    component_sizes = settlements_gdf_temp.groupby('geo_component').size()
    viable_components = component_sizes[component_sizes >= 5].index.tolist()
    
    print(f"\nCreating synthetic facilities per geographic component:")
    print(f"  Viable components (≥5 settlements): {len(viable_components)}")
    
    synthetic_facilities = []
    excess_capacity_by_type = {}  # Track excess capacity that needs reallocation
    
    for energy_type, total_shortfall in shortfall_by_type.items():
        if total_shortfall <= 0:
            continue
        
        print(f"\n  {energy_type}: {total_shortfall:,.2f} MWh total shortfall")
        
        # Calculate expected capacity per component based on energy mix
        component_allocations = {}
        total_allocated = 0
        
        for comp_id in viable_components:
            if comp_id < 0:  # Skip noise
                continue
            
            comp_demand_gap = component_demand_gap.get(comp_id, 0)
            comp_settlements = settlements_gdf_temp[settlements_gdf_temp['geo_component'] == comp_id]
            
            # Expected capacity for this energy type in this component (from demand gap, not total demand)
            expected_capacity = comp_demand_gap * expected_mix[energy_type]
            
            if expected_capacity > 0:
                component_allocations[comp_id] = {
                    'capacity': expected_capacity,
                    'settlements': comp_settlements
                }
                total_allocated += expected_capacity
        
        # Check if we have excess capacity that wasn't allocated (use tolerance for floating point)
        excess = total_shortfall - total_allocated
        
        if excess > 0.01:  # Use small tolerance for floating point comparison
            print(f"    Excess capacity: {excess:,.2f} MWh (total shortfall exceeds component demands)")
            excess_capacity_by_type[energy_type] = excess
            
            # Try to reallocate excess to components with same-type existing facilities
            components_with_type = []
            for comp_id in viable_components:
                if comp_id < 0:
                    continue
                comp_settlements = settlements_gdf_temp[settlements_gdf_temp['geo_component'] == comp_id]
                # Check if any settlement in this component already receives this energy type
                type_col = f"{energy_type}_{ANALYSIS_YEAR}_centroid"
                if type_col in comp_settlements.columns:
                    if comp_settlements[type_col].sum() > 0:
                        components_with_type.append(comp_id)
            
            if len(components_with_type) > 0:
                # Reallocate excess proportionally to components with this type
                print(f"    Reallocating to {len(components_with_type)} components with existing {energy_type} supply")
                excess_per_component = excess / len(components_with_type)
                
                for comp_id in components_with_type:
                    if comp_id in component_allocations:
                        component_allocations[comp_id]['capacity'] += excess_per_component
                    else:
                        comp_settlements = settlements_gdf_temp[settlements_gdf_temp['geo_component'] == comp_id]
                        component_allocations[comp_id] = {
                            'capacity': excess_per_component,
                            'settlements': comp_settlements
                        }
                excess = 0  # All excess reallocated
            
            # If still excess (no components with this type), create additional facility
            if excess > 0:
                print(f"    Creating additional facility for remaining excess: {excess:,.2f} MWh")
                # Place at overall demand-weighted centroid
                settlements_with_demand = settlements_gdf_temp[settlements_gdf_temp['demand_gap_mwh'] > 0].copy()
                
                if len(settlements_with_demand) > 0:
                    weights = settlements_with_demand['demand_gap_mwh'].values
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(len(weights)) / len(weights)
                    
                    optimal_lon = np.average(settlements_with_demand.geometry.x, weights=weights)
                    optimal_lat = np.average(settlements_with_demand.geometry.y, weights=weights)
                    optimal_location = Point(optimal_lon, optimal_lat)
                    
                    synthetic_facility = {
                        'gem_id': f'SYNTHETIC_{energy_type}_EXCESS',
                        'Grouped_Type': energy_type,
                        'remaining_mwh': excess,
                        'total_mwh': excess,
                        'available_total_mwh': excess,
                        'center_lat': optimal_location.y,
                        'center_lon': optimal_location.x,
                        'geometry': optimal_location,
                        'geo_component': -1  # Not tied to specific component
                    }
                    synthetic_facilities.append(synthetic_facility)
        
        # Create synthetic facilities for each component allocation
        for comp_id, allocation in component_allocations.items():
            capacity = allocation['capacity']
            comp_settlements = allocation['settlements']
            
            if capacity <= 0:
                continue
            
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
                'remaining_mwh': capacity,
                'total_mwh': capacity,
                'available_total_mwh': capacity,
                'center_lat': optimal_location.y,
                'center_lon': optimal_location.x,
                'geometry': optimal_location,
                'geo_component': comp_id
            }
            synthetic_facilities.append(synthetic_facility)
            
            print(f"    Component {comp_id}: {capacity:,.2f} MWh at ({optimal_location.y:.4f}, {optimal_location.x:.4f})")
    
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
    
    # Identify separate geographic components (islands, separated territories)
    settlements_gdf = identify_geographic_components(settlements_gdf, max_distance_km=100)
    
    n_components = len(set(settlements_gdf['geo_component'])) - (1 if -1 in settlements_gdf['geo_component'].values else 0)
    
    print(f"\nClustering parameters:")
    print(f"  Algorithm: Weighted K-means (within geographic components)")
    print(f"  Weight metric: demand_gap_mwh")
    print(f"  Total settlements: {len(settlements_gdf)}")
    print(f"  Geographic components: {n_components}")
    print(f"  Desired total clusters: {n_clusters}")
    
    # Allocate clusters proportionally to each component based on demand
    component_demand = settlements_gdf.groupby('geo_component')['demand_gap_mwh'].sum()
    total_demand = component_demand.sum()
    
    # Check if we have enough clusters for all components (minimum 1 per component)
    # Filter out small components (< 5 settlements) - assume off-grid solutions
    component_sizes = settlements_gdf.groupby('geo_component').size()
    viable_components = component_sizes[component_sizes >= 5].index.tolist()
    
    n_viable_components = len(viable_components)
    
    if n_viable_components < n_components:
        small_components = component_sizes[component_sizes < 5]
        print(f"\n  Excluding {len(small_components)} small components (< 5 settlements each) - assumed off-grid:")
        for comp_id, size in small_components.items():
            if comp_id >= 0:
                comp_demand = component_demand[comp_id]
                print(f"    Component {comp_id}: {size} settlements ({comp_demand:,.2f} MWh) → off-grid")
        n_components = n_viable_components
    
    if n_components > n_clusters:
        print(f"\n⚠ Warning: More viable components ({n_components}) than available clusters ({n_clusters})")
        print(f"  Some components will not receive dedicated clusters")
        print(f"  Allocating clusters to components with highest demand")
        
        # Sort viable components by demand (highest first)
        viable_component_demand = component_demand[viable_components]
        sorted_components = viable_component_demand.sort_values(ascending=False).head(n_clusters)
        
        clusters_per_component = {}
        for comp_id in sorted_components.index:
            if comp_id >= 0:
                clusters_per_component[comp_id] = 1
        
        print(f"\n  Clusters allocated to top {len(clusters_per_component)} viable components by demand")
    else:
        # Normal allocation: at least 1 cluster per viable component
        clusters_per_component = {}
        for comp_id in viable_components:
            if comp_id >= 0:  # Skip noise points
                comp_settlements = settlements_gdf[settlements_gdf['geo_component'] == comp_id]
                n_settlements_in_comp = len(comp_settlements)
                
                # Allocate clusters proportional to demand, but at least 1 per component
                proportion = component_demand[comp_id] / total_demand
                allocated_clusters = max(1, int(n_clusters * proportion))
                
                # Don't exceed number of settlements in component
                allocated_clusters = min(allocated_clusters, n_settlements_in_comp)
                
                clusters_per_component[comp_id] = allocated_clusters
        
        # Adjust if total doesn't match target (due to rounding)
        total_allocated = sum(clusters_per_component.values())
        if total_allocated != n_clusters:
            # Add/remove from largest component
            largest_comp = component_demand.idxmax()
            adjustment = n_clusters - total_allocated
            clusters_per_component[largest_comp] += adjustment
            clusters_per_component[largest_comp] = max(1, clusters_per_component[largest_comp])
    
    print(f"\nCluster allocation per geographic component:")
    for comp_id, n_clust in sorted(clusters_per_component.items()):
        comp_settlements = settlements_gdf[settlements_gdf['geo_component'] == comp_id]
        comp_demand = comp_settlements['demand_gap_mwh'].sum()
        print(f"  Component {comp_id}: {n_clust} clusters for {len(comp_settlements)} settlements ({comp_demand:,.2f} MWh)")
    
    # Perform K-means clustering within each component
    settlements_gdf['cluster_id'] = -1
    global_cluster_id = 0
    
    for comp_id, n_clust in clusters_per_component.items():
        comp_mask = settlements_gdf['geo_component'] == comp_id
        comp_settlements = settlements_gdf[comp_mask].copy()
        
        if len(comp_settlements) == 0:
            continue
        
        coords = np.column_stack([
            comp_settlements.geometry.x,
            comp_settlements.geometry.y
        ])
        
        # Use demand_gap_mwh as sample weights for clustering
        weights = comp_settlements['demand_gap_mwh'].values
        if weights.sum() > 0:
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(len(weights)) / len(weights)  # Equal weights if no demand gap
        
        if n_clust == 1 or len(comp_settlements) == 1:
            # Single cluster for this component
            settlements_gdf.loc[comp_mask, 'cluster_id'] = global_cluster_id
            global_cluster_id += 1
        else:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
            local_labels = kmeans.fit_predict(coords, sample_weight=weights)
            
            # Map local labels to global cluster IDs
            settlements_gdf.loc[comp_mask, 'cluster_id'] = local_labels + global_cluster_id
            global_cluster_id += n_clust
    
    # Handle settlements in components that didn't receive clusters
    # This includes: (1) small components (<5 settlements), (2) components without allocated clusters
    unassigned_mask = settlements_gdf['cluster_id'] == -1
    n_unassigned = unassigned_mask.sum()
    if n_unassigned > 0:
        unassigned_settlements = settlements_gdf[unassigned_mask].copy()
        assigned_settlements = settlements_gdf[~unassigned_mask & (settlements_gdf['cluster_id'] >= 0)]
        
        # Separate small components (off-grid) from unallocated components
        small_component_mask = unassigned_settlements['geo_component'].map(
            lambda c: component_sizes.get(c, 0) < 5
        )
        
        n_small_component = small_component_mask.sum()
        n_unallocated = (~small_component_mask).sum()
        
        if n_small_component > 0:
            print(f"\n  {n_small_component} settlements in small components (< 5 settlements) → off-grid (no cluster assigned)")
        
        if n_unallocated > 0:
            print(f"\n⚠ {n_unallocated} settlements in components without allocated clusters")
            print(f"  Assigning to nearest cluster within 100 km threshold")
            
            if len(assigned_settlements) > 0:
                # Get cluster centroids
                cluster_centroids = assigned_settlements.groupby('cluster_id').apply(
                    lambda x: Point(x.geometry.x.mean(), x.geometry.y.mean())
                )
                
                assigned_count = 0
                for idx, settlement in unassigned_settlements[~small_component_mask].iterrows():
                    # Find nearest cluster centroid within 100km
                    min_dist_km = float('inf')
                    nearest_cluster = -1
                    
                    for cluster_id, centroid in cluster_centroids.items():
                        # Calculate haversine distance
                        dist_km = haversine_distance_km(
                            settlement.geometry.y, settlement.geometry.x,
                            centroid.y, centroid.x
                        )
                        
                        if dist_km < min_dist_km and dist_km <= 100:  # Within 100km threshold
                            min_dist_km = dist_km
                            nearest_cluster = cluster_id
                    
                    if nearest_cluster >= 0:
                        settlements_gdf.loc[idx, 'cluster_id'] = nearest_cluster
                        assigned_count += 1
                    else:
                        # No cluster within 100km - remains unassigned (off-grid)
                        pass
                
                print(f"    Assigned {assigned_count} settlements to nearest clusters")
                remaining_unassigned = n_unallocated - assigned_count
                if remaining_unassigned > 0:
                    print(f"    {remaining_unassigned} settlements beyond 100km threshold → off-grid (no cluster)")
    
    print(f"\nClustering results:")
    cluster_sizes = settlements_gdf.groupby('cluster_id').size().sort_values(ascending=False)
    print(f"  Total clusters created: {len(cluster_sizes)}")
    print(f"  Largest cluster: {cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Smallest cluster: {cluster_sizes.iloc[-1] if len(cluster_sizes) > 0 else 0} settlements")
    print(f"  Average cluster size: {cluster_sizes.mean():.1f} settlements")
    print(f"  Median cluster size: {cluster_sizes.median():.1f} settlements")
    
    # Report total demand gap per cluster
    cluster_demand = settlements_gdf.groupby('cluster_id')['demand_gap_mwh'].sum().sort_values(ascending=False)
    print(f"\nDemand gap by cluster:")
    print(f"  Highest demand cluster: {cluster_demand.iloc[0]:,.2f} MWh")
    print(f"  Lowest demand cluster: {cluster_demand.iloc[-1]:,.2f} MWh")
    print(f"  Average per cluster: {cluster_demand.mean():.2f} MWh")
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
        # This now returns facilities_with_capacity which may include synthetic facilities
        n_clusters, clusters_per_type, total_supply, total_demand, facilities_with_capacity = calculate_num_clusters(facilities_gdf, settlements_gdf)
        
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
        networks_gdf = build_remote_networks(settlements_gdf, cluster_centers_gdf)
        
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
