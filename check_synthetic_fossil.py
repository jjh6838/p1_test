import geopandas as gpd
import pandas as pd
import numpy as np

# Load ADD_V2 data
cents = gpd.read_parquet('outputs_per_country/parquet/2030_supply_100%_add_v2/centroids_TLS_add_v2.parquet')
facs = gpd.read_parquet('outputs_per_country/parquet/2030_supply_100%_add_v2/facilities_TLS_add_v2.parquet')

print("=" * 60)
print("TLS SYNTHETIC FOSSIL FACILITY ANALYSIS (ADD_V2)")
print("=" * 60)

# Find synthetic Fossil facility (lat = -8.64495, lon = 126.45211)
print("\nAll facilities:")
for idx, fac in facs.iterrows():
    gem_id = fac.get('GEM unit/phase ID', 'N/A')
    print(f"  fid {idx}: {fac['Grouped_Type']}, {fac['total_mwh']:,.0f} MWh, remaining: {fac['remaining_mwh']:,.0f} MWh")
    print(f"    Location: ({fac['Latitude']:.4f}, {fac['Longitude']:.4f}), GEM ID: {gem_id}")

# Find the synthetic Fossil facility
synthetic_fossil = facs[(facs['Grouped_Type'] == 'Fossil') & 
                        (abs(facs['Latitude'] - (-8.64495)) < 0.01) & 
                        (abs(facs['Longitude'] - 126.45211) < 0.01)]

if len(synthetic_fossil) == 0:
    print("\n⚠ Synthetic Fossil facility not found!")
else:
    print(f"\n{'='*60}")
    print("SYNTHETIC FOSSIL FACILITY (Siting cluster)")
    print(f"{'='*60}")
    fac = synthetic_fossil.iloc[0]
    print(f"  fid: {synthetic_fossil.index[0]}")
    print(f"  Type: {fac['Grouped_Type']}")
    print(f"  Total capacity: {fac['total_mwh']:,.2f} MWh")
    print(f"  Supplied: {fac['supplied_mwh']:,.2f} MWh")
    print(f"  Remaining: {fac['remaining_mwh']:,.2f} MWh")
    print(f"  Location: ({fac['Latitude']:.6f}, {fac['Longitude']:.6f})")
    
    # Check unfilled settlements
    demand_col = 'Total_Demand_2030_centroid'
    unfilled = cents[cents['supply_status'].isin(['Not Filled', 'Partially Filled'])]
    print(f"\nUnfilled settlements: {len(unfilled)}")
    unfilled_demand = unfilled[demand_col] - unfilled['supply_received_mwh']
    print(f"Total unfilled demand: {unfilled_demand.sum():,.2f} MWh")
    
    # Calculate distances
    fac_geom = fac.geometry
    distances = unfilled.geometry.distance(fac_geom) * 111  # Convert to km
    
    within_250km = (distances <= 250).sum()
    print(f"\nUnfilled settlements within 250km: {within_250km}")
    print(f"  Min distance: {distances.min():.2f} km")
    print(f"  Max distance: {distances.max():.2f} km")
    print(f"  Mean distance: {distances.mean():.2f} km")
    
    # Show closest unfilled
    closest = unfilled.copy()
    closest['dist_km'] = distances
    closest['demand_gap'] = closest[demand_col] - closest['supply_received_mwh']
    closest = closest.sort_values('dist_km').head(10)
    
    print(f"\nClosest 10 unfilled settlements:")
    for idx, row in closest.iterrows():
        print(f"  {row['dist_km']:.2f} km: {row['demand_gap']:,.2f} MWh gap, status={row['supply_status']}")
