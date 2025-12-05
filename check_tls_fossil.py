import geopandas as gpd
import pandas as pd
import numpy as np

# Load data
cents = gpd.read_parquet('outputs_per_country/parquet/2030_supply_100%/centroids_TLS.parquet')
facs = gpd.read_parquet('outputs_per_country/parquet/2030_supply_100%/facilities_TLS.parquet')

print("=" * 60)
print("TLS FOSSIL FACILITY ANALYSIS")
print("=" * 60)

# Filter Fossil facilities
fossil_facs = facs[facs['Grouped_Type'] == 'Fossil']
print(f"\nFossil facilities: {len(fossil_facs)}")
for idx, fac in fossil_facs.iterrows():
    print(f"  ID {fac['GEM unit/phase ID']}: {fac['total_mwh']:,.2f} MWh total, {fac['remaining_mwh']:,.2f} MWh remaining")
    print(f"    Location: ({fac['Latitude']:.4f}, {fac['Longitude']:.4f})")

# Check unfilled settlements
unfilled = cents[cents['supply_status'].isin(['Not Filled', 'Partially Filled'])]
print(f"\nUnfilled settlements: {len(unfilled)}")
demand_col = 'Total_Demand_2030_centroid'
unfilled_demand = unfilled[demand_col] - unfilled['supply_received_mwh']
print(f"Total unfilled demand: {unfilled_demand.sum():,.2f} MWh")

# Calculate distances from unfilled settlements to Fossil facilities
if len(unfilled) > 0 and len(fossil_facs) > 0:
    print(f"\nCalculating distances from unfilled settlements to Fossil facilities...")
    
    for fac_idx, fac in fossil_facs.iterrows():
        fac_geom = fac.geometry
        
        distances = unfilled.geometry.distance(fac_geom) * 111  # Convert degrees to km (rough)
        
        within_250km = (distances <= 250).sum()
        
        print(f"\n  Fossil facility {fac['GEM unit/phase ID']}:")
        print(f"    Unfilled settlements within 250km: {within_250km}")
        print(f"    Min distance: {distances.min():.2f} km")
        print(f"    Max distance: {distances.max():.2f} km")
        print(f"    Mean distance: {distances.mean():.2f} km")
        
        # Show closest unfilled settlements
        closest_unfilled = unfilled.copy()
        closest_unfilled['dist_km'] = distances
        closest_unfilled['demand_gap'] = closest_unfilled[demand_col] - closest_unfilled['supply_received_mwh']
        closest_unfilled = closest_unfilled.sort_values('dist_km').head(5)
        
        print(f"\n    Closest 5 unfilled settlements:")
        for idx, row in closest_unfilled.iterrows():
            print(f"      {row['dist_km']:.2f} km: {row['demand_gap']:,.2f} MWh demand, pop={row['Population_2030_centroid']:.0f}, status={row['supply_status']}")

# Check what's in the facilities file for capacity
print(f"\n\nFacility capacity details:")
print(f"Total Fossil capacity (total_mwh): {fossil_facs['total_mwh'].sum():,.2f} MWh")
print(f"Total Fossil supplied: {fossil_facs['supplied_mwh'].sum():,.2f} MWh")
print(f"Total Fossil remaining: {fossil_facs['remaining_mwh'].sum():,.2f} MWh")
