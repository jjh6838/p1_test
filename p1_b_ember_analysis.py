import pandas as pd
import warnings

# Suppress openpyxl warning
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# File paths
energy_facilities_path = r"re_data\Global-Integrated-Power-June-2024.xlsx"
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format.csv"

# Define GEM types as primary reference with corresponding Ember mappings
gem_to_ember_dict = {
    "bioenergy": "Bioenergy",
    "coal": "Coal",
    "hydropower": "Hydro",
    "nuclear": "Nuclear",
    "solar": "Solar",
    "wind": "Wind",
    "geothermal": "Other Renewables",
    "oil/gas": ["Gas", "Other Fossil"]  # Combined categories in Ember
}

# 1. Ember data: MW, MWh, and Conversion rates (MWh/MW) by Type
def get_ember_data(data, category, unit):
    filtered_data = data[
        (data['Country code'] == 'KOR') &
        (data['Year'] == 2023) &
        (data['Category'] == category) &
        (data['Subcategory'] == 'Fuel') &
        (data['Unit'] == unit)
    ]
    
    # Get initial groupby result
    ember_values = filtered_data.groupby('Variable')['Value'].sum()
    
    # Create new dictionary with GEM types
    mapped_values = {}
    for gem_type, ember_type in gem_to_ember_dict.items():
        if isinstance(ember_type, list):
            # Sum values for combined categories (like oil/gas)
            mapped_values[gem_type] = sum(ember_values.get(t, 0) for t in ember_type)
        else:
            mapped_values[gem_type] = ember_values.get(ember_type, 0)
    
    return pd.Series(mapped_values)

# Calculate actual generation and conversion rates from Ember data
energy_data = pd.read_csv(energy_data_csv_path)
ember_generation = get_ember_data(energy_data, 'Electricity generation', 'TWh') * 1000000  # TWh to MWh
ember_capacity = get_ember_data(energy_data, 'Capacity', 'GW') * 1000  # GW to MW

# Calculate conversion rates based on Ember's actual generation and capacity
conversion_rates = {}
for gem_type in gem_to_ember_dict.keys():
    generation = ember_generation.get(gem_type, 0)  
    capacity = ember_capacity.get(gem_type, 0)      
    rate = generation / capacity if capacity > 0 else 0
    conversion_rates[gem_type] = rate

# 2. GEM data processing: Total capacity and capacity per type
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")
filtered_gem_df = energy_facilities_df[
    (energy_facilities_df['Country/area'] == 'South Korea') &
    (energy_facilities_df['Status'] == 'operating')
].copy()

gem_capacity = filtered_gem_df.groupby('Type')['Capacity (MW)'].sum()
total_gem_capacity = gem_capacity.sum()

# Process Ember data and map to GEM types
total_ember_capacity = ember_capacity.sum()

# Compare capacities and find larger values and differences
capacity_differences = {}
larger_capacities = {}

for gem_type in gem_to_ember_dict.keys():
    gem_value = gem_capacity.get(gem_type, 0)
    ember_value = ember_capacity.get(gem_type, 0)
    difference = ember_value - gem_value  # Positive means Ember is larger
    larger_value = max(gem_value, ember_value)
    capacity_differences[gem_type] = difference
    larger_capacities[gem_type] = larger_value

# Calculate potential generation using larger capacities and Ember conversion rates
potential_generation = {}
for fuel_type, capacity in larger_capacities.items():
    potential_mwh = capacity * conversion_rates.get(fuel_type, 0)
    potential_generation[fuel_type] = potential_mwh

# Export larger capacity values for other scripts
larger_value_bioenergy = larger_capacities.get('bioenergy', 0)
larger_value_coal = larger_capacities.get('coal', 0)
larger_value_hydro = larger_capacities.get('hydropower', 0)
larger_value_nuclear = larger_capacities.get('nuclear', 0)
larger_value_solar = larger_capacities.get('solar', 0)
larger_value_wind = larger_capacities.get('wind', 0)
larger_value_geothermal = larger_capacities.get('geothermal', 0)
larger_value_oilgas = larger_capacities.get('oil/gas', 0)
# Export conversion rates for other scripts
conversion_rate_bioenergy = conversion_rates.get('bioenergy', 0)
conversion_rate_coal = conversion_rates.get('coal', 0)
conversion_rate_hydro = conversion_rates.get('hydropower', 0)
conversion_rate_nuclear = conversion_rates.get('nuclear', 0)
conversion_rate_solar = conversion_rates.get('solar', 0)
conversion_rate_wind = conversion_rates.get('wind', 0)
conversion_rate_geothermal = conversion_rates.get('geothermal', 0)
conversion_rate_oilgas = conversion_rates.get('oil/gas', 0)

if __name__ == '__main__':
    print("\n=== Capacity Comparison ===")
    print(f"GEM - Total Capacity in KOR (MW): {total_gem_capacity:,.0f}")
    print(f"Ember - Total Capacity in KOR (MW): {total_ember_capacity:,.0f}")
    print(f"Total Difference (Ember - GEM) (MW): {total_ember_capacity - total_gem_capacity:,.0f}")
    
    print("\n=== Detailed Comparison by Type ===")
    for gem_type in gem_to_ember_dict.keys():
        gem_value = gem_capacity.get(gem_type, 0)
        ember_value = ember_capacity.get(gem_type, 0)
        difference = capacity_differences[gem_type]
        larger_value = larger_capacities[gem_type]
        
        print(f"\n{gem_type}:")
        print(f"  GEM Capacity (MW): {gem_value:,.0f}")
        print(f"  Ember Capacity (MW): {ember_value:,.0f}")
        print(f"  Difference (Ember - GEM) (MW): {difference:,.0f}")
        print(f"  Larger Capacity (MW): {larger_value:,.0f}")
        print(f"  Ember Actual Generation (MWh): {ember_generation.get(gem_type, 0):,.0f}")
        print(f"  Conversion Rate from Ember (MWh/MW): {conversion_rates.get(gem_type, 0):,.0f}")
        print(f"  Potential Generation using Larger Capacity (MWh): {potential_generation.get(gem_type, 0):,.0f}")

