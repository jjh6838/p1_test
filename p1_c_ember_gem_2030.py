import pandas as pd
import warnings
from pycountry import countries

# Suppress openpyxl warning
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Define GEM types as primary reference with corresponding Ember mappings
gem_to_ember_dict = {
    "Fossil": "Fossil",
    "Other Renewables": "Other Renewables",
    "hydropower": "Hydro",
    "nuclear": "Nuclear",
    "solar": "Solar",
    "wind": "Wind",
}

# 1. Load the proj_mw.csv file
proj_mw_path = r"outputs_processed_data\p1_a_proj_mw.csv"
proj_mw_df = pd.read_csv(proj_mw_path)

# Extract the required columns for South Korea
required_columns = [
    'ISO3_code', 'Fossil_2023', 'Nuclear_2023', 'Hydro_2030', 
    'Solar_2030', 'Wind_2030', 'Other Renewables_2030'
]
proj_mw_df = proj_mw_df[proj_mw_df['ISO3_code'] == 'KOR'][required_columns]

# Create variables for each energy type for South Korea
larger_value_fossil = proj_mw_df['Fossil_2023'].values[0]
larger_value_nuclear = proj_mw_df['Nuclear_2023'].values[0]
larger_value_hydro = proj_mw_df['Hydro_2030'].values[0]
larger_value_solar = proj_mw_df['Solar_2030'].values[0]
larger_value_wind = proj_mw_df['Wind_2030'].values[0]
larger_value_other_renewables = proj_mw_df['Other Renewables_2030'].values[0]

# 2. GEM data processing: Total capacity and capacity per type (for 2030)
energy_facilities_path = r"re_data\Global-Integrated-Power-April-2025.xlsx"
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")

# Convert 'Retired year' to numeric, coercing errors to NaN

# Manual mapping for non-standardized country names
manual_mapping = {
    "Bolivia": "BOL",
    "Bonaire, Sint Eustatius, and Saba": "BES",
    "Brunei": "BRN",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Holy See": "VAT",
    "Iran": "IRN",
    "Kosovo": "XKX",
    "Laos": "LAO",
    "Micronesia": "FSM",
    "Moldova": "MDA",
    "North Korea": "PRK",
    "Palestine": "PSE",
    "Republic of the Congo": "COG",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "The Gambia": "GMB",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Virgin Islands (U.S.)": "VIR"
}

def map_country_to_iso3(country_name):
    """
    Map a country name to its ISO3 code using pycountry or manual mapping.
    """
    if countries:
        try:
            return countries.lookup(country_name).alpha_3
        except LookupError:
            return manual_mapping.get(country_name, "unknown")
    else:
        return manual_mapping.get(country_name, "unknown")

# Add ISO3 column based on 'Country/area' values
energy_facilities_df['ISO3'] = energy_facilities_df['Country/area'].apply(map_country_to_iso3)

# Filter rows with valid ISO3 codes
energy_facilities_df = energy_facilities_df[energy_facilities_df['ISO3'] != "unknown"]


gem_data['Retired year'] = pd.to_numeric(gem_data['Retired year'], errors='coerce')
gem_data['Start year'] = pd.to_numeric(gem_data['Start year'], errors='coerce')

# Filter for operating status and relevant years
gem_data = gem_data[
    (gem_data['Status'] == 'operating') &
    ((gem_data['Retired year'].isna()) | (gem_data['Retired year'] > 2030)) |
    (gem_data['Status'].isin(['announced', 'pre-construction', 'construction']) & (gem_data['Start year'] <= 2030))
]

].copy()  # Add .copy() to avoid SettingWithCopyWarning

# Modify 'Type' column based on conditions
filtered_gem_df.loc[:, 'Type'] = filtered_gem_df['Type'].replace({
    'bioenergy': 'Other Renewables',
    'geothermal': 'Other Renewables',
    'coal': 'Fossil',
    'oil/gas': 'Fossil'
})

gem_capacity = filtered_gem_df.groupby('Type')['Capacity (MW)'].sum()
total_gem_capacity = gem_capacity.sum()

# Compare capacities and find larger values and differences
capacity_differences = {}
larger_capacities = {}

for gem_type in gem_to_ember_dict.keys():
    gem_value = gem_capacity.get(gem_type, 0)
    ember_value = proj_mw_df[f"{gem_to_ember_dict[gem_type]}_2023"].values[0] if gem_type in ['Fossil', 'Nuclear'] else proj_mw_df[f"{gem_to_ember_dict[gem_type]}_2030"].values[0]
    difference = ember_value - gem_value  # Positive means Ember is larger
    larger_value = max(gem_value, ember_value)
    capacity_differences[gem_type] = difference
    larger_capacities[gem_type] = larger_value

# Export larger capacity values for other scripts
larger_value_fossil = larger_capacities.get('Fossil', 0)
larger_value_nuclear = larger_capacities.get('nuclear', 0)
larger_value_hydro = larger_capacities.get('hydropower', 0)
larger_value_solar = larger_capacities.get('solar', 0)
larger_value_wind = larger_capacities.get('wind', 0)
larger_value_other_renewables = larger_capacities.get('Other Renewables', 0)

if __name__ == '__main__':
    print("\n=== Capacity Comparison ===")
    print(f"GEM - Total Capacity in KOR (MW): {total_gem_capacity:,.0f}")
    
    print("\n=== Detailed Comparison by Type ===")
    for gem_type in gem_to_ember_dict.keys():
        gem_value = gem_capacity.get(gem_type, 0)
        ember_value = proj_mw_df[f"{gem_to_ember_dict[gem_type]}_2023"].values[0] if gem_type in ['Fossil', 'Nuclear'] else proj_mw_df[f"{gem_to_ember_dict[gem_type]}_2030"].values[0]
        difference = capacity_differences[gem_type]
        larger_value = larger_capacities[gem_type]
        
        print(f"\n{gem_type}:")
        print(f"  GEM Capacity (MW): {gem_value:,.0f}")
        print(f"  Ember Capacity (MW): {ember_value:,.0f}")
        print(f"  Difference (Ember - GEM) (MW): {difference:,.0f}")
        print(f"  Larger Capacity (MW): {larger_value:,.0f}")

    # Print the values for verification
    print(f"Fossil_2023: {larger_value_fossil}")
    print(f"Nuclear_2023: {larger_value_nuclear}")
    print(f"Hydro_2030: {larger_value_hydro}")
    print(f"Solar_2030: {larger_value_solar}")
    print(f"Wind_2030: {larger_value_wind}")
    print(f"Other Renewables_2030: {larger_value_other_renewables}")

