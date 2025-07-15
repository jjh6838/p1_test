# 6/30/2025 at work

import pandas as pd
import warnings
try:
    from pycountry import countries
except ImportError:
    countries = None

# Suppress openpyxl warning
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Define granular and grouped energy categories
granular_categories = [
    "Bioenergy", "Coal", "Hydro", "Nuclear", "Solar", "Wind", 
    "Other Renewables", "Gas", "Other Fossil"
]

grouped_categories = {
    "Fossil": ["Coal", "Gas", "Other Fossil"],
    "Other Renewables": ["Bioenergy", "Other Renewables"],
    "Hydro": ["Hydro"],
    "Nuclear": ["Nuclear"],
    "Solar": ["Solar"],
    "Wind": ["Wind"]
}

# GEM to Ember category mapping
gem_to_ember_mapping = {
    "bioenergy": "Bioenergy",
    "coal": "Coal",
    "hydropower": "Hydro",
    "nuclear": "Nuclear",
    "solar": "Solar",
    "wind": "Wind",
    "geothermal": "Other Renewables",
    "oil/gas": ["Gas", "Other Fossil"]  # Split proportionally
}

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
    Map a country name to its ISO3 code using manual mapping first, then pycountry.
    """
    # Check manual mapping first
    if country_name in manual_mapping:
        return manual_mapping[country_name]
    
    # If not found in manual mapping, try pycountry
    if countries:
        try:
            return countries.lookup(country_name).alpha_3
        except LookupError:
            return "unknown"
    else:
        return "unknown"

def map_iso3_to_country_name(iso3_code):
    """
    Map an ISO3 code to a country name using manual mapping first, then pycountry.
    """
    # Create reverse mapping from manual_mapping (prioritize these names)
    reverse_manual_mapping = {v: k for k, v in manual_mapping.items()}
    
    # Check reverse manual mapping first
    if iso3_code in reverse_manual_mapping:
        return reverse_manual_mapping[iso3_code]
    
    # If not found in manual mapping, try pycountry
    if countries:
        try:
            return countries.lookup(iso3_code).name
        except LookupError:
            return iso3_code  # Return ISO3 code if no name found
    else:
        return iso3_code  # Return ISO3 code if no name found

# Load Ember data
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format2025-05-02.csv" #updated on May 2, 2025
energy_data = pd.read_csv(energy_data_csv_path)

# Separate filtering for Capacity and Electricity generation
years = [2023, 2022, 2021, 2020, 2019] # 2024 data is not fully available yet...


# Filter for Capacity data
capacity_data = energy_data[
    (energy_data["Year"].isin(years)) &
    (energy_data["Category"] == "Capacity") &
    (energy_data["Subcategory"] == "Fuel") &
    (energy_data["Unit"] == "GW") &
    (energy_data["Variable"].isin(granular_categories))
]

# Filter for Electricity generation data
generation_data = energy_data[
    (energy_data["Year"].isin(years)) &
    (energy_data["Category"] == "Electricity generation") &
    (energy_data["Subcategory"] == "Fuel") &
    (energy_data["Unit"] == "TWh") &
    (energy_data["Variable"].isin(granular_categories))
]

# Prioritize the most recent year for each country and variable in both datasets
capacity_data = capacity_data.sort_values(by="Year", ascending=False).drop_duplicates(subset=["Country code", "Variable"], keep="first")
generation_data = generation_data.sort_values(by="Year", ascending=False).drop_duplicates(subset=["Country code", "Variable"], keep="first")

# Load GEM data
energy_facilities_path = r"re_data\Global-Integrated-Power-April-2025.xlsx" # updated on May 2, 2025
gem_data = pd.read_excel(energy_facilities_path, sheet_name="Power facilities")

# Filter GEM data for operating facilities

# Add ISO3 codes to GEM data
gem_data['ISO3'] = gem_data['Country/area'].apply(map_country_to_iso3)

# Filter GEM data to exclude rows with unknown ISO3 codes
gem_data = gem_data[gem_data['ISO3'] != "unknown"]

# Get all unique country codes from Ember data
ember_country_codes = set(energy_data['Country code'].dropna().unique())

# Get all unique ISO3 codes from GEM data
gem_iso3_codes = set(gem_data['ISO3'].unique())

# Keep only the matching ISO3 codes between Ember and GEM data
matching_country_codes = ember_country_codes.intersection(gem_iso3_codes)

# Check for missing countries
missing_in_gem = ember_country_codes - gem_iso3_codes
missing_in_ember = gem_iso3_codes - ember_country_codes

print("Countries in Ember but not in GEM:", sorted(missing_in_gem))
print("Countries in GEM but not in Ember:", sorted(missing_in_ember))

# Check total counts
print("Total countries in GEM:", len(gem_iso3_codes))
print("Total countries in Ember:", len(ember_country_codes))
print("Matching countries:", len(matching_country_codes))

# List all matching countries
print("Matching countries available in both Ember and GEM datasets:", sorted(matching_country_codes))

# Initialize result DataFrames
granular_df = pd.DataFrame()
grouped_df = pd.DataFrame()

# Calculate global conversion rates for fallback
# Calculate global mean conversion rates by averaging local rates across countries
global_conversion_rates = {}

# Loop through each granular energy category to calculate its global conversion rate
for category in granular_categories:
    # Collect all local rates for the current category across countries
    local_rates = []
    for country_code in matching_country_codes:
        filtered_generation_data = generation_data[generation_data['Country code'] == country_code]
        filtered_capacity_data = capacity_data[capacity_data['Country code'] == country_code]
        
        generation = filtered_generation_data.groupby('Variable')['Value'].sum().get(category, 0) * 1000000  # Convert TWh to MWh
        capacity = filtered_capacity_data.groupby('Variable')['Value'].sum().get(category, 0) * 1000  # Convert GW to MW
        
        if capacity > 0:
            local_rate = generation / capacity
            local_rates.append(local_rate)
    
    # Calculate the global mean conversion rate as the average of local rates
    global_conversion_rates[category] = sum(local_rates) / len(local_rates) if local_rates else 0

# Process data for each matching country
for country_code in matching_country_codes:
    gem_data = gem_data[gem_data['Status'] == 'operating']  # for the current year (i.e., 2024)

    # Filter Ember generation data for the current country
    filtered_generation_data = generation_data[generation_data['Country code'] == country_code]
    if filtered_generation_data.empty:
        print(f"No Ember generation data found for country {country_code}.")
        continue

    # Calculate ember_generation
    ember_generation = filtered_generation_data.groupby('Variable')['Value'].sum() * 1000000  # Convert TWh to MWh

    # Filter Ember capacity data for the current country
    filtered_capacity_data = capacity_data[capacity_data['Country code'] == country_code]
    ember_capacity = filtered_capacity_data.groupby('Variable')['Value'].sum() * 1000  # Convert GW to MW


    # Filter GEM data for the current country (already in MW)
    gem_country_data = gem_data[gem_data['ISO3'] == country_code]
    gem_capacity = gem_country_data.groupby('Type')['Capacity (MW)'].sum()

    # Use Ember capacity data if GEM capacity data is missing or zero
    if gem_capacity.empty or gem_capacity.sum() == 0:
        print(f"No GEM capacity data found for country {country_code}.")
        

    # Calculate granular conversion rates for Ember data
    granular_conversion_rates = {}
    for category in granular_categories:
        generation = ember_generation.get(category, 0)
        capacity = ember_capacity.get(category, 0)
        # Use global rate if local rate is zero or missing
        if capacity > 0:
            rate = generation / capacity
        else:
            rate = 0
        if not rate:
            rate = global_conversion_rates.get(category, 0)
        granular_conversion_rates[category] = rate

    # Initialize granular data for the current country
    granular_data = {'Country Code': country_code, 'Country Name': map_iso3_to_country_name(country_code)}

    # Process each GEM type and its corresponding Ember type
    for gem_type, ember_type in gem_to_ember_mapping.items():
        gem_cap = gem_capacity.get(gem_type, 0)
        if isinstance(ember_type, list):  # Handle "oil/gas" split into "Gas" and "Other Fossil"
            ember_gas = ember_capacity.get("Gas", 0)
            ember_other_fossil = ember_capacity.get("Other Fossil", 0)
            ember_gas_and_other_fossil = ember_gas + ember_other_fossil

            # Determine proportions
            if ember_gas_and_other_fossil > 0:
                # Use Ember data to calculate proportions
                gas_ratio = ember_gas / ember_gas_and_other_fossil
                other_fossil_ratio = ember_other_fossil / ember_gas_and_other_fossil
            else:
                # No data in Ember, fallback to 5:5 split
                gas_ratio = 0.5
                other_fossil_ratio = 0.5

            if gem_cap > ember_gas_and_other_fossil:
                # GEM is larger: Split GEM's oil/gas proportionally
                gas_gem_cap = gem_cap * gas_ratio
                other_fossil_gem_cap = gem_cap * other_fossil_ratio

                granular_data["Gas_GEM_MW"] = gas_gem_cap
                granular_data["Gas_Ember_MW"] = ember_gas
                granular_data["Gas_Larger_MW"] = max(gas_gem_cap, ember_gas)
                granular_data["Gas_ConvRate"] = granular_conversion_rates.get("Gas", 0)
                granular_data["Gas_Potential_MWh"] = gas_gem_cap * granular_conversion_rates.get("Gas", 0)

                granular_data["Other Fossil_GEM_MW"] = other_fossil_gem_cap
                granular_data["Other Fossil_Ember_MW"] = ember_other_fossil
                granular_data["Other Fossil_Larger_MW"] = max(other_fossil_gem_cap, ember_other_fossil)
                granular_data["Other Fossil_ConvRate"] = granular_conversion_rates.get("Other Fossil", 0)
                granular_data["Other Fossil_Potential_MWh"] = other_fossil_gem_cap * granular_conversion_rates.get("Other Fossil", 0)
            else:
                # Ember is larger: Use Ember's granular data directly
                granular_data["Gas_GEM_MW"] = 0  # GEM does not contribute to Gas
                granular_data["Gas_Ember_MW"] = ember_gas
                granular_data["Gas_Larger_MW"] = ember_gas
                granular_data["Gas_ConvRate"] = granular_conversion_rates.get("Gas", 0)
                granular_data["Gas_Potential_MWh"] = ember_generation.get("Gas", 0)  # Use Ember's MWh directly

                granular_data["Other Fossil_GEM_MW"] = 0  # GEM does not contribute to Other Fossil
                granular_data["Other Fossil_Ember_MW"] = ember_other_fossil
                granular_data["Other Fossil_Larger_MW"] = ember_other_fossil
                granular_data["Other Fossil_ConvRate"] = granular_conversion_rates.get("Other Fossil", 0)
                granular_data["Other Fossil_Potential_MWh"] = ember_generation.get("Other Fossil", 0)  # Use Ember's MWh directly
        else:
            ember_cap = ember_capacity.get(ember_type, 0)
            larger_cap = ember_cap if gem_cap == 0 or pd.isna(gem_cap) else gem_cap if ember_cap == 0 or pd.isna(ember_cap) else max(ember_cap, gem_cap)

            granular_data[f'{ember_type}_GEM_MW'] = gem_cap
            granular_data[f'{ember_type}_Ember_MW'] = ember_cap
            granular_data[f'{ember_type}_Larger_MW'] = larger_cap
            granular_data[f'{ember_type}_ConvRate'] = granular_conversion_rates.get(ember_type, 0)

            if gem_cap > ember_cap:
                # GEM is larger: Calculate MWh using conversion rate
                granular_data[f'{ember_type}_Potential_MWh'] = larger_cap * granular_conversion_rates.get(ember_type, 0)
            else:
                # Ember is larger: Use Ember's MWh directly
                granular_data[f'{ember_type}_Potential_MWh'] = ember_generation.get(ember_type, 0)

    granular_df = pd.concat([granular_df, pd.DataFrame([granular_data])], ignore_index=True)

    # Calculate grouped conversion rates, capacity, and potential generation
    grouped_data = {'Country Code': country_code, 'Country Name': map_iso3_to_country_name(country_code)}
    for group, subcategories in grouped_categories.items():
        total_capacity = sum(granular_data.get(f'{sub}_Larger_MW', 0) for sub in subcategories)
        weighted_sum = sum(
            granular_data.get(f'{sub}_Larger_MW', 0) * granular_conversion_rates.get(sub, 0) for sub in subcategories
        )
        grouped_conversion_rate = weighted_sum / total_capacity if total_capacity > 0 else 0
        grouped_data[f'{group}_Larger_MW'] = total_capacity
        grouped_data[f'{group}_ConvRate'] = grouped_conversion_rate
        grouped_data[f'{group}_Potential_MWh'] = total_capacity * grouped_conversion_rate

    grouped_df = pd.concat([grouped_df, pd.DataFrame([grouped_data])], ignore_index=True)

# Add "Country/area (GEM)" from GEM and "Area (Ember)" from Ember to the matching countries
gem_country_mapping = gem_data[['ISO3', 'Country/area']].drop_duplicates()
ember_country_mapping = energy_data[['Country code', 'Area']].drop_duplicates()

# Rename columns for clarity
gem_country_mapping.rename(columns={'Country/area': 'Country/area (GEM)'}, inplace=True)
ember_country_mapping.rename(columns={'Area': 'Area (Ember)'}, inplace=True)

# Merge GEM and Ember country mappings independently
granular_df = pd.merge(
    granular_df,
    gem_country_mapping[['ISO3', 'Country/area (GEM)']],
    left_on='Country Code',
    right_on='ISO3',
    how='left'
)

granular_df = pd.merge(
    granular_df,
    ember_country_mapping[['Country code', 'Area (Ember)']],
    left_on='Country Code',
    right_on='Country code',
    how='left'
)

# Drop redundant columns after merging
granular_df.drop(columns=['ISO3', 'Country code'], inplace=True)

# Add "total_MWh" column to the granular DataFrame
granular_df["Total_Potential_MWh"] = granular_df.filter(like="_Potential_MWh").sum(axis=1)

# Add "total_MWh" column to the grouped DataFrame
grouped_df["Total_Potential_MWh"] = grouped_df.filter(like="_Potential_MWh").sum(axis=1)

# Reorder columns to have Country Name first
granular_columns = ['Country Name'] + [col for col in granular_df.columns if col != 'Country Name']
granular_df = granular_df[granular_columns]

grouped_columns = ['Country Name'] + [col for col in grouped_df.columns if col != 'Country Name']
grouped_df = grouped_df[grouped_columns]

### I want to generate another sheet, Grouped_cur_facilities, which contains facility-level data by country (Latitude, Longtitude, Watt adjusted proportionally)
# Create a new DataFrame for facility-level data
grouped_facilities_df = gem_data[['ISO3', 'Country/area', 'Type', 'Capacity (MW)', 'Latitude', 'Longitude', 'GEM location ID']].copy()
# Add ISO3 codes and standardize country names
grouped_facilities_df['Country Code'] = grouped_facilities_df['Country/area'].apply(map_country_to_iso3)
grouped_facilities_df['Country Name'] = grouped_facilities_df['Country Code'].apply(map_iso3_to_country_name)

# Filter out rows with unknown ISO3 codes
grouped_facilities_df = grouped_facilities_df[grouped_facilities_df['Country Code'] != "unknown"]

# Filter to only include countries that are in the matching_country_codes
grouped_facilities_df = grouped_facilities_df[grouped_facilities_df['Country Code'].isin(matching_country_codes)]

# Map GEM types to grouped categories
gem_to_grouped_mapping = {
    "bioenergy": "Other Renewables",
    "coal": "Fossil",
    "hydropower": "Hydro",
    "nuclear": "Nuclear",
    "solar": "Solar",
    "wind": "Wind",
    "geothermal": "Other Renewables",
    "oil/gas": "Fossil"
}

grouped_facilities_df['Grouped_Type'] = grouped_facilities_df['Type'].map(gem_to_grouped_mapping)

# Remove rows where Type cannot be mapped to a grouped category
grouped_facilities_df = grouped_facilities_df.dropna(subset=['Grouped_Type'])

# Calculate current capacity totals by country and grouped type
current_totals = grouped_facilities_df.groupby(['Country Code', 'Grouped_Type'])['Capacity (MW)'].sum().reset_index()
current_totals.rename(columns={'Capacity (MW)': 'Current_Total_MW'}, inplace=True)

# Get target totals from grouped_df
target_totals = []
for _, row in grouped_df.iterrows():
    country_code = row['Country Code']
    for group in grouped_categories.keys():
        target_mw = row.get(f'{group}_Larger_MW', 0)
        if target_mw > 0:
            target_totals.append({
                'Country Code': country_code,
                'Grouped_Type': group,
                'Target_Total_MW': target_mw
            })

target_totals_df = pd.DataFrame(target_totals)

# Merge current and target totals
totals_comparison = pd.merge(current_totals, target_totals_df, on=['Country Code', 'Grouped_Type'], how='outer')
totals_comparison['Current_Total_MW'] = totals_comparison['Current_Total_MW'].fillna(0)
totals_comparison['Target_Total_MW'] = totals_comparison['Target_Total_MW'].fillna(0)

# Calculate scaling factors
totals_comparison['Scaling_Factor'] = totals_comparison.apply(
    lambda row: row['Target_Total_MW'] / row['Current_Total_MW'] if row['Current_Total_MW'] > 0 else 1,
    axis=1
)

# Merge scaling factors back to facilities dataframe
grouped_facilities_df = pd.merge(
    grouped_facilities_df,
    totals_comparison[['Country Code', 'Grouped_Type', 'Scaling_Factor']],
    on=['Country Code', 'Grouped_Type'],
    how='left'
)

# Apply scaling factors to adjust capacity
grouped_facilities_df['Adjusted_Capacity_MW'] = grouped_facilities_df['Capacity (MW)'] * grouped_facilities_df['Scaling_Factor']

# Select and reorder final columns
grouped_facilities_df = grouped_facilities_df[['Country Name', 'Country Code', 'Type', 'Grouped_Type', 
                                               'Capacity (MW)', 'Adjusted_Capacity_MW', 'Latitude', 
                                               'Longitude', 'GEM location ID']]

# Save the results to an Excel file with two sheets
output_path = r"outputs_processed_data\p1_b_ember_gem_2024.xlsx"

with pd.ExcelWriter(output_path) as writer:
    granular_df.to_excel(writer, sheet_name='Granular_cur', index=False)
    grouped_df.to_excel(writer, sheet_name='Grouped_cur', index=False)
    grouped_facilities_df.to_excel(writer, sheet_name='Grouped_cur_facilities', index=False)

print(f"Analysis saved to {output_path} with three sheets: 'Granular_cur', 'Grouped_cur', and 'Grouped_cur_facilities'")

