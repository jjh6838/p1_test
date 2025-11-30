# 6/30/2025 at work

# ==================================================================================================
# SCRIPT PURPOSE: Global Power Plant Data Integration and Analysis
# ==================================================================================================
#
# WHAT THIS CODE DOES:
# This script harmonizes two major global energy datasets:
# 1. Ember: Provides country-level aggregate data on electricity generation (TWh) and capacity (GW).
# 2. Global Energy Monitor (GEM): Provides granular, facility-level data including location (lat/lon).
#
# WHY THIS IS NEEDED:
# - Ember data is authoritative for country-level statistics but lacks spatial resolution (where plants are).
# - GEM data provides precise locations but may not match official country-level aggregates perfectly.
# - By combining them, we create a dataset that has both:
#   a) Accurate aggregate capacity/generation numbers (taking the maximum of Ember vs GEM to ensure coverage).
#   b) Spatial distribution of power plants for mapping and regional analysis.
#
# KEY OUTPUTS:
# 1. Country-level summaries of capacity and potential generation by fuel type.
# 2. Facility-level datasets where individual power plant capacities are adjusted (scaled) to match
#    the harmonized country totals. This allows for spatial analysis that sums up to the correct totals.
# 3. Future projections (2030, 2050) considering facility retirements.
# ==================================================================================================

import pandas as pd
import warnings
from pycountry import countries

# Suppress openpyxl warning
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Define granular and grouped energy categories
# Granular categories represent specific fuel types used in the analysis
granular_categories = [
    "Bioenergy", "Coal", "Hydro", "Nuclear", "Solar", "Wind", 
    "Other Renewables", "Gas", "Other Fossil"
]

# Grouped categories aggregate granular types into broader classes
grouped_categories = {
    "Fossil": ["Coal", "Gas", "Other Fossil"],
    "Other Renewables": ["Bioenergy", "Other Renewables"],
    "Hydro": ["Hydro"],
    "Nuclear": ["Nuclear"],
    "Solar": ["Solar"],
    "Wind": ["Wind"]
}

# GEM to Ember category mapping
# Maps GEM's specific fuel types to Ember's categories for consistency
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
# Handles discrepancies between dataset country names and standard ISO codes
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
    Includes fuzzy matching for better results with non-standard names.
    """
    # Check manual mapping first
    if country_name in manual_mapping:
        return manual_mapping[country_name]
    
    # If not found in manual mapping, try pycountry
    if countries:
        try:
            # First, try a direct lookup
            return countries.lookup(country_name).alpha_3
        except LookupError:
            try:
                # If direct lookup fails, try a fuzzy search
                return countries.search_fuzzy(country_name)[0].alpha_3
            except LookupError:
                # If both fail, return "unknown"
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
# Reads the Ember dataset containing yearly energy statistics
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format2025-11-23.csv" 
# Updated on November 23, 2025 (Column name has been changed from 'Country code (old name)' to 'ISO 3 code (new name)' in the downloaded file, so I renamed it back from ISO 3 code' to 'Country code' for consistency)
# https://ember-energy.org/data/yearly-electricity-data/
energy_data = pd.read_csv(energy_data_csv_path)

# Separate filtering for Capacity and Electricity generation
# We use data from 2019-2024 as 2024 data is incomplete
years = [2024, 2023, 2022, 2021, 2020, 2019] # 2025 data is not fully available yet


# Filter for Capacity data (GW)
capacity_data = energy_data[
    (energy_data["Year"].isin(years)) &
    (energy_data["Category"] == "Capacity") &
    (energy_data["Subcategory"] == "Fuel") &
    (energy_data["Unit"] == "GW") &
    (energy_data["Variable"].isin(granular_categories))
]

# Filter for Electricity generation data (TWh)
generation_data = energy_data[
    (energy_data["Year"].isin(years)) &
    (energy_data["Category"] == "Electricity generation") &
    (energy_data["Subcategory"] == "Fuel") &
    (energy_data["Unit"] == "TWh") &
    (energy_data["Variable"].isin(granular_categories))
]

# Sort by year descending to prioritize most recent data
capacity_data = capacity_data.sort_values(by="Year", ascending=False)
generation_data = generation_data.sort_values(by="Year", ascending=False)

# For each country-variable pair, find the most recent year where BOTH capacity and generation exist
aligned_capacity = []
aligned_generation = []

for country_code in set(capacity_data['Country code'].unique()) | set(generation_data['Country code'].unique()):
    for variable in granular_categories:
        # Get all available years for this country-variable combination
        cap_rows = capacity_data[(capacity_data['Country code'] == country_code) & 
                                 (capacity_data['Variable'] == variable)]
        gen_rows = generation_data[(generation_data['Country code'] == country_code) & 
                                   (generation_data['Variable'] == variable)]
        
        # Find the most recent year where both exist
        matched = False
        for year in years:  # years is already sorted from newest to oldest
            cap_year_data = cap_rows[cap_rows['Year'] == year]
            gen_year_data = gen_rows[gen_rows['Year'] == year]
            
            if not cap_year_data.empty and not gen_year_data.empty:
                aligned_capacity.append(cap_year_data.iloc[0])
                aligned_generation.append(gen_year_data.iloc[0])
                matched = True
                break
        
        # If no matching year found, use the most recent available for each
        if not matched:
            if not cap_rows.empty:
                aligned_capacity.append(cap_rows.iloc[0])
            if not gen_rows.empty:
                aligned_generation.append(gen_rows.iloc[0])

# Convert back to DataFrames
capacity_data = pd.DataFrame(aligned_capacity)
generation_data = pd.DataFrame(aligned_generation)

# Load GEM data
# Reads the Global Energy Monitor dataset containing facility-level data
energy_facilities_path = r"re_data\Global-Integrated-Power-September-2025-II.xlsx" # updated on November 23, 2025
# read the source info and download for update: https://globalenergymonitor.org/about/who-uses-gem-data/

gem_data = pd.read_excel(energy_facilities_path, sheet_name="Power facilities")

# Filter GEM data for operating facilities
gem_data = gem_data[gem_data['Status'] == 'operating']

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
# This provides a default capacity factor if local data is missing.
# WHY: We need to convert Capacity (MW) to Generation (MWh). If a country has capacity but no reported generation
# for a fuel type, we use the global average performance of that technology to estimate generation.
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

# Calculate global grouped conversion rates
global_grouped_conversion_rates = {}
for group, subcategories in grouped_categories.items():
    # Calculate global capacity for this group across all countries
    total_global_capacity = 0
    weighted_sum = 0
    
    for category in subcategories:
        category_capacity = capacity_data[capacity_data['Variable'] == category]['Value'].sum() * 1000  # Convert GW to MW
        total_global_capacity += category_capacity
        weighted_sum += global_conversion_rates.get(category, 0) * category_capacity
    
    global_grouped_conversion_rates[group] = weighted_sum / total_global_capacity if total_global_capacity > 0 else 0

# Process data for each matching country
# Main loop to harmonize Ember and GEM data for each country
for country_code in matching_country_codes:
    # Filter GEM data for the current country and operating status
    gem_country_data = gem_data[(gem_data['ISO3'] == country_code) & (gem_data['Status'] == 'operating')]

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
            # If calculated rate is still 0, use global rate as fallback
            if rate == 0:
                rate = global_conversion_rates.get(category, 0)
        else:
            # No capacity available, use global rate
            rate = global_conversion_rates.get(category, 0)
        granular_conversion_rates[category] = rate

    # Initialize granular data for the current country
    granular_data = {'Country Code': country_code, 'Country Name': map_iso3_to_country_name(country_code)}

    # Process each GEM type and its corresponding Ember type
    # Compares capacity from both sources and takes the larger value
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
            # Determine the larger capacity between GEM and Ember
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
    # Aggregates granular data into broader categories (e.g., Solar, Wind, Fossil)
    grouped_data = {'Country Code': country_code, 'Country Name': map_iso3_to_country_name(country_code)}
    for group, subcategories in grouped_categories.items():
        total_capacity = sum(granular_data.get(f'{sub}_Larger_MW', 0) for sub in subcategories)
        weighted_sum = sum(
            granular_data.get(f'{sub}_Larger_MW', 0) * granular_conversion_rates.get(sub, 0) for sub in subcategories
        )
        grouped_conversion_rate = weighted_sum / total_capacity if total_capacity > 0 else 0
        
        # If grouped conversion rate is 0, use global grouped rate as fallback
        if grouped_conversion_rate == 0:
            grouped_conversion_rate = global_grouped_conversion_rates.get(group, 0)
            
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

### I want to generate another file with three sheets which contain facility-level data by country (Latitude, Longtitude, Watt adjusted proportionally)
# Create a new DataFrame for facility-level data
# This section prepares facility-level data for mapping and further analysis
grouped_facilities_df = gem_data[['ISO3', 'Country/area', 'Type', 'Capacity (MW)', 'Latitude', 'Longitude', 'GEM unit/phase ID']].copy()
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
        target_mwh = row.get(f'{group}_Potential_MWh', 0)
        if target_mw > 0:
            target_totals.append({
                'Country Code': country_code,
                'Grouped_Type': group,
                'Target_Total_MW': target_mw,
                'Target_Total_MWh': target_mwh
            })

target_totals_df = pd.DataFrame(target_totals)

# Merge current and target totals
totals_comparison = pd.merge(current_totals, target_totals_df, on=['Country Code', 'Grouped_Type'], how='outer')
totals_comparison['Current_Total_MW'] = totals_comparison['Current_Total_MW'].fillna(0)
totals_comparison['Target_Total_MW'] = totals_comparison['Target_Total_MW'].fillna(0)
totals_comparison['Target_Total_MWh'] = totals_comparison['Target_Total_MWh'].fillna(0)

# Calculate scaling factors
# Adjusts facility capacities to match the "Larger MW" calculated at the country level
# WHY: The facility-level data (GEM) might sum up to less (or more) than the official country total (Ember/Max).
# We scale the individual facilities so that their sum equals the authoritative total, preserving the spatial distribution
# while correcting the magnitude.
totals_comparison['Scaling_Factor'] = totals_comparison.apply(
    lambda row: row['Target_Total_MW'] / row['Current_Total_MW'] if row['Current_Total_MW'] > 0 else 1,
    axis=1
)

# Merge scaling factors back to facilities dataframe
grouped_facilities_df = pd.merge(
    grouped_facilities_df,
    totals_comparison[['Country Code', 'Grouped_Type', 'Scaling_Factor', 'Target_Total_MWh']],
    on=['Country Code', 'Grouped_Type'],
    how='left'
)

# Apply scaling factors to adjust capacity
grouped_facilities_df['Adjusted_Capacity_MW'] = grouped_facilities_df['Capacity (MW)'] * grouped_facilities_df['Scaling_Factor']

# Calculate total_mwh for each facility by distributing the target MWh proportionally
grouped_facilities_df['total_mwh'] = grouped_facilities_df.apply(
    lambda row: (row['Adjusted_Capacity_MW'] / 
                totals_comparison[
                    (totals_comparison['Country Code'] == row['Country Code']) & 
                    (totals_comparison['Grouped_Type'] == row['Grouped_Type'])
                ]['Target_Total_MW'].iloc[0] * row['Target_Total_MWh']) 
                if totals_comparison[
                    (totals_comparison['Country Code'] == row['Country Code']) & 
                    (totals_comparison['Grouped_Type'] == row['Grouped_Type'])
                ]['Target_Total_MW'].iloc[0] > 0 else 0,
    axis=1
)

# Select and reorder final columns
grouped_facilities_df = grouped_facilities_df[['Country Name', 'Country Code', 'Type', 'Grouped_Type', 
                                               'Capacity (MW)', 'Adjusted_Capacity_MW', 'total_mwh', 'Latitude', 
                                               'Longitude', 'GEM unit/phase ID']]

# Save the main results to an Excel file with two sheets
output_path = r"outputs_processed_data\p1_a_ember_gem_2024.xlsx"

with pd.ExcelWriter(output_path) as writer:
    granular_df.to_excel(writer, sheet_name='Granular_cur', index=False)
    grouped_df.to_excel(writer, sheet_name='Grouped_cur', index=False)

print(f"Main country-level analysis saved to {output_path} with two sheets: 'Granular_cur' and 'Grouped_cur'")

# Save the facility-level data to a separate Excel file
facilities_output_path = r"outputs_processed_data\p1_a_ember_gem_2024_fac_lvl.xlsx"

# For 2030 and 2050, filter out retired facilities
# First, identify facilities that should be excluded for each projection year
# This accounts for plant retirements in future scenarios

# Load the original GEM data again to access retirement year information
gem_data_full = pd.read_excel(energy_facilities_path, sheet_name="Power facilities")
gem_data_full['ISO3'] = gem_data_full['Country/area'].apply(map_country_to_iso3)
gem_data_full = gem_data_full[gem_data_full['ISO3'] != "unknown"]

# Convert retirement and start year columns to numeric
gem_data_full['Retired year'] = pd.to_numeric(gem_data_full['Retired year'], errors='coerce')
gem_data_full['Start year'] = pd.to_numeric(gem_data_full['Start year'], errors='coerce')

# Find GEM unit/phase IDs for facilities that should be EXCLUDED for 2030 projections
# (facilities that retire before or by 2030)
excluded_2030_ids = gem_data_full[
    (gem_data_full['Status'] == 'operating') &
    (gem_data_full['Retired year'].notna()) &
    (gem_data_full['Retired year'] <= 2030)
]['GEM unit/phase ID'].tolist()

# Find GEM unit/phase IDs for facilities that should be EXCLUDED for 2050 projections  
# (facilities that retire before or by 2050)
excluded_2050_ids = gem_data_full[
    (gem_data_full['Status'] == 'operating') &
    (gem_data_full['Retired year'].notna()) &
    (gem_data_full['Retired year'] <= 2050)
]['GEM unit/phase ID'].tolist()

print(f"Facilities to exclude for 2030 projection: {len(excluded_2030_ids)}")
print(f"Facilities to exclude for 2050 projection: {len(excluded_2050_ids)}")

# Create 2030 facility dataframe by removing excluded facilities
grouped_facilities_df_2030 = grouped_facilities_df[
    ~grouped_facilities_df['GEM unit/phase ID'].isin(excluded_2030_ids)
].copy()

# Create 2050 facility dataframe by removing excluded facilities
grouped_facilities_df_2050 = grouped_facilities_df[
    ~grouped_facilities_df['GEM unit/phase ID'].isin(excluded_2050_ids)
].copy()

print(f"Facilities in 2024 baseline: {len(grouped_facilities_df)}")
print(f"Facilities remaining in 2030: {len(grouped_facilities_df_2030)}")
print(f"Facilities remaining in 2050: {len(grouped_facilities_df_2050)}")

# Function to merge facilities within spatial grid cells (300 arcsec ~= 10km resolution)
def merge_facilities_by_location(df):
    """
    Merge facilities that have the same Grouped_Type and fall within the same 300 arcsecond grid cell.
    300 arcseconds is approximately 9-10km at the equator, providing spatial clustering of nearby facilities.
    Sum the Adjusted_Capacity_MW and use the centroid location of the cluster.
    This reduces the number of points for mapping/plotting while spatially clustering nearby facilities.
    """
    # Define grid resolution (300 arcseconds = 1/12 degree = 0.0833 degrees ~= 9.26km at equator)
    GRID_SIZE_DEG = 300 / 3600  # 300 arcseconds = 0.0833... degrees
    
    # Create grid cell identifiers by rounding coordinates to nearest grid cell
    df = df.copy()
    df['grid_lat'] = (df['Latitude'] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG
    df['grid_lon'] = (df['Longitude'] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG
    
    def merge_group(group):
        first_row = group.iloc[0].copy()
        # Use centroid of actual facility locations within the grid cell
        first_row['Latitude'] = group['Latitude'].mean()
        first_row['Longitude'] = group['Longitude'].mean()
        first_row['Adjusted_Capacity_MW'] = group['Adjusted_Capacity_MW'].sum()
        first_row['Capacity (MW)'] = group['Capacity (MW)'].sum()
        first_row['total_mwh'] = group['total_mwh'].sum()
        first_row['Num_of_Merged_Units'] = len(group)
        # Keep only the first GEM ID for simplicity (merged facilities treated as single unit)
        first_row['GEM unit/phase ID'] = str(group['GEM unit/phase ID'].iloc[0])
        return first_row
    
    # Group by energy type and grid cell
    merged_df = df.groupby(['Grouped_Type', 'grid_lat', 'grid_lon'], as_index=False).apply(merge_group, include_groups=False)
    
    # Drop the temporary grid columns
    merged_df = merged_df.drop(columns=['grid_lat', 'grid_lon'])
    merged_df = merged_df.reset_index(drop=True)
    
    return merged_df

# Apply merging to all three facility dataframes
print("\nMerging facilities by location...")
print(f"Before merging - 2024: {len(grouped_facilities_df)} facilities")
grouped_facilities_df_merged = merge_facilities_by_location(grouped_facilities_df)
print(f"After merging - 2024: {len(grouped_facilities_df_merged)} facilities")

print(f"Before merging - 2030: {len(grouped_facilities_df_2030)} facilities")
grouped_facilities_df_2030_merged = merge_facilities_by_location(grouped_facilities_df_2030)
print(f"After merging - 2030: {len(grouped_facilities_df_2030_merged)} facilities")

print(f"Before merging - 2050: {len(grouped_facilities_df_2050)} facilities")
grouped_facilities_df_2050_merged = merge_facilities_by_location(grouped_facilities_df_2050)
print(f"After merging - 2050: {len(grouped_facilities_df_2050_merged)} facilities")

# Reorder columns to include the new Num_of_Merged_Units column
final_columns = ['Country Name', 'Country Code', 'Type', 'Grouped_Type', 
                'Capacity (MW)', 'Adjusted_Capacity_MW', 'total_mwh', 'Latitude', 
                'Longitude', 'GEM unit/phase ID', 'Num_of_Merged_Units']

grouped_facilities_df_merged = grouped_facilities_df_merged[final_columns]
grouped_facilities_df_2030_merged = grouped_facilities_df_2030_merged[final_columns]
grouped_facilities_df_2050_merged = grouped_facilities_df_2050_merged[final_columns]

with pd.ExcelWriter(facilities_output_path) as writer:
    grouped_facilities_df_merged.to_excel(writer, sheet_name='Grouped_cur_fac_lvl', index=False)
    grouped_facilities_df_2030_merged.to_excel(writer, sheet_name='Grouped_2030_fac_lvl', index=False)
    grouped_facilities_df_2050_merged.to_excel(writer, sheet_name='Grouped_2050_fac_lvl', index=False)

print(f"Facility-level data saved to {facilities_output_path} with sheets: 'Grouped_cur_fac_lvl' (2024), 'Grouped_2030_fac_lvl' (2030), and 'Grouped_2050_fac_lvl' (2050)")

