import pandas as pd
import numpy as np

# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_b_ember_2024_30_50.xlsx')

# Generate random percentages for 2024, 2030, and 2050
np.random.seed(42)  # For reproducibility

# Define custom ranges for each energy type and year
ranges_2024 = {
    "exp_hydro_2024": (10, 15),
    "exp_solar_2024": (20, 30),
    "exp_wind_2024": (15, 25),
    "exp_other_renewables_2024": (1, 10),
    "exp_nuclear_2024": (1, 8),
    "exp_fossil_2024": (1, 8),
}
ranges_2030 = {
    "exp_hydro_2030": (12, 17),
    "exp_solar_2030": (25, 35),
    "exp_wind_2030": (25, 35),
    "exp_other_renewables_2030": (1, 15),
    "exp_nuclear_2030": (1, 8),
    "exp_fossil_2030": (1, 8),
}
ranges_2050 = {
    "exp_hydro_2050": (14, 19),
    "exp_solar_2050": (30, 45),
    "exp_wind_2050": (30, 40),
    "exp_other_renewables_2050": (1, 15),
    "exp_nuclear_2050": (1, 8),
    "exp_fossil_2050": (1, 8),
}

# Generate random percentages within the specified ranges
percentages_2024 = {col: np.random.uniform(*rng) for col, rng in ranges_2024.items()}
percentages_2030 = {col: np.random.uniform(*rng) for col, rng in ranges_2030.items()}
percentages_2050 = {col: np.random.uniform(*rng) for col, rng in ranges_2050.items()}

# Generate random percentages for risk-avoidance planning for 2024, 2030, and 2050
# For 2024, risk-avoidance values are exactly the same
percentages_2024_risk_avoid = {
    f'exp_risk_avoid_{k.split("_", 1)[1]}': v for k, v in percentages_2024.items()
}
# For 2030 and 2050, apply a reduction factor to the ranges
np.random.seed(123)  # Different seed for risk-avoidance scenario
ranges_2030_risk_avoid = {k.replace('exp_', 'exp_risk_avoid_'): (v[0] * 0.5, v[1] * 0.5) for k, v in ranges_2030.items()}
ranges_2050_risk_avoid = {k.replace('exp_', 'exp_risk_avoid_'): (v[0] * 0.3, v[1] * 0.3) for k, v in ranges_2050.items()}
percentages_2030_risk_avoid = {col: np.random.uniform(*rng) for col, rng in ranges_2030_risk_avoid.items()}
percentages_2050_risk_avoid = {col: np.random.uniform(*rng) for col, rng in ranges_2050_risk_avoid.items()}

# Map energy types to actual column patterns in your data
column_mapping = {
    "Hydro": "Hydro_",
    "Solar": "Solar_",
    "Wind": "Wind_",
    "Other_Renewables": "Other Renewables_",
    "Nuclear": "Nuclear_",
    "Fossil": "Fossil_"
}

# Sample conversion rates from MWh to USD by energy type
# Based on average levelized cost of electricity (LCOE) in USD/MWh
# Source: Sample data based on industry averages (Global LCOE median data)
conversion_rates = {
    "Hydro": 68,      # $68 per MWh
    "Solar": 60,      # $60 per MWh (utility-scale solar)
    "Wind": 60,       # $60 per MWh (onshore)
    "Other_Renewables": 120,  # $120 per MWh (biomass, geothermal)
    "Nuclear": 100,   # $100 per MWh
    "Fossil": 105,     # $105 per MWh (combined cycle natural gas)
}


# Calculate values by each type and year, then apply the exposure percentages
raw_type_values_dict = {}  # Non-exposure-adjusted values
type_values_dict = {}  # Exposure-adjusted values
risk_avoid_type_values_dict = {}  # Risk-avoidance adjusted values
raw_type_values_usd_dict = {}  # Non-exposure-adjusted values in USD
type_values_usd_dict = {}  # Exposure-adjusted values in USD
risk_avoid_type_values_usd_dict = {}  # Risk-avoidance adjusted values in USD

for energy_type, pattern in column_mapping.items():
    raw_type_values_dict[energy_type] = []
    type_values_dict[energy_type] = []
    risk_avoid_type_values_dict[energy_type] = []
    raw_type_values_usd_dict[energy_type] = []
    type_values_usd_dict[energy_type] = []
    risk_avoid_type_values_usd_dict[energy_type] = []
    
    for year in [2024, 2030, 2050]:
        # Calculate total MWh for the energy type and year
        type_col = f'{energy_type}_{year}'
        data[type_col] = data[[col for col in data.columns if pattern in col and str(year) in col]].sum(axis=1)
        
        # Store raw (non-exposure-adjusted) value
        raw_value = data[type_col].sum()
        raw_type_values_dict[energy_type].append(raw_value)
        
        # Convert to USD
        raw_value_usd = raw_value * conversion_rates[energy_type]
        raw_type_values_usd_dict[energy_type].append(raw_value_usd)
        
        # Apply exposure percentages
        # Fix the key generation to handle spaces in energy type names properly
        exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
        
        if year == 2024:
            pct = percentages_2024.get(exposure_col, 0)
        elif year == 2030:
            pct = percentages_2030.get(exposure_col, 0)
        else:
            pct = percentages_2050.get(exposure_col, 0)
        
        data[exposure_col] = data[type_col] * (pct / 100)
        
        # Store exposure-adjusted value
        exp_value = data[exposure_col].sum()
        type_values_dict[energy_type].append(exp_value)
        
        # Convert to USD
        exp_value_usd = exp_value * conversion_rates[energy_type]
        type_values_usd_dict[energy_type].append(exp_value_usd)
        
        # Apply risk-avoidance exposure percentages
        exposure_col_risk_avoid = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
 
        if year == 2024:
            pct_risk_avoid = percentages_2024_risk_avoid.get(exposure_col_risk_avoid, 0)
        elif year == 2030:
            pct_risk_avoid = percentages_2030_risk_avoid.get(exposure_col_risk_avoid, 0)
        else:
            pct_risk_avoid = percentages_2050_risk_avoid.get(exposure_col_risk_avoid, 0)
        
        data[exposure_col_risk_avoid] = data[type_col] * (pct_risk_avoid / 100)
        exp_value_risk_avoid = data[exposure_col_risk_avoid].sum()
        risk_avoid_type_values_dict[energy_type].append(exp_value_risk_avoid)
        
        exp_value_risk_avoid_usd = exp_value_risk_avoid * conversion_rates[energy_type]
        risk_avoid_type_values_usd_dict[energy_type].append(exp_value_risk_avoid_usd)

# Prepare data for plotting
years = [2024, 2030, 2050]
# Reorder energy types as requested: Fossil, Nuclear, Other Renewable, Hydro, Wind, and Solar
energy_types = ["Fossil", "Nuclear", "Other_Renewables", "Hydro", "Wind", "Solar"]

# Sample numbers for IEA scenarios
iea_scenarios = {
    "Stated Policies": [29863.4 * 1000000, 37488.89 * 1000000, 58352.13 * 1000000],
    "Announced Pledges": [29863.4 * 1000000, 38284.86 * 1000000, 70564.11 * 1000000],
    "Net Zero by 2050": [29863.4 * 1000000, 39782.75 * 1000000, 80194.39 * 1000000],
}

# Convert IEA scenarios to USD (using weighted average conversion rate)
iea_scenarios_usd = {}
for scenario, values in iea_scenarios.items():
    iea_scenarios_usd[scenario] = []
    for i, value in enumerate(values):
        # Calculate weighted average conversion rate based on energy mix in raw values
        total_energy = sum(raw_type_values_dict[et][i] for et in energy_types)
        weighted_rate = sum(raw_type_values_dict[et][i] / total_energy * conversion_rates[et] for et in energy_types)
        iea_scenarios_usd[scenario].append(value * weighted_rate)

# Convert all values from MWh to TWh and USD to billions USD for plotting
for energy_type in energy_types:
    raw_type_values_dict[energy_type] = [val / 1000000 for val in raw_type_values_dict[energy_type]]
    type_values_dict[energy_type] = [val / 1000000 for val in type_values_dict[energy_type]]
    risk_avoid_type_values_dict[energy_type] = [val / 1000000 for val in risk_avoid_type_values_dict[energy_type]]
    raw_type_values_usd_dict[energy_type] = [val / 1000000000 for val in raw_type_values_usd_dict[energy_type]]
    type_values_usd_dict[energy_type] = [val / 1000000000 for val in type_values_usd_dict[energy_type]]
    risk_avoid_type_values_usd_dict[energy_type] = [val / 1000000000 for val in risk_avoid_type_values_usd_dict[energy_type]]

for scenario in iea_scenarios:
    iea_scenarios[scenario] = [val / 1000000 for val in iea_scenarios[scenario]]
    iea_scenarios_usd[scenario] = [val / 1000000000 for val in iea_scenarios_usd[scenario]]

# Calculate totals for analysis
# These totals represent the aggregated values across all energy types for each year (2024, 2030, 2050).

# Total raw energy generation (non-adjusted) in TWh for each year
total_raw = [sum(raw_type_values_dict[et][i] for et in energy_types) for i in range(3)]

# Total exposure-adjusted energy generation in TWh for each year
total_exp = [sum(type_values_dict[et][i] for et in energy_types) for i in range(3)]

# Total risk-avoidance energy generation in TWh for each year
risk_avoid_total = [sum(risk_avoid_type_values_dict[et][i] for et in energy_types) for i in range(3)]

# Total raw economic value (non-adjusted) in billion USD for each year
total_raw_usd = [sum(raw_type_values_usd_dict[et][i] for et in energy_types) for i in range(3)]

# Total exposure-adjusted economic value in billion USD for each year
total_exp_usd = [sum(type_values_usd_dict[et][i] for et in energy_types) for i in range(3)]

# Total risk-avoidance economic value in billion USD for each year
risk_avoid_total_usd = [sum(risk_avoid_type_values_usd_dict[et][i] for et in energy_types) for i in range(3)]

# Print summary statistics
print("\n=== ENERGY GENERATION SUMMARY (TWh) ===")
for i, year in enumerate(years):
    print(f"\n{year}:")
    print(f"  Total Raw: {total_raw[i]:,.2f} TWh")
    print(f"  Total Exposure-Adjusted: {total_exp[i]:,.2f} TWh")
    print(f"  Total Risk-Avoidance: {risk_avoid_total[i]:,.2f} TWh")
    for scenario, values in iea_scenarios.items():
        print(f"  IEA {scenario}: {values[i]:,.2f} TWh")

print("\n=== ECONOMIC VALUE SUMMARY (Billion USD) ===")
for i, year in enumerate(years):
    print(f"\n{year}:")
    print(f"  Total Raw: ${total_raw_usd[i]:,.2f}B")
    print(f"  Total Exposure-Adjusted: ${total_exp_usd[i]:,.2f}B")
    print(f"  Total Risk-Avoidance: ${risk_avoid_total_usd[i]:,.2f}B")
    for scenario, values in iea_scenarios_usd.items():
        print(f"  IEA {scenario}: ${values[i]:,.2f}B")

print("\n=== DATA PROCESSING COMPLETE ===")
print("Processed data is available in the following dictionaries:")
print("  - raw_type_values_dict: Non-adjusted energy generation by type (TWh)")
print("  - type_values_dict: Exposure-adjusted energy generation by type (TWh)")
print("  - risk_avoid_type_values_dict: Risk-avoidance energy generation by type (TWh)")
print("  - raw_type_values_usd_dict: Non-adjusted economic value by type (Billion USD)")
print("  - type_values_usd_dict: Exposure-adjusted economic value by type (Billion USD)")
print("  - risk_avoid_type_values_usd_dict: Risk-avoidance economic value by type (Billion USD)")
print("  - iea_scenarios: IEA scenario projections (TWh)")
print("  - iea_scenarios_usd: IEA scenario projections (Billion USD)")

# Save processed data to Excel file
print("\n=== SAVING PROCESSED DATA TO EXCEL ===")

# Create a pandas ExcelWriter object
output_file = 'outputs_processed_data/p1_z_results_data_etl.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: Energy Generation (TWh) by Type
    energy_df = pd.DataFrame({
        'Energy_Type': energy_types,
        'Raw_2024_TWh': [raw_type_values_dict[et][0] for et in energy_types],
        'Raw_2030_TWh': [raw_type_values_dict[et][1] for et in energy_types],
        'Raw_2050_TWh': [raw_type_values_dict[et][2] for et in energy_types],
        'Exposure_2024_TWh': [type_values_dict[et][0] for et in energy_types],
        'Exposure_2030_TWh': [type_values_dict[et][1] for et in energy_types],
        'Exposure_2050_TWh': [type_values_dict[et][2] for et in energy_types],
        'Risk_Avoid_2024_TWh': [risk_avoid_type_values_dict[et][0] for et in energy_types],
        'Risk_Avoid_2030_TWh': [risk_avoid_type_values_dict[et][1] for et in energy_types],
        'Risk_Avoid_2050_TWh': [risk_avoid_type_values_dict[et][2] for et in energy_types],
    })
    energy_df.to_excel(writer, sheet_name='Energy_Generation_TWh', index=False)
    
    # Sheet 2: Economic Value (Billion USD) by Type
    economic_df = pd.DataFrame({
        'Energy_Type': energy_types,
        'Raw_2024_USD_B': [raw_type_values_usd_dict[et][0] for et in energy_types],
        'Raw_2030_USD_B': [raw_type_values_usd_dict[et][1] for et in energy_types],
        'Raw_2050_USD_B': [raw_type_values_usd_dict[et][2] for et in energy_types],
        'Exposure_2024_USD_B': [type_values_usd_dict[et][0] for et in energy_types],
        'Exposure_2030_USD_B': [type_values_usd_dict[et][1] for et in energy_types],
        'Exposure_2050_USD_B': [type_values_usd_dict[et][2] for et in energy_types],
        'Risk_Avoid_2024_USD_B': [risk_avoid_type_values_usd_dict[et][0] for et in energy_types],
        'Risk_Avoid_2030_USD_B': [risk_avoid_type_values_usd_dict[et][1] for et in energy_types],
        'Risk_Avoid_2050_USD_B': [risk_avoid_type_values_usd_dict[et][2] for et in energy_types],
    })
    economic_df.to_excel(writer, sheet_name='Economic_Value_USD_B', index=False)
    
    # Sheet 3: Total Summaries
    totals_df = pd.DataFrame({
        'Year': years,
        'Total_Raw_TWh': total_raw,
        'Total_Exposure_TWh': total_exp,
        'Total_Risk_Avoid_TWh': risk_avoid_total,
        'Total_Raw_USD_B': total_raw_usd,
        'Total_Exposure_USD_B': total_exp_usd,
        'Total_Risk_Avoid_USD_B': risk_avoid_total_usd,
    })
    totals_df.to_excel(writer, sheet_name='Total_Summaries', index=False)
    
    # Sheet 4: IEA Scenarios (TWh)
    iea_twh_df = pd.DataFrame({
        'Year': years,
        'IEA_Stated_Policies_TWh': iea_scenarios['Stated Policies'],
        'IEA_Announced_Pledges_TWh': iea_scenarios['Announced Pledges'],
        'IEA_Net_Zero_2050_TWh': iea_scenarios['Net Zero by 2050'],
    })
    iea_twh_df.to_excel(writer, sheet_name='IEA_Scenarios_TWh', index=False)
    
    # Sheet 5: IEA Scenarios (Billion USD)
    iea_usd_df = pd.DataFrame({
        'Year': years,
        'IEA_Stated_Policies_USD_B': iea_scenarios_usd['Stated Policies'],
        'IEA_Announced_Pledges_USD_B': iea_scenarios_usd['Announced Pledges'],
        'IEA_Net_Zero_2050_USD_B': iea_scenarios_usd['Net Zero by 2050'],
    })
    iea_usd_df.to_excel(writer, sheet_name='IEA_Scenarios_USD_B', index=False)
    
    # Sheet 6: Conversion Rates (USD per MWh)
    conversion_df = pd.DataFrame(list(conversion_rates.items()), columns=['Energy_Type', 'USD_per_MWh'])
    conversion_df.to_excel(writer, sheet_name='Conversion_Rates', index=False)

print(f"Data saved successfully to: {output_file}")
print("Excel sheets created:")
print("  1. Energy_Generation_TWh - Raw, Exposure, and Risk-Avoidance values by type")
print("  2. Economic_Value_USD_B - Raw, Exposure, and Risk-Avoidance economic values by type")
print("  3. Total_Summaries - Aggregated totals across all energy types")
print("  4. IEA_Scenarios_TWh - IEA scenario projections in TWh")
print("  5. IEA_Scenarios_USD_B - IEA scenario projections in Billion USD")
print("  6. Conversion_Rates - USD per MWh conversion rates by energy type")

