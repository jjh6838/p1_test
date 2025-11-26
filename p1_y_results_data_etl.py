import pandas as pd
import numpy as np

# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_b_ember_2024_30_50.xlsx')

# Keep ISO3_code and country identifying columns
id_cols = ['ISO3_code'] + [col for col in data.columns if col in ['Country', 'Country_Name', 'Region']]

# Define scenarios
years = [2030, 2050]
supply_scenarios = ['100%', '90%', '80%', '70%', '60%']
hazard_buffers = ['0km', '10km', '20km', '30km', '40km']
energy_types = ['hydro', 'solar', 'wind', 'other_renewables', 'nuclear', 'fossil']

# Generate random percentages for exposure
np.random.seed(42)  # For reproducibility

# Define custom ranges for each energy type
ranges_exp = {
    "hydro": (12, 19),
    "solar": (25, 45),
    "wind": (25, 40),
    "other_renewables": (1, 15),
    "nuclear": (1, 8),
    "fossil": (1, 8),
}

# Define ranges for risk-avoidance (reduced exposure)
ranges_risk_avoid = {
    "hydro": (6, 10),
    "solar": (12, 22),
    "wind": (12, 20),
    "other_renewables": (0.5, 7),
    "nuclear": (0.5, 4),
    "fossil": (0.5, 4),
}

# Generate random percentages with proportional relationships
# - As supply decreases (100% -> 60%), exposure decreases proportionally
# - As buffer increases (0km -> 40km), exposure increases proportionally
# - Risk-avoidance exposure remains relatively constant across buffers
exposure_percentages = {}
risk_avoid_percentages = {}

supply_values = [100, 90, 80, 70, 60]  # Numeric values for calculation
buffer_values = [0, 10, 20, 30, 40]  # Numeric values for calculation

for energy_type in energy_types:
    for year in years:
        # Get base percentage at 100% supply, 0km buffer
        base_exp = np.random.uniform(*ranges_exp[energy_type])
        base_risk = np.random.uniform(*ranges_risk_avoid[energy_type])
        
        for supply_idx, supply in enumerate(supply_scenarios):
            supply_val = supply_values[supply_idx]
            # Supply factor: 100% = 1.0, 90% = 0.9, etc. (linear decrease)
            supply_factor = supply_val / 100.0
            
            for buffer_idx, buffer in enumerate(hazard_buffers):
                buffer_val = buffer_values[buffer_idx]
                # Buffer factor: increases exposure as buffer increases
                # 0km = 1.0, 10km = 1.05, 20km = 1.10, etc. (5% increase per 10km)
                buffer_factor = 1.0 + (buffer_val / 10) * 0.05
                
                key = f"{energy_type}_{year}_supply_{supply}_{buffer}"
                
                # Exposure: affected by both supply and buffer
                # Add small random variation (±2%) for realism
                variation = np.random.uniform(0.98, 1.02)
                exposure_percentages[key] = base_exp * supply_factor * buffer_factor * variation
                
                # Risk-avoidance: affected by supply but CONSTANT across buffers
                # Only small random variation (±1%) for realism
                risk_variation = np.random.uniform(0.99, 1.01)
                risk_avoid_percentages[key] = base_risk * supply_factor * risk_variation

# Map energy types to actual column patterns in your data
column_mapping = {
    "hydro": "Hydro_",
    "solar": "Solar_",
    "wind": "Wind_",
    "other_renewables": "Other Renewables_",
    "nuclear": "Nuclear_",
    "fossil": "Fossil_"
}

# Conversion rates from MWh to USD by energy type
# Based on average levelized cost of electricity (LCOE) in USD/MWh
conversion_rates = {
    "hydro": 68,      # $68 per MWh
    "solar": 60,      # $60 per MWh (utility-scale solar)
    "wind": 60,       # $60 per MWh (onshore)
    "other_renewables": 120,  # $120 per MWh (biomass, geothermal)
    "nuclear": 100,   # $100 per MWh
    "fossil": 105,     # $105 per MWh (combined cycle natural gas)
}

# Calculate energy generation by type and year for each country
print("\n=== CALCULATING COUNTRY-LEVEL ENERGY VALUES ===")

# Create base energy columns for each type and year
for energy_type, pattern in column_mapping.items():
    for year in years:
        # Sum all columns matching this energy type and year
        matching_cols = [col for col in data.columns if pattern in col and str(year) in col]
        if matching_cols:
            data[f'{energy_type}_{year}_MWh'] = data[matching_cols].sum(axis=1)
        else:
            data[f'{energy_type}_{year}_MWh'] = 0
            print(f"Warning: No columns found for {energy_type} {year}")

# Calculate exposure and convert to long format directly (no intermediate columns)
print("Calculating exposures and building long format dataset...")
total_combinations = len(energy_types) * len(years) * len(supply_scenarios) * len(hazard_buffers)

# Collect all rows for long format
long_data = []
total_rows = len(data)

for idx, row in data.iterrows():
    iso3 = row['ISO3_code']
    
    for energy_type in energy_types:
        for year in years:
            base_mwh = row[f'{energy_type}_{year}_MWh']
            
            for supply in supply_scenarios:
                for buffer in hazard_buffers:
                    key = f"{energy_type}_{year}_supply_{supply}_{buffer}"
                    
                    # Get percentages
                    exp_pct = exposure_percentages[key]
                    risk_pct = risk_avoid_percentages[key]
                    
                    # Calculate values directly
                    exp_mwh = base_mwh * (exp_pct / 100)
                    exp_usd = exp_mwh * conversion_rates[energy_type]
                    risk_mwh = base_mwh * (risk_pct / 100)
                    risk_usd = risk_mwh * conversion_rates[energy_type]
                    
                    long_data.append({
                        'ISO3_code': iso3,
                        'energy_type': energy_type,
                        'year': year,
                        'supply_scenario': supply,
                        'hazard_buffer': buffer,
                        'exp_MWh': exp_mwh,
                        'exp_USD': exp_usd,
                        'exp_risk_avoid_MWh': risk_mwh,
                        'exp_risk_avoid_USD': risk_usd
                    })
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{total_rows} countries...")

# Create long format dataframe
output_df = pd.DataFrame(long_data)

# Save to Excel
output_file = 'outputs_processed_data/p1_y_results_data_etl.xlsx'
output_df.to_excel(output_file, index=False, engine='openpyxl')

print(f"\n✅ Data saved successfully to: {output_file}")
print(f"   Total rows: {len(output_df):,}")
print(f"   Countries: {output_df['ISO3_code'].nunique()}")
print(f"   Columns: {list(output_df.columns)}")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total exposure (exp_MWh): {output_df['exp_MWh'].sum():,.2f} MWh")
print(f"Total exposure (exp_USD): ${output_df['exp_USD'].sum():,.2f}")
print(f"Total risk-avoidance (exp_risk_avoid_MWh): {output_df['exp_risk_avoid_MWh'].sum():,.2f} MWh")
print(f"Total risk-avoidance (exp_risk_avoid_USD): ${output_df['exp_risk_avoid_USD'].sum():,.2f}")

# Sample data
print("\n=== SAMPLE DATA (first 10 rows) ===")
print(output_df.head(10).to_string(index=False))

print("\n=== PROCESSING COMPLETE ===")
print("Long format with columns:")
print("  - ISO3_code: Country code")
print("  - energy_type: hydro, solar, wind, other_renewables, nuclear, fossil")
print("  - year: 2030, 2050")
print("  - supply_scenario: 100%, 90%, 80%, 70%, 60%")
print("  - hazard_buffer: 0km, 10km, 20km, 30km, 40km")
print("  - exp_MWh: Standard exposure in MWh")
print("  - exp_USD: Standard exposure in USD")
print("  - exp_risk_avoid_MWh: Risk-avoidance exposure in MWh")
print("  - exp_risk_avoid_USD: Risk-avoidance exposure in USD")

