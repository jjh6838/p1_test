import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Oxford color guides
# https://communications.admin.ox.ac.uk/communications-resources/visual-identity/identity-guidelines/colours#:~:text=Do%20not%20use%20secondary%20colour,on%20top%20of%20Oxford%20blue

plt.rcParams['font.family'] = 'Arial'  # Or 'Helvetica'


# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_a_ember_2024_30.xlsx')

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

# Calculate totals for line graphs
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

# Font sizes
title_fs = 8
label_fs = 7
tick_fs = 7
legend_fs = 7
annot_fs = 7


# === FIGURE 3: Custom 11 subplots (2024–2050) - TWh ===
fig3 = plt.figure(figsize=(7.2, 5.5), dpi=300)
gs3 = gridspec.GridSpec(3, 4, wspace=0.15, hspace=0.3)

# Get unique income groups and regions from the data
income_groups = data['Income group'].unique()
regions = data['Region'].unique()

# Calculate aggregated TWh data for income groups
income_group_data_twh = {}
income_group_exposure_twh = {}
income_group_risk_avoid_twh = {}

for group in income_groups:
    group_data = data[data['Income group'] == group]
    twh_values = []
    exposure_values = []
    risk_avoid_values = []
    
    for year in [2024, 2030, 2050]:
        # Calculate total TWh for this income group and year
        total_mwh = 0
        total_exposure_mwh = 0
        total_risk_avoid_mwh = 0
        
        for energy_type in conversion_rates.keys():
            # Find columns for this energy type and year
            pattern = column_mapping[energy_type]
            energy_cols = [col for col in data.columns if pattern in col and str(year) in col]
            energy_mwh = group_data[energy_cols].sum().sum()
            total_mwh += energy_mwh
            
            # Calculate exposure and risk-avoidance values
            exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
            risk_avoid_col = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
            
            if year == 2024:
                pct = percentages_2024.get(exposure_col, 0)
                pct_risk = percentages_2024_risk_avoid.get(risk_avoid_col, 0)
            elif year == 2030:
                pct = percentages_2030.get(exposure_col, 0)
                pct_risk = percentages_2030_risk_avoid.get(risk_avoid_col, 0)
            else:
                pct = percentages_2050.get(exposure_col, 0)
                pct_risk = percentages_2050_risk_avoid.get(risk_avoid_col, 0)
            
            total_exposure_mwh += energy_mwh * (pct / 100)
            total_risk_avoid_mwh += energy_mwh * (pct_risk / 100)
        
        total_twh = total_mwh / 1000000
        exposure_twh = total_exposure_mwh / 1000000
        risk_avoid_twh = total_risk_avoid_mwh / 1000000
        
        twh_values.append(total_twh)
        exposure_values.append(exposure_twh)
        risk_avoid_values.append(risk_avoid_twh)
    
    income_group_data_twh[group] = twh_values
    income_group_exposure_twh[group] = exposure_values
    income_group_risk_avoid_twh[group] = risk_avoid_values

# Calculate aggregated TWh data for regions
regional_group_data_twh = {}
regional_group_exposure_twh = {}
regional_group_risk_avoid_twh = {}

for region in regions:
    region_data = data[data['Region'] == region]
    twh_values = []
    exposure_values = []
    risk_avoid_values = []
    
    for year in [2024, 2030, 2050]:
        # Calculate total TWh for this region and year
        total_mwh = 0
        total_exposure_mwh = 0
        total_risk_avoid_mwh = 0
        
        for energy_type in conversion_rates.keys():
            # Find columns for this energy type and year
            pattern = column_mapping[energy_type]
            energy_cols = [col for col in data.columns if pattern in col and str(year) in col]
            energy_mwh = region_data[energy_cols].sum().sum()
            total_mwh += energy_mwh
            
            # Calculate exposure and risk-avoidance values
            exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
            risk_avoid_col = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
            
            if year == 2024:
                pct = percentages_2024.get(exposure_col, 0)
                pct_risk = percentages_2024_risk_avoid.get(risk_avoid_col, 0)
            elif year == 2030:
                pct = percentages_2030.get(exposure_col, 0)
                pct_risk = percentages_2030_risk_avoid.get(risk_avoid_col, 0)
            else:
                pct = percentages_2050.get(exposure_col, 0)
                pct_risk = percentages_2050_risk_avoid.get(risk_avoid_col, 0)
            
            total_exposure_mwh += energy_mwh * (pct / 100)
            total_risk_avoid_mwh += energy_mwh * (pct_risk / 100)
        
        total_twh = total_mwh / 1000000
        exposure_twh = total_exposure_mwh / 1000000
        risk_avoid_twh = total_risk_avoid_mwh / 1000000
        
        twh_values.append(total_twh)
        exposure_values.append(exposure_twh)
        risk_avoid_values.append(risk_avoid_twh)
    
    regional_group_data_twh[region] = twh_values
    regional_group_exposure_twh[region] = exposure_values
    regional_group_risk_avoid_twh[region] = risk_avoid_values

# Sort income groups by total exposed generation (sum across all years)
income_groups_sorted = sorted(income_groups, 
                             key=lambda x: sum(income_group_exposure_twh[x]), 
                             reverse=True)

# Sort regions by total exposed generation (sum across all years)
regions_sorted = sorted(regions, 
                       key=lambda x: sum(regional_group_exposure_twh[x]), 
                       reverse=True)

# First row: Income groups (sorted by exposed generation, highest first)
for i, group in enumerate(income_groups_sorted):
    ax = fig3.add_subplot(gs3[0, i])

    # Calculate y-axis limits for Figure 3 (TWh) - find max across all groups
    max_twh_exposure = max([max(income_group_exposure_twh[group]) for group in income_groups] +
                           [max(regional_group_exposure_twh[region]) for region in regions])
    max_twh_risk_avoid = max([max(income_group_risk_avoid_twh[group]) for group in income_groups] +
                             [max(regional_group_risk_avoid_twh[region]) for region in regions])
    ylim_twh = max(max_twh_exposure, max_twh_risk_avoid) * 1.2  # Add 10% padding
    ylim_twh_min = max(max_twh_exposure, max_twh_risk_avoid) * -0.05  # Add 5% padding from zero
    xlim_twh = 2053  # Set x-axis limit to 2050
    xlim_twh_min = 2021  # Set x-axis limit to 2022

    # Plot TWh data - using Figure 1 colors
    ax.plot(years, income_group_risk_avoid_twh[group], 'o-', color='#002147', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, income_group_exposure_twh[group], 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposed Generation')

    # Annotations
    for j, year in enumerate(years):
        if income_group_exposure_twh[group][j] > 0:
            ax.text(year, income_group_exposure_twh[group][j] * 1.05, f'{income_group_exposure_twh[group][j]:.0f}', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#AA1A2D')
        if income_group_risk_avoid_twh[group][j] > 0 and year != 2024:
            ax.text(year, income_group_risk_avoid_twh[group][j] * 0.95, f'{income_group_risk_avoid_twh[group][j]:.0f}', 
                    ha='center', va='top', fontsize=annot_fs, color='#002147')
    
    # Styling
    ax.set_title(group, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_twh_min, ylim_twh])
    ax.set_xlim([xlim_twh_min, xlim_twh])  # Set x-axis limits to match years
    
    # Y-axis label
    if i == 0:
        ax.set_ylabel('Generation (TWh)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Second row: First 4 regional groups (sorted by exposed generation, highest first)
for i, region in enumerate(regions_sorted[:4]):
    ax = fig3.add_subplot(gs3[1, i])
    
    # Plot TWh data - using Figure 1 colors
    ax.plot(years, regional_group_risk_avoid_twh[region], 'o-', color='#002147', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, regional_group_exposure_twh[region], 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposed Generation')

    # Annotations
    for j, year in enumerate(years):
        if regional_group_exposure_twh[region][j] > 0:
            ax.text(year, regional_group_exposure_twh[region][j] * 1.05, f'{regional_group_exposure_twh[region][j]:.0f}', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#AA1A2D')
        if regional_group_risk_avoid_twh[region][j] > 0 and year != 2024:
            ax.text(year, regional_group_risk_avoid_twh[region][j] * 0.95, f'{regional_group_risk_avoid_twh[region][j]:.0f}', 
                    ha='center', va='top', fontsize=annot_fs, color='#002147')
    
    # Styling
    ax.set_title(region, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_twh_min, ylim_twh])
    ax.set_xlim([xlim_twh_min, xlim_twh])  # Set x-axis limits to match years
    
    # Y-axis label
    if i == 0:
        ax.set_ylabel('Generation (TWh)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Third row: Remaining regional groups + legend
remaining_regions_sorted = regions_sorted[4:]
for i, region in enumerate(remaining_regions_sorted):
    ax = fig3.add_subplot(gs3[2, i])
    
    # Plot TWh data - using Figure 1 colors
    ax.plot(years, regional_group_risk_avoid_twh[region], 'o-', color='#002147', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, regional_group_exposure_twh[region], 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposed Generation')
    
    
    # Annotations
    for j, year in enumerate(years):
        if regional_group_exposure_twh[region][j] > 0:
            ax.text(year, regional_group_exposure_twh[region][j] * 1.05, f'{regional_group_exposure_twh[region][j]:.0f}', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#AA1A2D')
        if regional_group_risk_avoid_twh[region][j] > 0:
            ax.text(year, regional_group_risk_avoid_twh[region][j] * 0.95, f'{regional_group_risk_avoid_twh[region][j]:.0f}', 
                    ha='center', va='top', fontsize=annot_fs, color='#002147')
    
    # Styling
    ax.set_title(region, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_twh_min, ylim_twh])
    ax.set_xlim([xlim_twh_min, xlim_twh])  # Set x-axis limits to match years
    
    # Y-axis label
    if i == 0:
        ax.set_ylabel('Generation (TWh)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Legend subplot (bottom right)
ax_legend = fig3.add_subplot(gs3[2, 3])
ax_legend.axis('off')

# Create legend handles using Figure 1 colors
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#AA1A2D', label='Exposed Generation'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#002147', label='Exposed (With Risk-Avoidance)')
]

# Add legend
ax_legend.legend(handles=legend_handles, loc='center', fontsize=legend_fs, frameon=True, 
                facecolor='white', edgecolor='gray', title='Legend', title_fontsize=title_fs)

# Main title and save
fig3.suptitle('Figure 3. Climate Risk Exposure of Electricity Generation by Income Group and Region (2024–2050)', 
              fontsize=10, fontweight='bold', y=0.95)
fig3.savefig('outputs_processed_fig/fig3_alt1.pdf', bbox_inches='tight', dpi=300)


# === FIGURE 4: Custom 11 subplots (2024–2050) - Economic Value (USD) ===
fig4 = plt.figure(figsize=(7.2, 5.5), dpi=300)
gs4 = gridspec.GridSpec(3, 4, wspace=0.15, hspace=0.3)


# Calculate aggregated USD data for income groups
income_group_data_usd = {}
income_group_exposure_usd = {}
income_group_risk_avoid_usd = {}

for group in income_groups:
    group_data = data[data['Income group'] == group]
    usd_values = []
    exposure_values = []
    risk_avoid_values = []
    
    for year in [2024, 2030, 2050]:
        # Calculate total USD value for this income group and year
        total_usd = 0
        total_exposure_usd = 0
        total_risk_avoid_usd = 0
        
        for energy_type, rate in conversion_rates.items():
            # Find columns for this energy type and year
            pattern = column_mapping[energy_type]
            energy_cols = [col for col in data.columns if pattern in col and str(year) in col]
            energy_mwh = group_data[energy_cols].sum().sum()
            total_usd += energy_mwh * rate
            
            # Calculate exposure and risk-avoidance values
            exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
            risk_avoid_col = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
            
            if year == 2024:
                pct = percentages_2024.get(exposure_col, 0)
                pct_risk = percentages_2024_risk_avoid.get(risk_avoid_col, 0)
            elif year == 2030:
                pct = percentages_2030.get(exposure_col, 0)
                pct_risk = percentages_2030_risk_avoid.get(risk_avoid_col, 0)
            else:
                pct = percentages_2050.get(exposure_col, 0)
                pct_risk = percentages_2050_risk_avoid.get(risk_avoid_col, 0)
            
            total_exposure_usd += energy_mwh * rate * (pct / 100)
            total_risk_avoid_usd += energy_mwh * rate * (pct_risk / 100)
        
        total_usd_billions = total_usd / 1000000000
        exposure_usd_billions = total_exposure_usd / 1000000000
        risk_avoid_usd_billions = total_risk_avoid_usd / 1000000000
        
        usd_values.append(total_usd_billions)
        exposure_values.append(exposure_usd_billions)
        risk_avoid_values.append(risk_avoid_usd_billions)
    
    income_group_data_usd[group] = usd_values
    income_group_exposure_usd[group] = exposure_values
    income_group_risk_avoid_usd[group] = risk_avoid_values

# Calculate aggregated USD data for regions
regional_group_data_usd = {}
regional_group_exposure_usd = {}
regional_group_risk_avoid_usd = {}

for region in regions:
    region_data = data[data['Region'] == region]
    usd_values = []
    exposure_values = []
    risk_avoid_values = []
    
    for year in [2024, 2030, 2050]:
        # Calculate total USD value for this region and year
        total_usd = 0
        total_exposure_usd = 0
        total_risk_avoid_usd = 0
        
        for energy_type, rate in conversion_rates.items():
            # Find columns for this energy type and year
            pattern = column_mapping[energy_type]
            energy_cols = [col for col in data.columns if pattern in col and str(year) in col]
            energy_mwh = region_data[energy_cols].sum().sum()
            total_usd += energy_mwh * rate
            
            # Calculate exposure and risk-avoidance values
            exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
            risk_avoid_col = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
            
            if year == 2024:
                pct = percentages_2024.get(exposure_col, 0)
                pct_risk = percentages_2024_risk_avoid.get(risk_avoid_col, 0)
            elif year == 2030:
                pct = percentages_2030.get(exposure_col, 0)
                pct_risk = percentages_2030_risk_avoid.get(risk_avoid_col, 0)
            else:
                pct = percentages_2050.get(exposure_col, 0)
                pct_risk = percentages_2050_risk_avoid.get(risk_avoid_col, 0)
            
            total_exposure_usd += energy_mwh * rate * (pct / 100)
            total_risk_avoid_usd += energy_mwh * rate * (pct_risk / 100)
        
        total_usd_billions = total_usd / 1000000000
        exposure_usd_billions = total_exposure_usd / 1000000000
        risk_avoid_usd_billions = total_risk_avoid_usd / 1000000000
        
        usd_values.append(total_usd_billions)
        exposure_values.append(exposure_usd_billions)
        risk_avoid_values.append(risk_avoid_usd_billions)
    
    regional_group_data_usd[region] = usd_values
    regional_group_exposure_usd[region] = exposure_values
    regional_group_risk_avoid_usd[region] = risk_avoid_values

# Sort income groups by total exposed economic value (sum across all years)
income_groups_sorted_usd = sorted(income_groups, 
                                 key=lambda x: sum(income_group_exposure_usd[x]), 
                                 reverse=True)

# Sort regions by total exposed economic value (sum across all years)
regions_sorted_usd = sorted(regions, 
                           key=lambda x: sum(regional_group_exposure_usd[x]), 
                           reverse=True)

# First row: Income groups (sorted by exposed economic value, highest first)
for i, group in enumerate(income_groups_sorted_usd):
    ax = fig4.add_subplot(gs4[0, i])
    
    # Calculate y-axis limits for Figure 4 (USD) - find max across all groups
    max_usd_exposure = max([max(income_group_exposure_usd[group]) for group in income_groups] +
                        [max(regional_group_exposure_usd[region]) for region in regions])
    max_usd_risk_avoid = max([max(income_group_risk_avoid_usd[group]) for group in income_groups] +
                            [max(regional_group_risk_avoid_usd[region]) for region in regions])
    ylim_usd = max(max_usd_exposure, max_usd_risk_avoid) * 1.2  # Add 10% padding
    ylim_usd_min = max(max_usd_exposure, max_usd_risk_avoid) * -0.05  # Add 10% padding
    xlim_usd = 2053  # Set x-axis limit to 2050
    xlim_usd_min = 2021  # Set x-axis limit to 2022


    # Plot USD data - using Figure 2 colors
    ax.plot(years, income_group_risk_avoid_usd[group], 'o-', color='#426A5A', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, income_group_exposure_usd[group], 'o-', color='#E2C044', linewidth=2.5, markersize=3, label='Exposed Economic Value')
    
    
    # Annotations
    for j, year in enumerate(years):
        if income_group_exposure_usd[group][j] > 0:
            ax.text(year, income_group_exposure_usd[group][j] * 1.05, f'${income_group_exposure_usd[group][j]:.0f}B', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#E2C044')
        if income_group_risk_avoid_usd[group][j] > 0 and year != 2024:
            ax.text(year, income_group_risk_avoid_usd[group][j] * 0.95, f'${income_group_risk_avoid_usd[group][j]:.0f}B', 
                    ha='center', va='top', fontsize=annot_fs, color='#426A5A')
    
    # Styling
    ax.set_title(group, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_usd_min, ylim_usd])
    ax.set_xlim([xlim_usd_min, xlim_usd])

    # Y-axis label
    if i == 0:
        ax.set_ylabel('Economic Value (Billion USD)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Second row: First 4 regional groups (sorted by exposed economic value, highest first)
for i, region in enumerate(regions_sorted_usd[:4]):
    ax = fig4.add_subplot(gs4[1, i])
    
    # Plot USD data - using Figure 2 colors
    ax.plot(years, regional_group_risk_avoid_usd[region], 'o-', color='#426A5A', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, regional_group_exposure_usd[region], 'o-', color='#E2C044', linewidth=2.5, markersize=3, label='Exposed Economic Value')
    
    
    # Annotations
    for j, year in enumerate(years):
        if regional_group_exposure_usd[region][j] > 0:
            ax.text(year, regional_group_exposure_usd[region][j] * 1.05, f'${regional_group_exposure_usd[region][j]:.0f}B', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#E2C044')
        if regional_group_risk_avoid_usd[region][j] > 0 and year != 2024:
            ax.text(year, regional_group_risk_avoid_usd[region][j] * 0.95, f'${regional_group_risk_avoid_usd[region][j]:.0f}B', 
                    ha='center', va='top', fontsize=annot_fs, color='#426A5A')
    
    # Styling
    ax.set_title(region, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_usd_min, ylim_usd])
    ax.set_xlim([xlim_usd_min, xlim_usd]) 
    
    # Y-axis label
    if i == 0:
        ax.set_ylabel('Economic Value (Billion USD)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Third row: Remaining regional groups + legend
remaining_regions_sorted_usd = regions_sorted_usd[4:]
for i, region in enumerate(remaining_regions_sorted_usd):
    ax = fig4.add_subplot(gs4[2, i])
    
    # Plot USD data - using Figure 2 colors
    ax.plot(years, regional_group_risk_avoid_usd[region], 'o-', color='#426A5A', linewidth=2.5, markersize=3, label='Exposed (With Risk-Avoidance)')
    ax.plot(years, regional_group_exposure_usd[region], 'o-', color='#E2C044', linewidth=2.5, markersize=3, label='Exposed Economic Value')
    
    
    # Annotations
    for j, year in enumerate(years):
        if regional_group_exposure_usd[region][j] > 0:
            ax.text(year, regional_group_exposure_usd[region][j] * 1.05, f'${regional_group_exposure_usd[region][j]:.0f}B', 
                    ha='center', va='bottom', fontsize=annot_fs, color='#E2C044')
        if regional_group_risk_avoid_usd[region][j] > 0 and year != 2024:
            ax.text(year, regional_group_risk_avoid_usd[region][j] * 0.95, f'${regional_group_risk_avoid_usd[region][j]:.0f}B', 
                    ha='center', va='top', fontsize=annot_fs, color='#426A5A')
    
    # Styling
    ax.set_title(region, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.1)
    ax.set_ylim([ylim_usd_min, ylim_usd])
    ax.set_xlim([xlim_usd_min, xlim_usd]) 
    
    # Y-axis label
    if i == 0:
        ax.set_ylabel('Economic Value (Billion USD)', fontsize=label_fs)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Legend subplot (bottom right)
ax_legend = fig4.add_subplot(gs4[2, 3])
ax_legend.axis('off')

# Create legend handles using Figure 2 colors
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#E2C044', label='Exposed Economic Value'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#426A5A', label='Exposed (With Risk-Avoidance)')
]

# Add legend
ax_legend.legend(handles=legend_handles, loc='center', fontsize=legend_fs, frameon=True, 
                facecolor='white', edgecolor='gray', title='Legend', title_fontsize=title_fs)

# Main title and save
fig4.suptitle('Figure 4. Climate Risk Exposure of Electricity Generation Value by Income Group and Region (2024–2050)', 
              fontsize=10, fontweight='bold', y=0.95)
fig4.savefig('outputs_processed_fig/fig4_alt1.pdf', bbox_inches='tight', dpi=300)

