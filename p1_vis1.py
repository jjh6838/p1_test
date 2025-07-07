import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_a_ember_2023_30.xlsx')

# Generate random percentages for 2023, 2030, and 2050
np.random.seed(42)  # For reproducibility

# Define custom ranges for each energy type and year
ranges_2023 = {
    "exp_hydro_2023": (10, 20),
    "exp_solar_2023": (20, 30),
    "exp_wind_2023": (15, 25),
    "exp_other_renewables_2023": (0, 10),
    "exp_nuclear_2023": (0, 5),
    "exp_fossil_2023": (0, 5),
}
ranges_2030 = {
    "exp_hydro_2030": (10, 25),
    "exp_solar_2030": (25, 35),
    "exp_wind_2030": (25, 35),
    "exp_other_renewables_2030": (0, 15),
    "exp_nuclear_2030": (0, 5),
    "exp_fossil_2030": (0, 5),
}
ranges_2050 = {
    "exp_hydro_2050": (15, 30),
    "exp_solar_2050": (25, 40),
    "exp_wind_2050": (30, 40),
    "exp_other_renewables_2050": (0, 15),
    "exp_nuclear_2050": (0, 5),
    "exp_fossil_2050": (0, 5),
}

# Generate random percentages within the specified ranges
percentages_2023 = {col: np.random.uniform(*rng) for col, rng in ranges_2023.items()}
percentages_2030 = {col: np.random.uniform(*rng) for col, rng in ranges_2030.items()}
percentages_2050 = {col: np.random.uniform(*rng) for col, rng in ranges_2050.items()}

# Map energy types to actual column patterns in your data
column_mapping = {
    "Hydro": "Hydro_",
    "Solar": "Solar_",
    "Wind": "Wind_",
    "Other Renewables": "Other Renewables_",
    "Nuclear": "Nuclear_",
    "Fossil": "Fossil_"
}

# Sample conversion rates from MWh to USD by energy type
# Based on average levelized cost of electricity (LCOE) in USD/MWh
# Source: Sample data based on industry averages
conversion_rates = {
    "Hydro": 65,      # $65 per MWh
    "Solar": 40,      # $40 per MWh (utility-scale solar)
    "Wind": 38,       # $38 per MWh (onshore)
    "Other Renewables": 85,  # $85 per MWh (biomass, geothermal)
    "Nuclear": 105,   # $105 per MWh
    "Fossil": 75,     # $75 per MWh (combined cycle natural gas)
}

# Calculate values by each type and year, then apply the exposure percentages
type_values_dict = {}  # Exposure-adjusted values
raw_type_values_dict = {}  # Non-exposure-adjusted values
type_values_usd_dict = {}  # Exposure-adjusted values in USD
raw_type_values_usd_dict = {}  # Non-exposure-adjusted values in USD

for energy_type, pattern in column_mapping.items():
    raw_type_values_dict[energy_type] = []
    type_values_dict[energy_type] = []
    raw_type_values_usd_dict[energy_type] = []
    type_values_usd_dict[energy_type] = []
    
    for year in [2023, 2030, 2050]:
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
        exposure_col = f'exp_{energy_type.lower()}_{year}'
        if year == 2023:
            pct = percentages_2023.get(exposure_col, 0)
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

# Prepare data for plotting
years = [2023, 2030, 2050]
# Reorder energy types as requested: Fossil, Nuclear, Other Renewable, Hydro, Wind, and Solar
energy_types = ["Fossil", "Nuclear", "Other Renewables", "Hydro", "Wind", "Solar"]

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
    type_values_dict[energy_type] = [val / 1000000 for val in type_values_dict[energy_type]]
    raw_type_values_dict[energy_type] = [val / 1000000 for val in raw_type_values_dict[energy_type]]
    type_values_usd_dict[energy_type] = [val / 1000000000 for val in type_values_usd_dict[energy_type]]
    raw_type_values_usd_dict[energy_type] = [val / 1000000000 for val in raw_type_values_usd_dict[energy_type]]

for scenario in iea_scenarios:
    iea_scenarios[scenario] = [val / 1000000 for val in iea_scenarios[scenario]]
    iea_scenarios_usd[scenario] = [val / 1000000000 for val in iea_scenarios_usd[scenario]]

# Calculate totals for line graphs
total_raw = [sum(raw_type_values_dict[et][i] for et in energy_types) for i in range(3)]
total_exp = [sum(type_values_dict[et][i] for et in energy_types) for i in range(3)]
total_raw_usd = [sum(raw_type_values_usd_dict[et][i] for et in energy_types) for i in range(3)]
total_exp_usd = [sum(type_values_usd_dict[et][i] for et in energy_types) for i in range(3)]

# Create figure with four subplots (2x2 grid), A4 landscape size, sharing y-axis within rows
fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), sharey='row')

# Correctly unpack the 2D array of axes
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# Colors for energy types (colorblind-friendly palette)
colors = ['#76b7b2', '#e15759', '#57606f', '#1170aa', '#a3acb2', '#fc7d0b']

# Font sizes
title_fs = 11
label_fs = 10
tick_fs = 9
legend_fs = 9
annot_fs = 8

# SUBPLOT 1: Line Graph for Totals and Reference Scenarios (Energy - TWh)
ax1.set_axisbelow(True)
ax1.grid(True, alpha=0.3)
ax1.plot(years, total_exp, 'o-', color='blue', linewidth=2, markersize=5, label='Exposure-Adjusted Total')
ax1.plot(years, total_raw, 's--', color='green', linewidth=2, markersize=5, label='Non-Adjusted Total')
line_styles = ['-', '--', '-.']
scenario_colors = ['#AAAAAA', '#777777', '#444444']

# Plot IEA scenarios WITH labels for the legend
for i, (scenario, values) in enumerate(iea_scenarios.items()):
    ax1.plot(years, values, marker='o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, markersize=4, label=f'IEA {scenario}')  # Add label parameter

# Add text annotations only for your main lines, not IEA scenarios
for i, year in enumerate(years):
    ax1.text(year, total_exp[i] * 1.01, f'{total_exp[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
    ax1.text(year, total_raw[i] * 1.01, f'{total_raw[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
    # Text annotations for IEA scenarios
    for j, (scenario, values) in enumerate(iea_scenarios.items()):
        ax1.text(year, values[i] * 1.01, f'{values[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)

ax1.set_title('Totals (TWh)', fontsize=title_fs, fontweight='bold', pad=10)
ax1.set_ylabel('TWh', fontsize=label_fs)
ax1.set_xlabel('', fontsize=label_fs)  # Remove "Year" label
ax1.ticklabel_format(style='plain', axis='y')
ax1.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax1.tick_params(axis='y', labelsize=tick_fs)
ax1.legend(loc='upper left', fontsize=legend_fs)  

# SUBPLOT 2: Stacked Bar Graph for Energy Mix (Energy - TWh)
ax2.set_axisbelow(True)
ax2.grid(True, axis='y', alpha=0.3)
x = np.arange(len(years))
width = 0.35
exp_bottom = np.zeros(len(years))
for i, energy_type in enumerate(energy_types):
    values = type_values_dict[energy_type]
    ax2.bar(x - width/2, values, width, bottom=exp_bottom, color=colors[i], label=f'{energy_type}')
    exp_bottom += values
# Update text annotations without decimal points and with comma separators
for i in range(len(years)):
    ax2.text(x[i] - width/2, total_exp[i] * 1.01, f'{total_exp[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
    ax2.text(x[i] + width/2, total_raw[i] * 1.01, f'{total_raw[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
raw_bottom = np.zeros(len(years))
for i, energy_type in enumerate(energy_types):
    values = raw_type_values_dict[energy_type]
    # Use alpha=0.7 only, no hatch
    ax2.bar(x + width/2, values, width, bottom=raw_bottom, color=colors[i], alpha=0.7)
    raw_bottom += values
for i in range(len(years)):
    ax2.text(x[i] + width/2, total_raw[i] * 1.01, f'{total_raw[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
ax2.set_title('Mix (TWh)', fontsize=title_fs, fontweight='bold', pad=10)
ax2.set_ylabel('TWh', fontsize=label_fs)
ax2.set_xlabel('', fontsize=label_fs)  # Remove "Year" label
ax2.set_xticks(x)
ax2.set_xticklabels(years, fontsize=tick_fs)
ax2.tick_params(axis='x', which='both', length=0)
ax2.tick_params(axis='y', labelsize=tick_fs)
ax2.ticklabel_format(style='plain', axis='y')

# Legend for ax2 (subplot 0,1) is already at upper right, but for clarity:
from matplotlib.patches import Rectangle
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(energy_types))]
handles.extend([
    Rectangle((0, 0), 1, 1, color='white', label='Exposure-Adjusted (Left Bar)'),
    Rectangle((0, 0), 1, 1, color='white', alpha=0.7, label='Non-Adjusted (Right Bar)')  # No hatch
])
labels = energy_types + ['Exposure-Adjusted (Left Bar)', 'Non-Adjusted (Right Bar)']
ax2.legend(handles, labels, fontsize=legend_fs, loc='upper right')

# SUBPLOT 3: Line Graph for Totals and Reference Scenarios (USD - Billions)
ax3.set_axisbelow(True)
ax3.grid(True, alpha=0.3)
ax3.plot(years, total_exp_usd, 'o-', color='blue', linewidth=2, markersize=5, label='Exposure-Adjusted Total')
ax3.plot(years, total_raw_usd, 's--', color='green', linewidth=2, markersize=5, label='Non-Adjusted Total')

# Plot IEA scenarios WITH labels (will be ignored since we won't show ax3 legend)
for i, (scenario, values) in enumerate(iea_scenarios_usd.items()):
    ax3.plot(years, values, marker='o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, markersize=4, label=f'IEA {scenario}')

# Add text annotations only for your main lines, not IEA scenarios
for i, year in enumerate(years):
    ax3.text(year, total_exp_usd[i] * 1.01, f'${total_exp_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
    ax3.text(year, total_raw_usd[i] * 1.01, f'${total_raw_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
    # Text annotations for IEA scenarios
    for j, (scenario, values) in enumerate(iea_scenarios_usd.items()):
        ax3.text(year, values[i] * 1.01, f'${values[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
ax3.set_title('Totals (USD)', fontsize=title_fs, fontweight='bold', pad=10)
ax3.set_ylabel('Billion USD', fontsize=label_fs)
ax3.set_xlabel('', fontsize=label_fs)  # Remove "Year" label
ax3.ticklabel_format(style='plain', axis='y')
ax3.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax3.tick_params(axis='y', labelsize=tick_fs)
# REMOVE legend for ax3
# ax3.legend(loc='lower right', fontsize=legend_fs)  # REMOVED

# SUBPLOT 4: Stacked Bar Graph for Energy Mix (USD - Billions)
# Update the same way for ax4
exp_bottom_usd = np.zeros(len(years))
for i, energy_type in enumerate(energy_types):
    values = type_values_usd_dict[energy_type]
    ax4.bar(x - width/2, values, width, bottom=exp_bottom_usd, color=colors[i], label=f'{energy_type}')
    exp_bottom_usd += values
# Update text annotations without decimal points and with comma separators
for i in range(len(years)):
    ax4.text(x[i] - width/2, total_exp_usd[i] * 1.01, f'${total_exp_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
    ax4.text(x[i] + width/2, total_raw_usd[i] * 1.01, f'${total_raw_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
raw_bottom_usd = np.zeros(len(years))
for i, energy_type in enumerate(energy_types):
    values = raw_type_values_usd_dict[energy_type]
    # Use alpha=0.7 only, no hatch
    ax4.bar(x + width/2, values, width, bottom=raw_bottom_usd, color=colors[i], alpha=0.7)
    raw_bottom_usd += values
for i in range(len(years)):
    ax4.text(x[i] + width/2, total_raw_usd[i] * 1.01, f'${total_raw_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
ax4.set_title('Mix (USD)', fontsize=title_fs, fontweight='bold', pad=10)
ax4.set_ylabel('', fontsize=label_fs)  # Already empty due to sharey='row'
ax4.set_xlabel('', fontsize=label_fs)  # Remove "Year" label
ax4.set_xticks(x)
ax4.set_xticklabels(years, fontsize=tick_fs)
ax4.tick_params(axis='x', which='both', length=0)
ax4.tick_params(axis='y', labelsize=tick_fs)
ax4.ticklabel_format(style='plain', axis='y')

# Since axes now share y-axis by row, we should remove the y-axis labels and ticks from the right plots
ax2.set_ylabel('')  # Remove y-axis label from the right plot in the top row
ax4.set_ylabel('')  # Remove y-axis label from the right plot in the bottom row

# REMOVE the global legend
# fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=legend_fs)  # REMOVED

# Add a main title for the entire figure
fig.suptitle('Global Energy Generation Analysis: 2023-2050 Projections\nEnergy (TWh) and Economic (USD) Comparison', fontsize=12, fontweight='bold', y=0.99)

# Adjust the tight_layout rect - since we removed the bottom legend, we can reduce the bottom margin
plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Reduced bottom margin from 0.05 to 0.02

# Save to file with tight bounding box
plt.savefig('energy_figure_a4.png', bbox_inches='tight', dpi=300)
plt.show()

# Print comprehensive summary with all scenarios
print("=" * 80)
print("COMPREHENSIVE ENERGY SCENARIO COMPARISON")
print("=" * 80)

print(f"\nTOTAL GENERATION COMPARISON (2050):")
print(f"Actual Total Generation:     {total_raw[2]:>15,.1f} TWh (${total_raw_usd[2]:,.1f} billion)")
print(f"Exposure-Adjusted Total:     {total_exp[2]:>15,.1f} TWh (${total_exp_usd[2]:,.1f} billion)")

for scenario, values in iea_scenarios.items():
    print(f"IEA {scenario:<20}: {values[2]:>15,.1f} TWh (${iea_scenarios_usd[scenario][2]:,.1f} billion)")

print(f"\nACTUAL ENERGY MIX BREAKDOWN (2050):")
for energy_type in energy_types:
    share = (raw_type_values_dict[energy_type][2] / total_raw[2]) * 100
    share_usd = (raw_type_values_usd_dict[energy_type][2] / total_raw_usd[2]) * 100
    print(f"  {energy_type:<18}: {raw_type_values_dict[energy_type][2]:>12,.1f} TWh ({share:>5.1f}%) | ${raw_type_values_usd_dict[energy_type][2]:>12,.1f}B ({share_usd:>5.1f}%)")

print(f"\nEXPOSURE-ADJUSTED ENERGY MIX BREAKDOWN (2050):")
for energy_type in energy_types:
    share = (type_values_dict[energy_type][2] / total_exp[2]) * 100
    share_usd = (type_values_usd_dict[energy_type][2] / total_exp_usd[2]) * 100
    print(f"  {energy_type:<18}: {type_values_dict[energy_type][2]:>12,.1f} TWh ({share:>5.1f}%) | ${type_values_usd_dict[energy_type][2]:>12,.1f}B ({share_usd:>5.1f}%)")

print(f"\nRENEWABLE SHARE COMPARISON (2050):")
renewable_raw = sum(raw_type_values_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other Renewables"])
renewable_exp = sum(type_values_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other Renewables"])
renewable_raw_usd = sum(raw_type_values_usd_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other Renewables"])
renewable_exp_usd = sum(type_values_usd_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other Renewables"])

print(f"Actual Renewable Share:      {(renewable_raw / total_raw[2]) * 100:.1f}% (Energy) | {(renewable_raw_usd / total_raw_usd[2]) * 100:.1f}% (USD)")
print(f"Exposure-Adjusted Share:     {(renewable_exp / total_exp[2]) * 100:.1f}% (Energy) | {(renewable_exp_usd / total_exp_usd[2]) * 100:.1f}% (USD)")

print("\nCONVERSION RATES (USD/MWh):")
for energy_type in energy_types:
    print(f"  {energy_type:<18}: ${conversion_rates[energy_type]:>5}")

print("=" * 80)