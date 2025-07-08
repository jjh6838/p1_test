import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Oxford color guides
# https://communications.admin.ox.ac.uk/communications-resources/visual-identity/identity-guidelines/colours#:~:text=Do%20not%20use%20secondary%20colour,on%20top%20of%20Oxford%20blue

plt.rcParams['font.family'] = 'Arial'  # Or 'Helvetica'


# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_a_ember_2023_30.xlsx')

# Generate random percentages for 2023, 2030, and 2050
np.random.seed(42)  # For reproducibility

# Define custom ranges for each energy type and year
ranges_2023 = {
    "exp_hydro_2023": (10, 15),
    "exp_solar_2023": (20, 30),
    "exp_wind_2023": (15, 25),
    "exp_other_renewables_2023": (1, 10),
    "exp_nuclear_2023": (1, 8),
    "exp_fossil_2023": (1, 8),
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
percentages_2023 = {col: np.random.uniform(*rng) for col, rng in ranges_2023.items()}
percentages_2030 = {col: np.random.uniform(*rng) for col, rng in ranges_2030.items()}
percentages_2050 = {col: np.random.uniform(*rng) for col, rng in ranges_2050.items()}

# Generate random percentages for risk-avoidance planning for 2023, 2030, and 2050
# For 2023, risk-avoidance values are exactly the same
percentages_2023_risk_avoid = {
    f'exp_risk_avoid_{k.split("_", 1)[1]}': v for k, v in percentages_2023.items()
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
# Source: Sample data based on industry averages
conversion_rates = {
    "Hydro": 65,      # $65 per MWh
    "Solar": 40,      # $40 per MWh (utility-scale solar)
    "Wind": 38,       # $38 per MWh (onshore)
    "Other_Renewables": 85,  # $85 per MWh (biomass, geothermal)
    "Nuclear": 105,   # $105 per MWh
    "Fossil": 75,     # $75 per MWh (combined cycle natural gas)
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
        # Fix the key generation to handle spaces in energy type names properly
        exposure_col = f'exp_{energy_type.lower().replace(" ", "_")}_{year}'
        
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
        
        # Apply risk-avoidance exposure percentages
        exposure_col_risk_avoid = f'exp_risk_avoid_{energy_type.lower().replace(" ", "_")}_{year}'
 
        if year == 2023:
            pct_risk_avoid = percentages_2023_risk_avoid.get(exposure_col_risk_avoid, 0)
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
years = [2023, 2030, 2050]
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
# These totals represent the aggregated values across all energy types for each year (2023, 2030, 2050).

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

# === FIGURE 1: ENERGY (TWh) with custom 6 subplots layout ===
fig1 = plt.figure(figsize=(7.2, 4), dpi=300)
gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], wspace=0.1, hspace=0.35)

# Wide subplot for totals (spans (0,0) and (1,0))
ax1 = fig1.add_subplot(gs[:, 0])
ax1.set_axisbelow(True)
line_styles = ['-', '-', '-']  # solid, dashed, frequent dotted
#line_styles = [ (0, (1, 1)), (0, (2, 1)), '-']  # solid, dashed, frequent dotted

scenario_colors = ['#D9D8D6', '#D9D8D6', '#D9D8D6']
for i, (scenario, values) in enumerate(iea_scenarios.items()):
    ax1.plot(years, values, marker = 'o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, markersize=2, label=f'IEA {scenario}')
ax1.grid(True, alpha=0.1, linewidth=0.5)

# to indicate IEA scenarios, I will put numbers with frames on each line
for i, (scenario, values) in enumerate(iea_scenarios.items()):
    ax1.plot(years, values, linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, label=f'IEA {scenario}')

# Put numbered frames on IEA scenario lines with scenario text
# Map scenarios to their x-positions, numbers, and text positions
scenario_positions = {
    "Net Zero by 2050": (2041, 1, "left"),
    "Announced Pledges": (2042, 2, "left"), 
    "Stated Policies": (2043, 3, "below")
}

for scenario, values in iea_scenarios.items():
    x_pos, number, text_pos = scenario_positions[scenario]
    # Interpolate the y-value at the specified x position
    y_pos = np.interp(x_pos, years, values)
    ax1.text(x_pos, y_pos, f'{number}', ha='center', va='center', fontsize=annot_fs, color='#61615F',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#61615F', linewidth=0.5))
    
    # Add scenario text based on position
    if text_pos == "left":
        ax1.text(x_pos - 1.5, y_pos, scenario, ha='right', va='center', fontsize=annot_fs, color='#61615F')
    elif text_pos == "below":
        ax1.text(x_pos, y_pos - 3000, scenario.replace(" ", "\n"), ha='center', va='top', fontsize=annot_fs, color='#61615F')

ax1.plot(years, total_raw, 'o-', color='#61615F', linewidth=2.5, markersize=3, label='Non-Adjusted Total')
ax1.plot(years, total_exp, 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposure-Adjusted Total')
ax1.plot(years, risk_avoid_total, 'o-', color='#002147', linewidth=2.5, markersize=3, label='Risk-Avoidance Total')

for i, year in enumerate(years):
    # Annotate the Non-Adjusted Total line with values
    ax1.text(year, total_raw[i] - 1000, f'{total_raw[i]:,.0f}', ha='center', va='top', fontsize=annot_fs)
    
    # Annotate the Exposure-Adjusted Total line with values  
    ax1.text(year, total_exp[i] + 1000, f'{total_exp[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
    
    # Annotate the Risk-Avoidance Total line with values
    if year != 2023:  # Skip annotation for 2023 to avoid overlap
        ax1.text(year, risk_avoid_total[i] - 1000, f'{risk_avoid_total[i]:,.0f}', ha='center', va='top', fontsize=annot_fs)
    
    # Commented out: IEA scenario annotations
    # for j, (scenario, values) in enumerate(iea_scenarios.items()):
    #     ax1.text(year, values[i] * 1.01, f'{values[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)

# Set custom y-axis ticks with TWh labels and remove padding
ymin, ymid, ymax = 0, 40000, 80000
yticks = range(0, 90000, 10000)
ytick_labels = [f'{tick:.0f}' if tick in {ymin, ymid, ymax} else '' for tick in yticks]

# Set y-axis limits to remove padding
ax1.set_ylim([ymin - 2000, ymax + 2000])
ax1.set_xlim([2023 - 3, 2050 + 3])

ax1.set_yticks(yticks)
ax1.set_yticklabels(ytick_labels, verticalalignment='center')
ax1.tick_params(axis='y', which='major', pad=5)

# Styling
ax1.tick_params(axis='y', direction='out', length=3, labelrotation=90, labelsize=tick_fs)
ax1.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax1.set_xticks([2023, 2030, 2040, 2050])

# Legend with only the three main lines
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#61615F', label='Electricity Generation'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#AA1A2D', label='Exposed to Risks'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#002147', label='Exposed After Risk-Avoidance'),
]
ax1.legend(handles=handles, fontsize=legend_fs, loc='upper left', bbox_to_anchor=(-0.01, 1.01), frameon=False)

# Set title for the main total generation subplot
ax1.set_title('Projected Generation (TWh)', fontsize=title_fs, fontweight='bold', pad=5)

### Generate Six subplots for each energy type
# Mapping for subplot positions: (row, col): energy type
subplot_positions = {
    (0, 1): 'Solar',
    (0, 2): 'Wind',
    (0, 3): 'Hydro',
    (1, 1): 'Other_Renewables',
    (1, 2): 'Nuclear',
    (1, 3): 'Fossil',
}

ymin = 0
ymax = 100

for (row, col), etype in subplot_positions.items():
    ax = fig1.add_subplot(gs[row, col])
    bar_x = np.array([0, 1, 2])
    bar_width = 0.3
    spacing = 0.05
    
    # Calculate percentages relative to Non-Adjusted (100%)
    exposure_pct = [(type_values_dict[etype][i] / raw_type_values_dict[etype][i] * 100) if raw_type_values_dict[etype][i] > 0 else 0 for i in range(3)]
    risk_avoid_pct = [(risk_avoid_type_values_dict[etype][i] / raw_type_values_dict[etype][i] * 100) if raw_type_values_dict[etype][i] > 0 else 0 for i in range(3)]
    
    # Plot background bars with spacing
    ax.bar(bar_x - bar_width/2 - spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1, label='Non-Adjusted (100%)')
    ax.bar(bar_x + bar_width/2 + spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    
    # Plot percentage bars with spacing
    ax.bar(bar_x - bar_width/2 - spacing, exposure_pct, width=bar_width, color='#AA1A2D', alpha=1, label='Exposure-Adjusted %')
    ax.bar(bar_x + bar_width/2 + spacing, risk_avoid_pct, width=bar_width, color='#002147', alpha=1, label='Risk-Avoidance %')
    
    # Annotate percentage values
    for i in range(3):
        ax.text(bar_x[i] - bar_width/2 - spacing, exposure_pct[i] + 2, f'{exposure_pct[i]:.0f}%', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
        ax.text(bar_x[i] + bar_width/2 + spacing, risk_avoid_pct[i]+ 2, f'{risk_avoid_pct[i]:.0f}%', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
    
    # Set subplot title
    # Format the title based on energy type
    title_text = etype.replace('_', ' ') + ' (%)'
    ax.set_title(title_text, fontsize=title_fs, fontweight='bold', pad=5)
    
    ax.set_xticks(bar_x)
    ax.set_xticklabels(years, fontsize=tick_fs)

    ax.tick_params(axis='y', direction='out', length=3, labelsize=tick_fs)
    ax.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
    ax.yaxis.tick_right()
    
    # Set y-axis to percentage scale
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 50, 100])
    
    if (col == 3):
        ax.set_yticklabels(['0%', '50%', '100%'], va='center')
    else:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])


# fig1.legend(fontsize=legend_fs, loc='upper left', bbox_to_anchor=(0.15, 0.85))
fig1.suptitle('Global Electricity Generation (TWh) by Energy Type, Scenario, and Climate Risk Exposure (%) \nWith and Without Risk-Avoidance Planning (2023–2050)', fontsize=9, fontweight='bold', y=1.01)
fig1.savefig('energy_figure_a4_energy_disagg.png', bbox_inches='tight', dpi=300)
# plt.show()




# === FIGURE 2: ECONOMIC (USD) with custom 6 subplots layout ===
fig2 = plt.figure(figsize=(7.2, 4), dpi=300)
gs2 = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], wspace=0.1, hspace=0.35)

# Wide subplot for totals (spans (0,0) and (1,0))
ax3 = fig2.add_subplot(gs2[:, 0])
ax3.set_axisbelow(True)
line_styles = ['-', '-', '-']

scenario_colors = ['#D9D8D6', '#D9D8D6', '#D9D8D6']
for i, (scenario, values) in enumerate(iea_scenarios_usd.items()):
    ax3.plot(years, values, marker = 'o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, markersize=2, label=f'IEA {scenario}')
ax3.grid(True, alpha=0.1, linewidth=0.5)

# Put numbered frames on IEA scenario lines with scenario text
scenario_positions_usd = {
    "Net Zero by 2050": (2041, 1, "left"),
    "Announced Pledges": (2042, 2, "left"), 
    "Stated Policies": (2043, 3, "below")
}

for scenario, values in iea_scenarios_usd.items():
    x_pos, number, text_pos = scenario_positions_usd[scenario]
    y_pos = np.interp(x_pos, years, values)
    ax3.text(x_pos, y_pos, f'{number}', ha='center', va='center', fontsize=annot_fs, color='#61615F',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#61615F', linewidth=0.5))
    
    if text_pos == "left":
        ax3.text(x_pos - 1.5, y_pos, scenario, ha='right', va='center', fontsize=annot_fs, color='#61615F')
    elif text_pos == "below":
        ax3.text(x_pos, y_pos - 300, scenario.replace(" ", "\n"), ha='center', va='top', fontsize=annot_fs, color='#61615F')

ax3.plot(years, total_raw_usd, 'o-', color='#61615F', linewidth=2.5, markersize=3, label='Non-Adjusted Total')
ax3.plot(years, total_exp_usd, 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposure-Adjusted Total')
ax3.plot(years, risk_avoid_total_usd, 'o-', color='#002147', linewidth=2.5, markersize=3, label='Risk-Avoidance Total')

for i, year in enumerate(years):
    ax3.text(year, total_raw_usd[i] - 100, f'${total_raw_usd[i]:,.0f}B', ha='center', va='top', fontsize=annot_fs)
    ax3.text(year, total_exp_usd[i] + 100, f'${total_exp_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
    if year != 2023:
        ax3.text(year, risk_avoid_total_usd[i] - 100, f'${risk_avoid_total_usd[i]:,.0f}B', ha='center', va='top', fontsize=annot_fs)

# Set y-axis limits
ymin_usd, ymax_usd = 0, max(max(iea_scenarios_usd[scenario]) for scenario in iea_scenarios_usd)
ax3.set_ylim([ymin_usd - 200, ymax_usd + 200])
ax3.set_xlim([2023 - 3, 2050 + 3])

ax3.tick_params(axis='y', direction='out', length=3, labelrotation=90, labelsize=tick_fs)
ax3.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax3.set_xticks([2023, 2030, 2040, 2050])

# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#61615F', label='Economic Value'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#AA1A2D', label='Exposed to Risks'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#002147', label='Exposed After Risk-Avoidance'),
]
ax3.legend(handles=handles, fontsize=legend_fs, loc='upper left', bbox_to_anchor=(-0.01, 1.01), frameon=False)

ax3.set_title('Projected Economic Value (Billion USD)', fontsize=title_fs, fontweight='bold', pad=5)

# 6 subplots for each energy type (percentage charts, USD)
for (row, col), etype in subplot_positions.items():
    ax = fig2.add_subplot(gs2[row, col])
    bar_x = np.array([0, 1, 2])
    bar_width = 0.3
    spacing = 0.05
    
    # Calculate percentages relative to Non-Adjusted (100%)
    exposure_pct_usd = [(type_values_usd_dict[etype][i] / raw_type_values_usd_dict[etype][i] * 100) if raw_type_values_usd_dict[etype][i] > 0 else 0 for i in range(3)]
    risk_avoid_pct_usd = [(risk_avoid_type_values_usd_dict[etype][i] / raw_type_values_usd_dict[etype][i] * 100) if raw_type_values_usd_dict[etype][i] > 0 else 0 for i in range(3)]
    
    # Plot background bars
    ax.bar(bar_x - bar_width/2 - spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    ax.bar(bar_x + bar_width/2 + spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    
    # Plot percentage bars
    ax.bar(bar_x - bar_width/2 - spacing, exposure_pct_usd, width=bar_width, color='#AA1A2D', alpha=1)
    ax.bar(bar_x + bar_width/2 + spacing, risk_avoid_pct_usd, width=bar_width, color='#002147', alpha=1)
    
    # Annotate percentage values
    for i in range(3):
        ax.text(bar_x[i] - bar_width/2 - spacing, exposure_pct_usd[i] + 2, f'{exposure_pct_usd[i]:.0f}%', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
        ax.text(bar_x[i] + bar_width/2 + spacing, risk_avoid_pct_usd[i] + 2, f'{risk_avoid_pct_usd[i]:.0f}%', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
    
    # Set subplot title
    title_text = etype.replace('_', ' ') + ' (%)'
    ax.set_title(title_text, fontsize=title_fs, fontweight='bold', pad=5)
    
    ax.set_xticks(bar_x)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', direction='out', length=3, labelsize=tick_fs)
    ax.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
    ax.yaxis.tick_right()
    
    # Set y-axis to percentage scale
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 50, 100])
    
    if (col == 3):
        ax.set_yticklabels(['0%', '50%', '100%'], va='center')
    else:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])

fig2.suptitle('Global Electricity Generation Economic Value (USD) by Energy Type, Scenario, and Climate Risk Exposure (%) \nWith and Without Risk-Avoidance Planning (2023–2050)', fontsize=9, fontweight='bold', y=1.01)
fig2.savefig('energy_figure_a4_usd_disagg.png', bbox_inches='tight', dpi=300)
plt.show()

# Print comprehensive summary with all scenarios
print("=" * 80)
print("COMPREHENSIVE ENERGY SCENARIO COMPARISON")
print("=" * 80)

print(f"\nTOTAL GENERATION COMPARISON (2050):")
print(f"Actual Total Generation:     {total_raw[2]:>15,.1f} TWh (${total_raw_usd[2]:,.1f} billion)")
print(f"Exposure-Adjusted Total:     {total_exp[2]:>15,.1f} TWh (${total_exp_usd[2]:,.1f} billion)")
print(f"Risk-Avoidance Total:        {risk_avoid_total[2]:>15,.1f} TWh (${risk_avoid_total_usd[2]:,.1f} billion)")

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
renewable_raw = sum(raw_type_values_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other_Renewables"])
renewable_exp = sum(type_values_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other_Renewables"])
renewable_raw_usd = sum(raw_type_values_usd_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other_Renewables"])
renewable_exp_usd = sum(type_values_usd_dict[energy_type][2] for energy_type in ["Hydro", "Solar", "Wind", "Other_Renewables"])

share_renewable_raw = (renewable_raw / total_raw[2]) * 100
share_renewable_exp = (renewable_exp / total_exp[2]) * 100
share_renewable_raw_usd = (renewable_raw_usd / total_raw_usd[2]) * 100
share_renewable_exp_usd = (renewable_exp_usd / total_exp_usd[2]) * 100

print(f"  Renewable (Raw)       : {renewable_raw:>12,.1f} TWh ({share_renewable_raw:>5.1f}%) | ${renewable_raw_usd:>12,.1f}B ({share_renewable_raw_usd:>5.1f}%)")
print(f"  Renewable (Adjusted)   : {renewable_exp:>12,.1f} TWh ({share_renewable_exp:>5.1f}%) | ${renewable_exp_usd:>12,.1f}B ({share_renewable_exp_usd:>5.1f}%)")

# Detailed scenario comparison for 2050
detailed_scenario_comparison = pd.DataFrame({
    "Energy Type": energy_types,
    "IEA Stated Policies": [raw_type_values_dict[et][2] for et in energy_types],
    "IEA Announced Pledges": [raw_type_values_dict[et][2] for et in energy_types],
    "IEA Net Zero by 2050": [raw_type_values_dict[et][2] for et in energy_types],
})

detailed_scenario_comparison["IEA Stated Policies USD"] = detailed_scenario_comparison["IEA Stated Policies"] * [conversion_rates[et] for et in energy_types]
detailed_scenario_comparison["IEA Announced Pledges USD"] = detailed_scenario_comparison["IEA Announced Pledges"] * [conversion_rates[et] for et in energy_types]
detailed_scenario_comparison["IEA Net Zero by 2050 USD"] = detailed_scenario_comparison["IEA Net Zero by 2050"] * [conversion_rates[et] for et in energy_types]

# Display the detailed scenario comparison for 2050
print("\nDETAILED SCENARIO COMPARISON (2050):")
print(detailed_scenario_comparison.to_string(index=False, float_format="%.1f"))