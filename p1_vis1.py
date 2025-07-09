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

# === FIGURE 1: ENERGY (TWh) with custom 6 subplots layout ===
fig1 = plt.figure(figsize=(7.2, 4), dpi=300)
gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], wspace=0.1, hspace=0.4)

# Wide subplot for totals (spans (0,0) and (1,0))
ax1 = fig1.add_subplot(gs[:, 0])
ax1.set_axisbelow(True)
line_styles = ['-', '-', '-']  # solid, dashed, frequent dotted
#line_styles = [ (0, (1, 1)), (0, (2, 1)), '-']  # solid, dashed, frequent dotted

scenario_colors = ['#D9D8D6', '#D9D8D6', '#D9D8D6']
for i, (scenario, values) in enumerate(iea_scenarios.items()):
    ax1.plot(years, values, marker = 'o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.25, markersize=1.5, label=f'IEA {scenario}')
ax1.grid(True, alpha=0.1, linewidth=0.5)

# to indicate IEA scenarios, I will put numbers with frames on each line
for i, (scenario, values) in enumerate(iea_scenarios.items()):
    ax1.plot(years, values, linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, label=f'IEA {scenario}')

# Put numbered frames on IEA scenario lines with scenario text
# Map scenarios to their x-positions, numbers, and text positions
scenario_positions = {
    "Net Zero by 2050": (2041, 'a', "left"),
    "Announced Pledges": (2042, 'b', "left"), 
    "Stated Policies": (2043, 'c', "below")
}

for scenario, values in iea_scenarios.items():
    x_pos, number, text_pos = scenario_positions[scenario]
    # Interpolate the y-value at the specified x position
    y_pos = np.interp(x_pos, years, values)
    ax1.text(x_pos, y_pos, f'{number}', ha='center', va='center', fontsize=annot_fs, color='#61615F',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#61615F', linewidth=0.5))
    
    # Add scenario text based on position
    if text_pos == "left":
        ax1.text(x_pos - 1.25, y_pos, scenario, ha='right', va='center', fontsize=annot_fs, color='#61615F')
    elif text_pos == "below":
        ax1.text(x_pos, y_pos - 2800, scenario.replace(" ", "\n"), ha='center', va='top', fontsize=annot_fs, color='#61615F')

# Plot the main total generation lines
ax1.plot(years, total_raw, 'o-', color='#61615F', linewidth=2.5, markersize=3, label='Non-Adjusted Total')
ax1.plot(years, risk_avoid_total, 'o-', color='#002147', linewidth=2.5, markersize=3, label='Risk-Avoidance Total')
ax1.plot(years, total_exp, 'o-', color='#AA1A2D', linewidth=2.5, markersize=3, label='Exposure-Adjusted Total')

for i, year in enumerate(years):
    # Annotate the Non-Adjusted Total line with values
    ax1.text(year, total_raw[i] - 1000, f'{total_raw[i]:,.0f}', ha='center', va='top', fontsize=annot_fs)
    
    # Annotate the Exposure-Adjusted Total line with values  
    ax1.text(year, total_exp[i] + 1000, f'{total_exp[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)
    
    # Annotate the Risk-Avoidance Total line with values
    if year != 2024:  # Skip annotation for 2024 to avoid overlap
        ax1.text(year, risk_avoid_total[i] - 1000, f'{risk_avoid_total[i]:,.0f}', ha='center', va='top', fontsize=annot_fs)
    
    # Commented out: IEA scenario annotations
    # for j, (scenario, values) in enumerate(iea_scenarios.items()):
    #     ax1.text(year, values[i] * 1.01, f'{values[i]:,.0f}', ha='center', va='bottom', fontsize=annot_fs)

# Set custom y-axis ticks with TWh labels and remove padding
ymin, ymid, ymax = 0, 40000, 80000
yticks = range(0, 90000, 10000)
ax1.set_yticks(yticks)  # First set the tick positions

# Then set the tick labels
ytick_labels = [f'{tick:,.0f}' if tick in {ymin, ymid, ymax} else '' for tick in yticks]
ax1.set_yticklabels(ytick_labels, verticalalignment='center', fontsize=tick_fs, rotation=0)  


# Set y-axis limits to remove padding
ax1.set_ylim([ymin - 2000, ymax + 2000])
ax1.set_xlim([2024 - 3, 2050 + 3])

# Styling
ax1.tick_params(axis='y', pad=1) 
ax1.tick_params(axis='y', which='major', direction='out', length=3, labelrotation=90, labelsize=tick_fs)
ax1.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax1.set_xticks([2024, 2030, 2040, 2050])

# Legend with only the three main lines
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#61615F', label='Total Generation'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#AA1A2D', label='Exposed Generation'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#002147', label='Exposed (With Risk-Avoidance)'),
]
ax1.legend(handles=handles, fontsize=legend_fs, loc='upper left', bbox_to_anchor=(-0.01, 1.01), frameon=False)

# Set title for the main total generation subplot
ax1.set_title('Electricity\nGeneration (TWh)', fontsize=title_fs, fontweight='bold', pad=5)



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
    
    # Calculate absolute values in TWh for Exposure-Adjusted and Risk-Avoidance
    exposure_twh = type_values_dict[etype]
    risk_avoid_twh = risk_avoid_type_values_dict[etype]
    
    # Plot background bars with spacing
    ax.bar(bar_x - bar_width/2 - spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1, label='Non-Adjusted (100%)')
    ax.bar(bar_x + bar_width/2 + spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    
    # Plot percentage bars with spacing
    ax.bar(bar_x - bar_width/2 - spacing, exposure_pct, width=bar_width, color='#AA1A2D', alpha=1, label='Exposure-Adjusted %')
    ax.bar(bar_x[1:] + bar_width/2 + spacing, risk_avoid_pct[1:], width=bar_width, color='#002147', alpha=1, label='Risk-Avoidance %')
    
    # Annotate percentage values and differences
    for i in range(3):
        # Always show the exposure percentage
        ax.text(bar_x[i] - bar_width/2 - spacing, exposure_pct[i] + 2, f'{exposure_pct[i]:.0f}%', 
                ha='center', va='bottom', rotation=90, fontsize=annot_fs)
        
        # For risk avoidance, show "NA" or - mark for 2024 (index 0) and percentage for other years
        if i == 0:  # 2024
            ax.text(bar_x[i] + bar_width/2 + spacing, 1, "-", 
                    ha='center', va='bottom', rotation=0, fontsize=annot_fs)
        else:  # 2030 and 2050
            ax.text(bar_x[i] + bar_width/2 + spacing, risk_avoid_pct[i] + 2, f'{risk_avoid_pct[i]:.0f}%', 
                    ha='center', va='bottom', rotation=90, fontsize=annot_fs)
            
            # Calculate the difference in TWh between exposure and risk-avoidance scenarios
            # Positive diff_twh means risk-avoidance reduces exposure (Unexpected outcome)
            # Negative diff_twh means risk-avoidance increases exposure (Good outcome)
            diff_twh = - (exposure_twh[i] - risk_avoid_twh[i])
            
            # Calculate positions for arrow and text annotations
            arrow_x = bar_x[i] + bar_width/2 + spacing  # X position above the right bar (risk-avoidance)
            center_x = bar_x[i]  # Center position between the two bars for text placement
            arrow_mid_y = (exposure_pct[i] + risk_avoid_pct[i]) / 2 + 45  # Y position for text box
            arrow_start_y = arrow_mid_y - 10  # Arrow start position (above text)
            arrow_end_y = risk_avoid_pct[i] + 20  # Arrow end position (above risk-avoidance bar)
            
            # Draw arrow indicating the direction of change from exposure to risk-avoidance scenario
            if diff_twh > 0:
                # Risk-avoidance increases exposure: arrow pointing up (unexpected outcome)
                ax.annotate(
                    '', 
                    xy=(arrow_x, arrow_end_y), 
                    xytext=(arrow_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='<-', color='#AA1A2D', lw=1)
                )
            else:
                # Risk-avoidance decreases exposure: arrow pointing down (good outcome)
                ax.annotate(
                    '', 
                    xy=(arrow_x, arrow_start_y), 
                    xytext=(arrow_x, arrow_end_y),
                    arrowprops=dict(arrowstyle='<-', color='#002147', lw=1)
                )

            # Add text box showing the absolute change in TWh between scenarios
            ax.text(center_x, arrow_mid_y, f'{diff_twh:,.0f}\nTWh', 
                    ha='center', va='center', fontsize=annot_fs, color='#61615F',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=1, linewidth=0.5))


    # Set subplot title
    title_text = etype.replace('_', ' ') + '\nExposure (%)'
    ax.set_title(title_text, fontsize=title_fs, fontweight='bold', pad=5)
    
    ax.set_xticks(bar_x)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', pad=1) 
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
fig1.suptitle('Figure 1. Climate Risk Exposure of Electricity Generation (2024–2050), by Energy Type', fontsize=9, fontweight='bold', y=1.01, x=0.1, ha='left')
fig1.savefig('outputs_processed_vis/vis1a.pdf', bbox_inches='tight', dpi=300)





# === FIGURE 2: ECONOMIC (USD) with custom 6 subplots layout ===
fig2 = plt.figure(figsize=(7.2, 4), dpi=300)
gs2 = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], wspace=0.1, hspace=0.4)

# Wide subplot for totals (spans (0,0) and (1,0))
ax3 = fig2.add_subplot(gs2[:, 0])
ax3.set_axisbelow(True)
line_styles = ['-', '-', '-']

scenario_colors = ['#D9D8D6', '#D9D8D6', '#D9D8D6']
for i, (scenario, values) in enumerate(iea_scenarios_usd.items()):
    ax3.plot(years, values, marker='o', linestyle=line_styles[i], color=scenario_colors[i], 
             linewidth=1.5, markersize=2, label=f'IEA {scenario}')
ax3.grid(True, alpha=0.1, linewidth=0.5)

# Put numbered frames on IEA scenario lines with scenario text
scenario_positions_usd = {
    "Net Zero by 2050": (2041, 'a', "left"),
    "Announced Pledges": (2042, 'b', "left"), 
    "Stated Policies": (2043, 'c', "below")
}

for scenario, values in iea_scenarios_usd.items():
    x_pos, number, text_pos = scenario_positions_usd[scenario]
    y_pos = np.interp(x_pos, years, values)
    ax3.text(x_pos, y_pos, f'{number}', ha='center', va='center', fontsize=annot_fs, color='#61615F',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#61615F', linewidth=0.5))
    
    if text_pos == "left":
        ax3.text(x_pos - 1.25, y_pos, scenario, ha='right', va='center', fontsize=annot_fs, color='#61615F')
    elif text_pos == "below":
        ax3.text(x_pos, y_pos - 300, scenario.replace(" ", "\n"), ha='center', va='top', fontsize=annot_fs, color='#61615F')

ax3.plot(years, total_raw_usd, 'o-', color='#61615F', linewidth=2.5, markersize=3, label='Non-Adjusted Total')
ax3.plot(years, risk_avoid_total_usd, 'o-', color='#426A5A', linewidth=2.5, markersize=3, label='Risk-Avoidance Total')
ax3.plot(years, total_exp_usd, 'o-', color='#E2C044', linewidth=2.5, markersize=3, label='Exposure-Adjusted Total')

for i, year in enumerate(years):
    ax3.text(year, total_raw_usd[i] - 100, f'${total_raw_usd[i]:,.0f}B', ha='center', va='top', fontsize=annot_fs)
    ax3.text(year, total_exp_usd[i] + 100, f'${total_exp_usd[i]:,.0f}B', ha='center', va='bottom', fontsize=annot_fs)
    if year != 2024:
        ax3.text(year, risk_avoid_total_usd[i] - 100, f'${risk_avoid_total_usd[i]:,.0f}B', ha='center', va='top', fontsize=annot_fs)

# Set y-axis limits
ymin_usd, ymid_usd, ymax_usd = 0, 3000, 6000
yticks_usd = range(0, 7000, 1000)
ax3.set_yticks(yticks_usd)  # First set the tick positions

# Then set the tick labels
ytick_labels_usd = [f'{tick:,.0f}' if tick in {ymin_usd, ymid_usd, ymax_usd} else '' for tick in yticks_usd]
ax3.set_yticklabels(ytick_labels_usd, verticalalignment='center', fontsize=tick_fs, rotation=0)

ax3.set_ylim([ymin_usd - 200, ymax_usd + 200])
ax3.set_xlim([2024 - 3, 2050 + 3])

ax3.tick_params(axis='y', pad=1) 
ax3.tick_params(axis='y', direction='out', length=3, labelrotation=90, labelsize=tick_fs)
ax3.tick_params(axis='x', which='both', length=0, labelsize=tick_fs)
ax3.set_xticks([2024, 2030, 2040, 2050])


# Set custom y-axis ticks with TWh labels and remove padding
ymin, ymid, ymax = 0, 40000, 80000
yticks = range(0, 90000, 10000)
ax1.set_yticks(yticks)  # First set the tick positions

# Then set the tick labels
ytick_labels = [f'{tick:,.0f}' if tick in {ymin, ymid, ymax} else '' for tick in yticks]
ax1.set_yticklabels(ytick_labels, verticalalignment='center', fontsize=tick_fs, rotation=0)  


# Set y-axis limits to remove padding
ax1.set_ylim([ymin - 2000, ymax + 2000])
ax1.set_xlim([2024 - 3, 2050 + 3])

# Styling
ax1.tick_params(axis='y', pad=1) 



# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#61615F', label='Total Economic Value'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#E2C044', label='Exposed Economic Value'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#426A5A', label='Exposed (With Risk-Avoidance)'),
]
ax3.legend(handles=handles, fontsize=legend_fs, loc='upper left', bbox_to_anchor=(-0.01, 1.01), frameon=False)

ax3.set_title('Economic Value\n(Billion USD)', fontsize=title_fs, fontweight='bold', pad=5)




# 6 subplots for each energy type (percentage charts, USD)
for (row, col), etype in subplot_positions.items():
    ax = fig2.add_subplot(gs2[row, col])
    bar_x = np.array([0, 1, 2])
    bar_width = 0.3
    spacing = 0.05
    
    # Calculate percentages relative to Non-Adjusted (100%)
    exposure_pct_usd = [(type_values_usd_dict[etype][i] / raw_type_values_usd_dict[etype][i] * 100) if raw_type_values_usd_dict[etype][i] > 0 else 0 for i in range(3)]
    risk_avoid_pct_usd = [(risk_avoid_type_values_usd_dict[etype][i] / raw_type_values_usd_dict[etype][i] * 100) if raw_type_values_usd_dict[etype][i] > 0 else 0 for i in range(3)]
    
    # Calculate absolute values in billion USD for Exposure-Adjusted and Risk-Avoidance
    exposure_usd = type_values_usd_dict[etype]
    risk_avoid_usd = risk_avoid_type_values_usd_dict[etype]
    
    # Plot background bars
    ax.bar(bar_x - bar_width/2 - spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    ax.bar(bar_x + bar_width/2 + spacing, [100, 100, 100], width=bar_width, color='#F2F0F0', alpha=1)
    
    # Plot percentage bars
    ax.bar(bar_x - bar_width/2 - spacing, exposure_pct_usd, width=bar_width, color='#E2C044', alpha=1)
    ax.bar(bar_x[1:] + bar_width/2 + spacing, risk_avoid_pct_usd[1:], width=bar_width, color='#426A5A', alpha=1)
    

    # Annotate percentage values and differences
    for i in range(3):
        # Always show the exposure percentage
        ax.text(bar_x[i] - bar_width/2 - spacing, exposure_pct_usd[i] + 2, f'{exposure_pct_usd[i]:.0f}%', 
                ha='center', va='bottom', rotation=90, fontsize=annot_fs)
        
        # For risk avoidance, show "NA" or - mark for 2024 (index 0) and percentage for other years
        if i == 0:  # 2024
            ax.text(bar_x[i] + bar_width/2 + spacing, 1, "-", 
                    ha='center', va='bottom', rotation=0, fontsize=annot_fs)
        else:  # 2030 and 2050
            ax.text(bar_x[i] + bar_width/2 + spacing, risk_avoid_pct_usd[i] + 2, f'{risk_avoid_pct_usd[i]:.0f}%', 
                    ha='center', va='bottom', rotation=90, fontsize=annot_fs)
            
            # Calculate the difference in billion USD between exposure and risk-avoidance scenarios
            # Positive diff_usd means risk-avoidance reduces exposure (Unexpected outcome)
            # Negative diff_usd means risk-avoidance increases exposure (Good outcome)
            diff_usd = - (exposure_usd[i] - risk_avoid_usd[i])
            
            # Calculate positions for arrow and text annotations
            arrow_x = bar_x[i] + bar_width/2 + spacing  # X position above the right bar (risk-avoidance)
            center_x = bar_x[i]  # Center position between the two bars for text placement
            arrow_mid_y = (exposure_pct_usd[i] + risk_avoid_pct_usd[i]) / 2 + 45 - 5  # Y position for text box
            arrow_start_y = (arrow_mid_y + 5) - 10  # Arrow start position (above text)
            arrow_end_y = risk_avoid_pct_usd[i] + 20  # Arrow end position (above risk-avoidance bar)
            
            # Draw arrow indicating the direction of change from exposure to risk-avoidance scenario
            if diff_usd > 0:
                # Risk-avoidance increases exposure: arrow pointing up (unexpected outcome)
                ax.annotate(
                    '', 
                    xy=(arrow_x, arrow_end_y), 
                    xytext=(arrow_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='<-', color='#E2C044', lw=1)
                )
            else:
                # Risk-avoidance decreases exposure: arrow pointing down (good outcome)
                ax.annotate(
                    '', 
                    xy=(arrow_x, arrow_start_y), 
                    xytext=(arrow_x, arrow_end_y),
                    arrowprops=dict(arrowstyle='<-', color='#426A5A', lw=1)
                )

            # Add text box showing the absolute change in billion USD between scenarios
            ax.text(center_x, arrow_mid_y, f'${diff_usd:,.0f}B', 
                    ha='center', va='center', fontsize=annot_fs, color='#61615F',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=1, linewidth=0.5))

    # Set subplot title
    title_text = etype.replace('_', ' ') + '\nExposure (%)'
    ax.set_title(title_text, fontsize=title_fs, fontweight='bold', pad=5)
    
    
    ax.set_xticks(bar_x)
    ax.set_xticklabels(years, fontsize=tick_fs)
    ax.tick_params(axis='y', pad=1) 
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
        
fig2.suptitle('Figure 2. Climate Risk Exposure of Electricity Generation Value (2024–2050), by Energy Type', fontsize=9, fontweight='bold', y=1.01, x=0.1, ha='left')
fig2.savefig('outputs_processed_vis/vis1b.pdf', bbox_inches='tight', dpi=300)


# === FIGURE 3: Custom 11 subplots (2024–2050) ===
# This figure will have 11 subplots, so use 3 rows and 4 columns. 
# The first row will have 4 subplots indicating four income groups. 
# The second row will have 4 subplots indicating first four regional groups.
# The third row will have 3 subplots indicating the last three regional groups. The last subplot will be empty, but you can use it for legend.
# Each subplot will show the total generation TWh and Billion USD for each country group over 2024 through 2050.

# Define country groups
income_groups = ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']
regional_groups = ['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean', 
                  'Middle East & North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']

# Generate sample data for each group (you can replace this with actual data processing)
np.random.seed(100)  # For reproducibility

# Create sample data for income groups
income_data_twh = {}
income_data_usd = {}
for group in income_groups:
    # Generate increasing values over time with some variation
    base_twh = np.random.uniform(5000, 15000)
    growth_rate = np.random.uniform(1.2, 1.8)
    income_data_twh[group] = [base_twh * (growth_rate ** i) for i in range(3)]
    
    # Convert to USD using weighted average conversion rate
    weighted_rate = 75  # Sample average rate
    income_data_usd[group] = [val * weighted_rate / 1000000000 for val in income_data_twh[group]]
    income_data_twh[group] = [val / 1000000 for val in income_data_twh[group]]  # Convert to TWh

# Create sample data for regional groups
regional_data_twh = {}
regional_data_usd = {}
for group in regional_groups:
    base_twh = np.random.uniform(3000, 12000)
    growth_rate = np.random.uniform(1.1, 1.9)
    regional_data_twh[group] = [base_twh * (growth_rate ** i) for i in range(3)]
    
    weighted_rate = 75
    regional_data_usd[group] = [val * weighted_rate / 1000000000 for val in regional_data_twh[group]]
    regional_data_twh[group] = [val / 1000000 for val in regional_data_twh[group]]

# Create Figure 3
fig3 = plt.figure(figsize=(7.2, 6), dpi=300)
gs3 = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

# Plot income groups (first row)
for i, group in enumerate(income_groups):
    ax = fig3.add_subplot(gs3[0, i])
    
    # Plot TWh and USD on dual y-axis
    ax2 = ax.twinx()
    
    # Plot TWh (left axis)
    line1 = ax.plot(years, income_data_twh[group], 'o-', color='#AA1A2D', linewidth=2, markersize=3, label='Generation (TWh)')
    
    # Plot USD (right axis)
    line2 = ax2.plot(years, income_data_usd[group], 's-', color='#E2C044', linewidth=2, markersize=3, label='Value (Billion USD)')
    
    # Annotations
    for j, year in enumerate(years):
        ax.text(year, income_data_twh[group][j] * 1.05, f'{income_data_twh[group][j]:,.0f}', 
                ha='center', va='bottom', fontsize=annot_fs-1, color='#AA1A2D')
        ax2.text(year, income_data_usd[group][j] * 0.95, f'${income_data_usd[group][j]:,.0f}B', 
                 ha='center', va='top', fontsize=annot_fs-1, color='#E2C044')
    
    # Styling
    ax.set_title(group, fontsize=title_fs, fontweight='bold', pad=5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=tick_fs-1)
    ax.tick_params(axis='y', labelsize=tick_fs-1, colors='#AA1A2D')
    ax2.tick_params(axis='y', labelsize=tick_fs-1, colors='#E2C044')
    ax.tick_params(axis='x', which
