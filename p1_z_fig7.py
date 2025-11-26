import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Oxford color scheme
plt.rcParams['font.family'] = 'Arial'

# Load the processed data
print("Loading data from p1_y_results_data_etl.xlsx...")
data = pd.read_excel('outputs_processed_data/p1_y_results_data_etl.xlsx')

print(f"Loaded {len(data):,} rows")
print(f"Years: {sorted(data['year'].unique())}")
print(f"Energy types: {sorted(data['energy_type'].unique())}")

# Filter to only Solar, Wind, and Hydro
energy_types = ['solar', 'wind', 'hydro']
data_filtered = data[data['energy_type'].isin(energy_types)].copy()

print(f"\nFiltered to {len(data_filtered):,} rows for Solar, Wind, Hydro")

# Create figure with 3 rows × 6 columns
fig = plt.figure(figsize=(24, 12))
gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)

# Color schemes (Oxford colors)
bar_colors = ['#AA1A2D', '#002147']  # Red for exposed, Oxford Blue for risk-avoided
cmap_exp = 'Reds'  # Red scale for exposure
cmap_risk = 'Blues'  # Blue scale for risk avoidance

# Process each energy type (row)
for row_idx, energy_type in enumerate(energy_types):
    print(f"\nProcessing {energy_type.upper()}...")
    
    # Filter data for this energy type
    energy_data = data_filtered[data_filtered['energy_type'] == energy_type].copy()
    
    # ========== 2030 CHARTS (Columns 0-2) ==========
    year = 2030
    year_data = energy_data[energy_data['year'] == year].copy()
    
    # COLUMN 0: Bar chart (100% & 0km) - exp_MWh vs exp_risk_avoid_MWh
    ax_bar_2030 = fig.add_subplot(gs[row_idx, 0])
    
    # Get totals for 100% supply, 0km buffer
    bar_data = year_data[
        (year_data['supply_scenario'] == '100%') & 
        (year_data['hazard_buffer'] == '0km')
    ]
    
    total_exp = bar_data['exp_MWh'].sum() / 1e6  # Convert to TWh
    total_risk = bar_data['exp_risk_avoid_MWh'].sum() / 1e6  # Convert to TWh
    
    bars = ax_bar_2030.bar(['Exposed', 'Exposed with\nRisk-Avoidance'], [total_exp, total_risk], 
                           color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1)
    ax_bar_2030.set_ylabel('', fontsize=9, fontweight='bold')
    ax_bar_2030.set_title(f'{energy_type.capitalize()} {year}\n100% Supply, 0km Buffer', 
                          fontsize=10, fontweight='bold')
    ax_bar_2030.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar_2030.set_ylim(0, max(total_exp, total_risk) * 1.2)
    ax_bar_2030.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_bar_2030.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # COLUMN 1: Heatmap for exp_MWh (2030)
    ax_heat1_2030 = fig.add_subplot(gs[row_idx, 1])
    
    # Create pivot table for heatmap
    pivot_exp = year_data.pivot_table(
        values='exp_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    # Reorder: 100% on left, 60% on right; 40km on top, 0km on bottom
    supply_order = ['100%', '90%', '80%', '70%', '60%']
    buffer_order = ['40km', '30km', '20km', '10km', '0km']
    pivot_exp = pivot_exp.reindex(index=buffer_order, columns=supply_order)
    
    # Convert to TWh for display
    pivot_exp_twh = pivot_exp / 1e6
    
    # Create custom colormap: start with bar color for max value
    from matplotlib.colors import LinearSegmentedColormap
    colors_exp = ['#FFFFFF', '#FFCCCC', '#FF9999', '#FF6666', '#AA1A2D']  # White to Red
    cmap_exp_custom = LinearSegmentedColormap.from_list('custom_red', colors_exp)
    
    sns.heatmap(pivot_exp_twh, annot=False, fmt='.1f', cmap=cmap_exp_custom, 
                cbar_kws={'label': ''}, ax=ax_heat1_2030,
                linewidths=0.5, linecolor='gray', vmin=0)
    ax_heat1_2030.set_title(f'{energy_type.capitalize()} {year}\nExposure (TWh)', 
                            fontsize=10, fontweight='bold')
    # Only show x-axis label for bottom row (Hydro, row_idx=2)
    if row_idx == 2:
        ax_heat1_2030.set_xlabel('Supply Scenario', fontsize=9, fontweight='bold')
    else:
        ax_heat1_2030.set_xlabel('', fontsize=9, fontweight='bold')
    ax_heat1_2030.set_ylabel('Hazard Buffer', fontsize=9, fontweight='bold')
    ax_heat1_2030.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # COLUMN 2: Heatmap for exp_risk_avoid_MWh (2030)
    ax_heat2_2030 = fig.add_subplot(gs[row_idx, 2])
    
    pivot_risk = year_data.pivot_table(
        values='exp_risk_avoid_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_risk = pivot_risk.reindex(index=buffer_order, columns=supply_order)
    pivot_risk_twh = pivot_risk / 1e6
    
    # Create custom colormap: start with bar color for max value
    colors_risk = ['#FFFFFF', '#CCE5FF', '#99CCFF', '#6699FF', '#002147']  # White to Oxford Blue
    cmap_risk_custom = LinearSegmentedColormap.from_list('custom_blue', colors_risk)
    
    sns.heatmap(pivot_risk_twh, annot=False, fmt='.1f', cmap=cmap_risk_custom,
                cbar_kws={'label': ''}, ax=ax_heat2_2030,
                linewidths=0.5, linecolor='gray', vmin=0)
    ax_heat2_2030.set_title(f'{energy_type.capitalize()} {year}\nExposed with Risk-Avoidance (TWh)', 
                            fontsize=10, fontweight='bold')
    # Only show x-axis label for bottom row (Hydro, row_idx=2)
    if row_idx == 2:
        ax_heat2_2030.set_xlabel('Supply Scenario', fontsize=9, fontweight='bold')
    else:
        ax_heat2_2030.set_xlabel('', fontsize=9, fontweight='bold')
    ax_heat2_2030.set_ylabel('Hazard Buffer', fontsize=9, fontweight='bold')
    ax_heat2_2030.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # ========== 2050 CHARTS (Columns 3-5) ==========
    year = 2050
    year_data = energy_data[energy_data['year'] == year].copy()
    
    # COLUMN 3: Bar chart (100% & 0km) - exp_MWh vs exp_risk_avoid_MWh
    ax_bar_2050 = fig.add_subplot(gs[row_idx, 3])
    
    bar_data = year_data[
        (year_data['supply_scenario'] == '100%') & 
        (year_data['hazard_buffer'] == '0km')
    ]
    
    total_exp = bar_data['exp_MWh'].sum() / 1e6
    total_risk = bar_data['exp_risk_avoid_MWh'].sum() / 1e6
    
    bars = ax_bar_2050.bar(['Exposed', 'Exposed with\nRisk-Avoidance'], [total_exp, total_risk],
                           color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1)
    ax_bar_2050.set_ylabel('', fontsize=9, fontweight='bold')
    ax_bar_2050.set_title(f'{energy_type.capitalize()} {year}\n100% Supply, 0km Buffer',
                          fontsize=10, fontweight='bold')
    ax_bar_2050.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar_2050.set_ylim(0, max(total_exp, total_risk) * 1.2)
    ax_bar_2050.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    for bar in bars:
        height = bar.get_height()
        ax_bar_2050.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # COLUMN 4: Heatmap for exp_MWh (2050)
    ax_heat1_2050 = fig.add_subplot(gs[row_idx, 4])
    
    pivot_exp = year_data.pivot_table(
        values='exp_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_exp = pivot_exp.reindex(index=buffer_order, columns=supply_order)
    pivot_exp_twh = pivot_exp / 1e6
    
    sns.heatmap(pivot_exp_twh, annot=False, fmt='.1f', cmap=cmap_exp_custom,
                cbar_kws={'label': ''}, ax=ax_heat1_2050,
                linewidths=0.5, linecolor='gray', vmin=0)
    ax_heat1_2050.set_title(f'{energy_type.capitalize()} {year}\nExposure (TWh)',
                            fontsize=10, fontweight='bold')
    # Only show x-axis label for bottom row (Hydro, row_idx=2)
    if row_idx == 2:
        ax_heat1_2050.set_xlabel('Supply Scenario', fontsize=9, fontweight='bold')
    else:
        ax_heat1_2050.set_xlabel('', fontsize=9, fontweight='bold')
    ax_heat1_2050.set_ylabel('Hazard Buffer', fontsize=9, fontweight='bold')
    ax_heat1_2050.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # COLUMN 5: Heatmap for exp_risk_avoid_MWh (2050)
    ax_heat2_2050 = fig.add_subplot(gs[row_idx, 5])
    
    pivot_risk = year_data.pivot_table(
        values='exp_risk_avoid_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_risk = pivot_risk.reindex(index=buffer_order, columns=supply_order)
    pivot_risk_twh = pivot_risk / 1e6
    
    sns.heatmap(pivot_risk_twh, annot=False, fmt='.1f', cmap=cmap_risk_custom,
                cbar_kws={'label': ''}, ax=ax_heat2_2050,
                linewidths=0.5, linecolor='gray', vmin=0)
    ax_heat2_2050.set_title(f'{energy_type.capitalize()} {year}\nExposed with Risk-Avoidance (TWh)',
                            fontsize=10, fontweight='bold')
    # Only show x-axis label for bottom row (Hydro, row_idx=2)
    if row_idx == 2:
        ax_heat2_2050.set_xlabel('Supply Scenario', fontsize=9, fontweight='bold')
    else:
        ax_heat2_2050.set_xlabel('', fontsize=9, fontweight='bold')
    ax_heat2_2050.set_ylabel('Hazard Buffer', fontsize=9, fontweight='bold')
    ax_heat2_2050.tick_params(axis='both', which='both', length=0)  # Remove ticks

# Save figure
output_file = 'outputs_processed_fig/p1_z_fig7.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure saved to: {output_file}")

# Also save as PDF for publication quality
output_pdf = 'outputs_processed_fig/p1_z_fig7.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"✅ PDF saved to: {output_pdf}")

plt.show()

print("\n=== FIGURE GENERATION COMPLETE ===")
print("Layout: 3 rows × 6 columns")
print("  Rows: Solar (top), Wind (middle), Hydro (bottom)")
print("  Columns 1-3: 2030 (Bar, Exposure Heatmap, Risk-Avoidance Heatmap)")
print("  Columns 4-6: 2050 (Bar, Exposure Heatmap, Risk-Avoidance Heatmap)")
