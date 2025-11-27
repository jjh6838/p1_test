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
print(f"Hazard types: {sorted(data['hazard_type'].unique())}")

# Filter to only Solar, Wind, and Hydro (with capitalized names)
energy_types = ['Solar', 'Wind', 'Hydro']
data_filtered = data[data['energy_type'].isin(energy_types)].copy()

print(f"\nFiltered to {len(data_filtered):,} rows for Solar, Wind, Hydro")

# Aggregate across all hazard types (sum exposure across all hazards)
print("Aggregating across all hazard types...")
data_aggregated = data_filtered.groupby(
    ['ISO3_code', 'energy_type', 'year', 'supply_scenario', 'hazard_buffer']
).agg({
    'exp_MWh': 'sum',
    'exp_USD': 'sum',
    'exp_risk_avoid_MWh': 'sum',
    'exp_risk_avoid_USD': 'sum'
}).reset_index()

print(f"Aggregated to {len(data_aggregated):,} rows (summed across hazard types)")

# Create figure with 3 rows × 6 columns + space for colorbar
fig = plt.figure(figsize=(7, 4))
gs = fig.add_gridspec(3, 7, hspace=0.3, wspace=0.3, width_ratios=[1, 1, 1, 1, 1, 1, 0.15])

# Color schemes (Oxford colors)
bar_colors = ['#AA1A2D', '#002147']  # Red for exposed, Oxford Blue for risk-avoided

# Create custom colormaps
from matplotlib.colors import LinearSegmentedColormap
# Diverging colormap: blue (negative) - white (0) - red (positive)
colors_normalized = ['#0000CC', '#3366FF', '#6699FF', '#99CCFF', '#FFFFFF', '#FFCCCC', '#FF9999', '#FF6666', '#CC0000']
cmap_normalized = LinearSegmentedColormap.from_list('diverging', colors_normalized)

# Process each energy type (row)
for row_idx, energy_type in enumerate(energy_types):
    print(f"\nProcessing {energy_type.upper()}...")
    
    # Filter data for this energy type
    energy_data = data_aggregated[data_aggregated['energy_type'] == energy_type].copy()
    
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
                           color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_bar_2030.set_ylabel('')
    ax_bar_2030.set_xlabel('')
    ax_bar_2030.set_title(f'{energy_type} {year}\n100% Supply, 0km Buffer', 
                          fontsize=7, fontweight='bold')
    ax_bar_2030.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar_2030.set_ylim(0, max(total_exp, total_risk) * 1.2)
    ax_bar_2030.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_bar_2030.tick_params(axis='x', which='both', bottom=False, labelsize=6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_bar_2030.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    # COLUMN 1: Heatmap for exp_MWh (2030) - Normalized as %
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
    
    # Get baseline (100% supply, 0km buffer) for normalization
    baseline_exp = pivot_exp.loc['0km', '100%']
    
    # Calculate percentage change from baseline
    pivot_exp_pct = ((pivot_exp - baseline_exp) / baseline_exp * 100)
    
    sns.heatmap(pivot_exp_pct, annot=False, fmt='.0f', cmap=cmap_normalized, 
                cbar=False, ax=ax_heat1_2030,
                linewidths=0.3, linecolor='gray', vmin=-100, vmax=100, center=0)
    ax_heat1_2030.set_title(f'{energy_type} {year}\nExposure (%)', 
                            fontsize=7, fontweight='bold')
    ax_heat1_2030.set_xlabel('', fontsize=6, fontweight='bold')
    ax_heat1_2030.set_ylabel('', fontsize=6, fontweight='bold')
    ax_heat1_2030.set_xticklabels([])
    ax_heat1_2030.set_yticklabels([])
    ax_heat1_2030.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # COLUMN 2: Heatmap for exp_risk_avoid_MWh (2030) - Normalized as %
    ax_heat2_2030 = fig.add_subplot(gs[row_idx, 2])
    
    pivot_risk = year_data.pivot_table(
        values='exp_risk_avoid_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_risk = pivot_risk.reindex(index=buffer_order, columns=supply_order)
    
    # Get baseline (100% supply, 0km buffer) for normalization
    baseline_risk = pivot_risk.loc['0km', '100%']
    
    # Calculate percentage change from baseline
    pivot_risk_pct = ((pivot_risk - baseline_risk) / baseline_risk * 100)
    
    sns.heatmap(pivot_risk_pct, annot=False, fmt='.0f', cmap=cmap_normalized,
                cbar=False, ax=ax_heat2_2030,
                linewidths=0.3, linecolor='gray', vmin=-100, vmax=100, center=0)
    ax_heat2_2030.set_title(f'{energy_type} {year}\nRisk-Avoidance (%)', 
                            fontsize=7, fontweight='bold')
    ax_heat2_2030.set_xlabel('', fontsize=6, fontweight='bold')
    ax_heat2_2030.set_ylabel('', fontsize=6, fontweight='bold')
    ax_heat2_2030.set_xticklabels([])
    ax_heat2_2030.set_yticklabels([])
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
                           color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_bar_2050.set_ylabel('')
    ax_bar_2050.set_xlabel('')
    ax_bar_2050.set_title(f'{energy_type} {year}\n100% Supply, 0km Buffer',
                          fontsize=7, fontweight='bold')
    ax_bar_2050.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar_2050.set_ylim(0, max(total_exp, total_risk) * 1.2)
    ax_bar_2050.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_bar_2050.tick_params(axis='x', which='both', bottom=False, labelsize=6)
    
    for bar in bars:
        height = bar.get_height()
        ax_bar_2050.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    # COLUMN 4: Heatmap for exp_MWh (2050) - Normalized as %
    ax_heat1_2050 = fig.add_subplot(gs[row_idx, 4])
    
    pivot_exp = year_data.pivot_table(
        values='exp_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_exp = pivot_exp.reindex(index=buffer_order, columns=supply_order)
    
    # Get baseline (100% supply, 0km buffer) for normalization
    baseline_exp = pivot_exp.loc['0km', '100%']
    
    # Calculate percentage change from baseline
    pivot_exp_pct = ((pivot_exp - baseline_exp) / baseline_exp * 100)
    
    sns.heatmap(pivot_exp_pct, annot=False, fmt='.0f', cmap=cmap_normalized,
                cbar=False, ax=ax_heat1_2050,
                linewidths=0.3, linecolor='gray', vmin=-100, vmax=100, center=0)
    ax_heat1_2050.set_title(f'{energy_type} {year}\nExposure (%)',
                            fontsize=7, fontweight='bold')
    ax_heat1_2050.set_xlabel('', fontsize=6, fontweight='bold')
    ax_heat1_2050.set_ylabel('', fontsize=6, fontweight='bold')
    ax_heat1_2050.set_xticklabels([])
    ax_heat1_2050.set_yticklabels([])
    ax_heat1_2050.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # COLUMN 5: Heatmap for exp_risk_avoid_MWh (2050) - Normalized as %
    ax_heat2_2050 = fig.add_subplot(gs[row_idx, 5])
    
    pivot_risk = year_data.pivot_table(
        values='exp_risk_avoid_MWh',
        index='hazard_buffer',
        columns='supply_scenario',
        aggfunc='sum'
    )
    
    pivot_risk = pivot_risk.reindex(index=buffer_order, columns=supply_order)
    
    # Get baseline (100% supply, 0km buffer) for normalization
    baseline_risk = pivot_risk.loc['0km', '100%']
    
    # Calculate percentage change from baseline
    pivot_risk_pct = ((pivot_risk - baseline_risk) / baseline_risk * 100)
    
    sns.heatmap(pivot_risk_pct, annot=False, fmt='.0f', cmap=cmap_normalized,
                cbar=False, ax=ax_heat2_2050,
                linewidths=0.3, linecolor='gray', vmin=-100, vmax=100, center=0)
    ax_heat2_2050.set_title(f'{energy_type} {year}\nRisk-Avoidance (%)',
                            fontsize=7, fontweight='bold')
    ax_heat2_2050.set_xlabel('', fontsize=6, fontweight='bold')
    ax_heat2_2050.set_ylabel('', fontsize=6, fontweight='bold')
    ax_heat2_2050.set_xticklabels([])
    ax_heat2_2050.set_yticklabels([])
    ax_heat2_2050.tick_params(axis='both', which='both', length=0)  # Remove ticks

# Add single colorbar on the right side (spans all rows)
ax_cbar = fig.add_subplot(gs[:, 6])
im = plt.cm.ScalarMappable(cmap=cmap_normalized, norm=plt.Normalize(vmin=-100, vmax=100))
im.set_array([])
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.set_label('% Change from Baseline\n(100% Supply, 0km Buffer)', fontsize=7, fontweight='bold')
cbar.ax.tick_params(labelsize=6)

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
