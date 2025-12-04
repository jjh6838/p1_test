import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Oxford color scheme
plt.rcParams['font.family'] = 'Arial'

# Load the total generation data from p1_b_ember_2024_30_50.xlsx
print("Loading total generation data from p1_b_ember_2024_30_50.xlsx...")
total_gen_df = pd.read_excel('outputs_processed_data/p1_b_ember_2024_30_50.xlsx')

# Calculate total generation for each energy type and year
total_generation = {}
for energy_type in ['Solar', 'Wind', 'Hydro']:
    total_generation[energy_type] = {}
    for year in [2024, 2030, 2050]:
        col_name = f'{energy_type}_{year}_MWh'
        if col_name in total_gen_df.columns:
            total_generation[energy_type][year] = total_gen_df[col_name].sum() / 1e6
        else:
            total_generation[energy_type][year] = 0

# Load the processed data (NOT aggregated by hazard)
print("\nLoading exposure data from p1_y_results_data_etl.xlsx...")
data = pd.read_excel('outputs_processed_data/p1_y_results_data_etl.xlsx')

print(f"Loaded {len(data):,} rows")
print(f"Years: {sorted(data['year'].unique())}")
print(f"Energy types: {sorted(data['energy_type'].unique())}")
print(f"Hazard types: {sorted(data['hazard_type'].unique())}")

# Filter to only Solar, Wind, and Hydro
energy_types = ['Solar', 'Wind', 'Hydro']
data_filtered = data[data['energy_type'].isin(energy_types)].copy()

print(f"\nFiltered to {len(data_filtered):,} rows for Solar, Wind, Hydro")

# Get list of all hazard types
hazard_types = sorted(data_filtered['hazard_type'].unique())
print(f"Hazard types: {hazard_types}")

# Define colors for each hazard type
hazard_colors = {
    'Cyclone': '#002147',      # Oxford Blue
    'Earthquake': '#AA1A2D',   # Oxford Red
    'Flood': '#4E8098',        # Light Blue
    'Landslide': '#BE9B7B',    # Tan
    'Volcano': '#872434',      # Dark Red
    'Wildfire': '#CF4520'      # Orange-Red
}

# Create figure: 3 rows (energy types) × 2 columns (2030, 2050)
# Size: 180mm width = 7.087 inches, height proportional
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Hazard-Specific Exposure Breakdown\n100% Supply, 0km Buffer', 
             fontsize=14, fontweight='bold', y=0.995)

# Process each energy type (row)
for row_idx, energy_type in enumerate(energy_types):
    print(f"\nProcessing {energy_type.upper()}...")
    
    # Filter data for this energy type
    energy_data = data_filtered[data_filtered['energy_type'] == energy_type].copy()
    
    # Process both years
    for col_idx, year in enumerate([2030, 2050]):
        ax = axes[row_idx, col_idx]
        
        # Filter for 100% supply, 0km buffer
        year_data = energy_data[
            (energy_data['year'] == year) &
            (energy_data['supply_scenario'] == '100%') &
            (energy_data['hazard_buffer'] == '0km')
        ].copy()
        
        # Aggregate by hazard type (sum across countries)
        hazard_totals = year_data.groupby('hazard_type').agg({
            'exp_MWh': 'sum',
            'exp_risk_avoid_MWh': 'sum'
        }).reset_index()
        
        # Convert to TWh
        hazard_totals['exp_TWh'] = hazard_totals['exp_MWh'] / 1e6
        hazard_totals['risk_avoid_TWh'] = hazard_totals['exp_risk_avoid_MWh'] / 1e6
        hazard_totals['avoided_TWh'] = hazard_totals['exp_TWh'] - hazard_totals['risk_avoid_TWh']
        
        # Sort by exposure (descending)
        hazard_totals = hazard_totals.sort_values('exp_TWh', ascending=True)
        
        # Create stacked horizontal bar chart
        hazard_labels = hazard_totals['hazard_type'].tolist()
        exposed = hazard_totals['exp_TWh'].tolist()
        avoided = hazard_totals['avoided_TWh'].tolist()
        
        # Get colors for this set of hazards
        colors_exp = [hazard_colors.get(h, '#999999') for h in hazard_labels]
        
        # Create bars
        y_pos = np.arange(len(hazard_labels))
        
        # Exposed (full bar)
        bars1 = ax.barh(y_pos, exposed, color=colors_exp, alpha=0.85, 
                        edgecolor='black', linewidth=0.5, label='Exposed')
        
        # Avoided (lighter, on top)
        bars2 = ax.barh(y_pos, avoided, left=[e - a for e, a in zip(exposed, avoided)],
                        color='#2A9D8F', alpha=0.6, edgecolor='black', linewidth=0.5, 
                        label='Avoided')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hazard_labels, fontsize=9)
        ax.set_xlabel('TWh', fontsize=9, fontweight='bold')
        
        # Title with total generation reference
        total_gen = total_generation[energy_type][year]
        total_exp = hazard_totals['exp_TWh'].sum()
        ax.set_title(f'{energy_type} {year}\n(Planned: {total_gen:.1f} TWh, Exposed: {total_exp:.1f} TWh)',
                    fontsize=10, fontweight='bold')
        
        # Add value labels on bars
        for i, (exp, avoid) in enumerate(zip(exposed, avoided)):
            if exp > 0:
                # Show total exposure value
                ax.text(exp, i, f' {exp:.1f}', 
                       va='center', ha='left', fontsize=7, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add legend only to top-right subplot
        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

plt.tight_layout()

# Save as PDF
output_pdf = 'outputs_processed_fig/p1_z_fig8.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"\n✅ PDF saved to: {output_pdf}")

plt.show()

print("\n=== FIGURE 8 GENERATION COMPLETE ===")
print("Layout: 3 rows × 2 columns")
print("  Rows: Solar (top), Wind (middle), Hydro (bottom)")
print("  Columns: 2030 (left), 2050 (right)")
print("  Each cell: Horizontal stacked bar chart by hazard type")
print("  - Full bar = Exposed capacity")
print("  - Green portion = Avoided exposure (risk-avoidance)")
