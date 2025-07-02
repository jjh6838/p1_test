#visual1: Capacity MW - Global, by Region, and Income Group

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel('outputs_processed_data/p1_a_ember_2023_30.xlsx')

# Ensure the data is a DataFrame
if not isinstance(data, pd.DataFrame):
    raise ValueError("The data loaded from p1_a_ember_2023_30.xlsx must be a pandas DataFrame.")

# Add Fossil_2030 and Nuclear_2030 columns using 2023 data if they don't exist
if 'Fossil_2030' not in data.columns and 'Fossil_2023' in data.columns:
    data['Fossil_2030'] = data['Fossil_2023']
if 'Nuclear_2030' not in data.columns and 'Nuclear_2023' in data.columns:
    data['Nuclear_2030'] = data['Nuclear_2023']

# Reorder columns to match the desired order
desired_order = [
    'Fossil_2023', 'Other Renewables_2023', 'Hydro_2023', 'Nuclear_2023', 'Solar_2023', 'Wind_2023',
    'Hydro_2030', 'Solar_2030', 'Wind_2030', 'Other Renewables_2030', 'Fossil_2030', 'Nuclear_2030'
]
data = data.reindex(columns=['Region', 'Income group'] + desired_order, fill_value=0)

# Group by Region and Income group, summing up the energy capacities
grouped_data = data.groupby(['Region', 'Income group']).sum().reset_index()

# Melt the data for plotting
melted_data = grouped_data.melt(
    id_vars=['Region', 'Income group'], 
    value_vars=desired_order, 
    var_name='Energy Type and Year', 
    value_name='Capacity (MW)'
)
melted_data['Year'] = melted_data['Energy Type and Year'].str.extract(r'_(\d{4})')[0]
melted_data['Energy Type'] = melted_data['Energy Type and Year'].str.extract(r'^(.*)_\d{4}')[0]

# --- NEW: Plot all country-level lines in light grey ---
# Reload the original data (not grouped)
country_melted = data.melt(
    id_vars=['Region', 'Income group'],
    value_vars=desired_order,
    var_name='Energy Type and Year',
    value_name='Capacity (MW)'
)
country_melted['Year'] = country_melted['Energy Type and Year'].str.extract(r'_(\d{4})')[0]
country_melted['Energy Type'] = country_melted['Energy Type and Year'].str.extract(r'^(.*)_\d{4}')[0]

# Prepare data for stacking
stacked_data = country_melted.groupby(['Region', 'Income group', 'Year', 'Energy Type'], as_index=False)['Capacity (MW)'].mean()

regions = list(stacked_data['Region'].unique())
income_groups = list(stacked_data['Income group'].unique())
years = sorted(stacked_data['Year'].unique())
energy_types = [et for et in desired_order if et.split('_')[0] in stacked_data['Energy Type'].unique()]

# Use all years for plotting, but only use 2023 for legend labels and order
energy_types_2023 = [et for et in desired_order if et.endswith('_2023') and et.split('_')[0] in stacked_data['Energy Type'].unique()]
energy_type_names = [et.split('_')[0] for et in energy_types_2023]

fig, axes = plt.subplots(3, 4, figsize=(20, 15), squeeze=False)

# 1x1: Global
ax = axes[0, 0]
global_pivot = stacked_data.pivot_table(index='Year', columns='Energy Type', values='Capacity (MW)', aggfunc='sum', fill_value=0)
global_pivot = global_pivot.reindex(columns=energy_type_names, fill_value=0)
global_pivot.plot.area(ax=ax, stacked=True, alpha=0.8, legend=False)
ax.set_title('Global')
ax.set_ylabel('Capacity (MW)')
ax.set_xlabel('Year')
ax.tick_params(axis='x', rotation=45)

# 1x2 to 2x4: Regions (7 total)
for i, region in enumerate(regions):
    row = (i + 1) // 4
    col = (i + 1) % 4
    ax = axes[row, col]
    region_data = stacked_data[stacked_data['Region'] == region]
    pivot = region_data.pivot_table(index='Year', columns='Energy Type', values='Capacity (MW)', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=energy_type_names, fill_value=0)
    pivot.plot.area(ax=ax, stacked=True, alpha=0.8, legend=False)
    ax.set_title(f'Region: {region}')
    ax.set_ylabel('Capacity (MW)')
    ax.set_xlabel('Year')
    ax.tick_params(axis='x', rotation=45)

# 3x1 to 3x4: Income groups (4 total)
for i, income in enumerate(income_groups):
    ax = axes[2, i]
    income_data = stacked_data[stacked_data['Income group'] == income]
    pivot = income_data.pivot_table(index='Year', columns='Energy Type', values='Capacity (MW)', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=energy_type_names, fill_value=0)
    pivot.plot.area(ax=ax, stacked=True, alpha=0.8, legend=False)
    ax.set_title(f'Income group: {income}')
    ax.set_ylabel('Capacity (MW)')
    ax.set_xlabel('Year')
    ax.tick_params(axis='x', rotation=45)

# Hide unused axes
for r in range(3):
    for c in range(4):
        if (r, c) not in [(0, 0)] + [((i+1)//4, (i+1)%4) for i in range(len(regions))] + [(2, i) for i in range(len(income_groups))]:
            fig.delaxes(axes[r, c])

# Add a single global legend using 2023 energy types/colors
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, energy_type_names, title='Energy Type', loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize='large')

fig.suptitle('Stacked Energy Capacity: Global, by Region, and by Income Group', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
