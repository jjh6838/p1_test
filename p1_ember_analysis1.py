
import pandas as pd

# File paths
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format.csv"
energy_facilities_path = r"re_data\Global-Integrated-Power-June-2024.xlsx"

# Read the energy data CSV and calculate the sum of "Value" for the specified conditions
energy_data = pd.read_csv(energy_data_csv_path)
filtered_energy_data = energy_data[
    (energy_data['Country code'] == 'KOR') &
    (energy_data['Year'] == 2023) &
    (energy_data['Category'] == 'Capacity') &
    (energy_data['Subcategory'] == 'Fuel') &
    (energy_data['Unit'] == 'GW')
]
total_capacity = filtered_energy_data['Value'].sum()
print(f"Total Capacity for KOR in 2023: {total_capacity} GW")

# Read the energy facilities data from the Excel file
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")
filtered_energy_facilities_df = energy_facilities_df[
    (energy_facilities_df['Country/area'] == 'South Korea') &
    (energy_facilities_df['Status'] == 'operating')
]
total_capacity_mw = filtered_energy_facilities_df['Capacity (MW)'].sum()
print(f"Total Capacity of operating facilities in South Korea: {total_capacity_mw} MW")


match_dict = {
    "Ember": "GEM",
    "Bioenergy": "bioenergy",
    "Coal": "coal",
    "Hydro": "hydropower",
    "Nuclear": "nuclear",
    "Solar": "solar",
    "Wind": "wind",
    "Other Renewables": "geothermal", #Other Renewables in Ember: Geothermal, Tide, Wave. 
    "Gas": "oil/gas",
    "Other Fossil": "oil/gas"
}

# Conversion rate per technology, based on Ember data
# wind: From MW to MWh = 3.39 TWh / 2.17 GW = Wind: 1,562,211.98 MWh per MW
# oil/gas: From MW to MWh = 168.82+7.8 TWh / 46.58+3.05 GW = 3,558,734.64 MWh per MW

KOR_2023_wind_conversion_rate_jeju = 1562211.98
KOR_2023_oilgas_conversion_rate_jeju = 3558734.64
