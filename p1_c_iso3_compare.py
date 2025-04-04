import pandas as pd
from pycountry import countries

# Load the Excel file
file_path = 'ember_energy_data/iso3_compare_ember_gem.xlsx'
df = pd.read_excel(file_path)

# Check the columns in the DataFrame
print(df.columns)

# Get the country names from the third column
country_names = df.iloc[:, 2]  # Third column (index 2)

# Manual mapping for country names not found in pycountry
manual_mapping = {
    "Bolivia": "BOL",
    "Bonaire, Sint Eustatius, and Saba": "BES",
    "Brunei": "BRN",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Holy See": "VAT",
    "Iran": "IRN",
    "Kosovo": "XKX",
    "Laos": "LAO",
    "Micronesia": "FSM",
    "Moldova": "MDA",
    "North Korea": "PRK",
    "Palestine": "PSE",
    "Republic of the Congo": "COG",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "The Gambia": "GMB",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Virgin Islands (U.S.)": "VIR"
}

# Map country names to ISO3 codes, put "unknown" if not found
iso3_codes = country_names.apply(lambda x: manual_mapping.get(x, countries.get(name=x).alpha_3 if countries.get(name=x) else "unknown"))

# Add the ISO3 codes to the fourth column
df['ISO3'] = iso3_codes

# Compare the ISO3 codes between the first and fourth columns
iso3_first_column = df.iloc[:, 0].dropna().unique()
iso3_fourth_column = df['ISO3'].dropna().unique()

missing_in_first = set(iso3_fourth_column).difference(iso3_first_column)
missing_in_fourth = set(iso3_first_column).difference(iso3_fourth_column)

print("Missing in first column:", missing_in_first)
print("Missing in fourth column:", missing_in_fourth)

# Compare the ISO3 codes between the first and second columns
iso3_first_column = df.iloc[:, 0].dropna().unique()
iso3_second_column = df.iloc[:, 1].dropna().unique()

missing_in_first = set(iso3_second_column).difference(iso3_first_column)
missing_in_second = set(iso3_first_column).difference(iso3_second_column)

print("Missing in first column:", missing_in_first)
print("Missing in second column:", missing_in_second)


# ''' Results:
# Missing in First Column (present in Fourth Column but not in First):

# Åland Islands (ALA)
# Andorra (AND)
# Caribbean Netherlands (BES)
# Curaçao (CUW)
# Federated States of Micronesia (FSM)
# Guernsey (GGY)
# Isle of Man (IMN)
# British Indian Ocean Territory (IOT)
# Jersey (JEY)
# Marshall Islands (MHL)
# Northern Mariana Islands (MNP)
# Mayotte (MYT)
# Norfolk Island (NFK)
# Palau (PLW)
# Vatican City (VAT)

# Missing in Fourth Column (present in First Column but not in Fourth):

# Falkland Islands (FLK)
# Gibraltar (GIB)
# Niue (NIU)
# Saint Helena, Ascension, and Tristan da Cunha (SHN)
# Saint Pierre and Miquelon (SPM)
# Turks and Caicos Islands (TCA)
# Saint Vincent and the Grenadines (VCT)
# British Virgin Islands (VGB)
# Remaining Countries After Dropping Missing Ones:

# 207 countries remain where the first and fourth columns match. '''
