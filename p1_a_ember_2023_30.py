# 6/30/2025 at work

import pandas as pd

# Task 1
# Load the first dataset 
file_path1 = "un_pop/WPP2024_TotalPopulationBySex2025-05-02.csv" # updated 5/2/2025 
# https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format

df1 = pd.read_csv(file_path1, low_memory=False)

# Filter the first dataset for "Medium" variant and years 2023, 2030, and 2050
filtered_df1 = df1[(df1["Variant"] == "Medium") & (df1["Time"].isin([2023, 2030, 2050]))]

# Select required columns from the first dataset
selected_columns1 = ["ISO3_code", "Time", "PopTotal"]
filtered_df1 = filtered_df1[selected_columns1]

# Remove rows with missing ISO codes from the first dataset
filtered_df1 = filtered_df1.dropna(subset="ISO3_code")

# Pivot the first dataset to have years as columns
pivot_df1 = filtered_df1.pivot(index="ISO3_code", columns="Time", values="PopTotal").reset_index()

# Rename columns for clarity in the first dataset
pivot_df1.columns = ["ISO3_code", "PopTotal_2023", "PopTotal_2030", "PopTotal_2050"]

# Calculate the population growth factor from 2023 to 2030 and 2050
pivot_df1["Growth_Factor_2023_2030"] = pivot_df1["PopTotal_2030"] / pivot_df1["PopTotal_2023"]
pivot_df1["Growth_Factor_2023_2050"] = pivot_df1["PopTotal_2050"] / pivot_df1["PopTotal_2023"]


### Task 2
# Load the processed data from p1_b_ember_gem_2023.py (# 2nd dataset)
p1_b_output_path = r"outputs_processed_data\p1_b_ember_gem_2023.xlsx"
grouped_df = pd.read_excel(p1_b_output_path, sheet_name="Grouped_cur")

# Extract relevant columns for merging
granular_columns = ["Country Name", "Country Code", "Total_Potential_MWh"] + [col for col in grouped_df.columns if "_Larger_MW" in col]
grouped_df = grouped_df[granular_columns]

# Rename columns for clarity
grouped_df.rename(columns={"Country Code": "ISO3_code"}, inplace=True)
grouped_df.rename(columns={"Total_Potential_MWh": "Total_MWh_2023"}, inplace=True)

# Rename columns to follow the {type}_2023 naming convention
grouped_df.rename(columns={col: col.replace("_Larger_MW", "_2023") for col in grouped_df.columns if "_Larger_MW" in col}, inplace=True)

# Merge the two datasets by "ISO3_code"
merged_df = pd.merge(pivot_df1, grouped_df, on="ISO3_code", how="inner")

#Calcluate "per capita MWh 2023"
merged_df["Per_Capita_MWh_2023"] = merged_df["Total_MWh_2023"] / merged_df["PopTotal_2023"]

# Extrapolate the 2023 data to 2030/2050 using the population growth factors for Total_MWh_2023
merged_df["Total_MWh_2030"] = merged_df["Total_MWh_2023"] * merged_df["Growth_Factor_2023_2030"] * 1.2  # energy consumption 20% up for 2030
merged_df["Total_MWh_2050"] = merged_df["Total_MWh_2023"] * merged_df["Growth_Factor_2023_2050"] * 1.5  # energy consumption 50% up for 2050


### Task 3
# Load the third dataset
file_path3 = "ember_energy_data/targets_download2025-05-02.xlsx" # updated 5/2/2025
df3_original = pd.read_excel(file_path3, sheet_name="capacity_target_wide")

# Duplicate the sheet to create "capacity_target_wide_2"
df3 = df3_original.copy()

# Clean up the data in "capacity_target_wide_2"
# 1. Add "Offshore Wind" and "Onshore Wind" values to the existing "Wind" column
df3["Wind"] = df3.get("Wind", 0).fillna(0) + df3.get("Offshore Wind", 0).fillna(0) + df3.get("Onshore Wind", 0).fillna(0)

# 2. Merge "Bioenergy" and "Other Renewables" into "Other Renewables"
df3["Other Renewables"] = df3.get("Bioenergy", 0).fillna(0) + df3.get("Other Renewables", 0).fillna(0)

# 3. Drop "Offshore Wind", "Onshore Wind", and "Bioenergy" columns
df3.drop(columns=["Offshore Wind", "Onshore Wind", "Bioenergy"], inplace=True)

df3 = df3[df3["country_code"] != "EU"]

# Create a new column "Category" based on the specified conditions
def determine_category(row):
    if row.get("Renewables", 0) > 0:
        return 5
    elif row.get("Hydro, bio and other renewables", 0) > 0:
        return 3
    elif row.get("Rest of renewables", 0) > 0 and row.get("Hydro", 0) > 0:
        return 1
    elif row.get("Rest of renewables", 0) > 0 and row.get("Wind", 0) > 0:
        return 4
    else:
        return 2

df3["Category"] = df3.apply(determine_category, axis=1)

# Print the number of unique country codes in df3
print("Number of unique country codes in df3:", df3["country_code"].nunique())

# Print the number of unique country codes in merged_df
print("Number of unique country codes in merged_df:", merged_df["ISO3_code"].nunique())

# Define a function to disaggregate based on 2023 data: only for 80+ countries with 2030 NDCs targets 
def disaggregate(row, merged_df, categories, column_name):
    """
    Disaggregate the values in the specified column into the given categories based on proportions derived from 2023 data.

    Args:
        row (pd.Series): A row from the dataframe containing the data to be disaggregated.
        merged_df (pd.DataFrame): The dataframe containing the reference data for proportions.
        categories (list): List of category names to disaggregate into.
        column_name (str): The name of the column to disaggregate.

    Returns:
        dict: A dictionary with the disaggregated values for each category.
    """
    # Ensure required columns exist in merged_df
    missing_columns = [f"{cat}_2023" for cat in categories if f"{cat}_2023" not in merged_df.columns]
    if missing_columns:
        print(f"Missing columns in merged_df: {missing_columns}")

    # Extract the country code from the row
    country_code = row["country_code"]

    # Filter merged_df for the specific country code
    filtered_merged_df = merged_df[merged_df["ISO3_code"] == country_code]

    # Check if the country code exists in merged_df
    if filtered_merged_df.empty:
        print(f"Country code {country_code} not found in merged_df")
        return {cat: 0 for cat in categories}

    # Calculate the total 2023 Ember capacity for the categories
    total_2023 = filtered_merged_df[[f"{cat}_2023" for cat in categories]].fillna(0).sum(axis=1).values[0]

    # Handle cases where total_2023 is zero
    if total_2023 == 0:
        print(f"Total 2023 for country code {country_code} is 0")
        return {cat: 0 for cat in categories}

    # Calculate proportions for each category based on 2023 Ember capacity data
    proportions = {
        cat: filtered_merged_df[f"{cat}_2023"].fillna(0).values[0] / total_2023
        for cat in categories
    }

    # Handle cases where total_2023 is zero
    if total_2023 == 0:
        print(f"Total 2023 for country code {country_code} is 0")

    # Calculate proportions for each category based on 2023 data
    proportions = {
        cat: (
            merged_df.loc[merged_df["ISO3_code"] == country_code, f"{cat}_2023"].fillna(0).values[0]
            / total_2023
        )
        if total_2023 != 0
        else 0
        for cat in categories
    }

    # Disaggregate the values in the specified column based on the calculated proportions
    return {cat: row[column_name] * proportions[cat] for cat in categories} # * 1000 to convert from GWh to MWh

# Disaggregate [Rest of renewables] for Category 1
df3.loc[df3["Category"] == 1, ["Solar", "Wind", "Other Renewables"]] = df3[df3["Category"] == 1].apply(
    lambda row: pd.Series(disaggregate(row, merged_df, ["Solar", "Wind", "Other Renewables"], "Rest of renewables")), axis=1
) 

# Disaggregate [Rest of renewables] for Category 4
df3.loc[df3["Category"] == 4, ["Solar", "Hydro", "Other Renewables"]] = df3[df3["Category"] == 4].apply(
    lambda row: pd.Series(disaggregate(row, merged_df, ["Solar", "Hydro", "Other Renewables"], "Rest of renewables")), axis=1
) 

# Disaggregate [Hydro, bio and other renewables] for Category 3
df3.loc[df3["Category"] == 3, ["Hydro", "Other Renewables"]] = df3[df3["Category"] == 3].apply(
    lambda row: pd.Series(disaggregate(row, merged_df, ["Hydro", "Other Renewables"], "Hydro, bio and other renewables")), axis=1
) 

# Disaggregate [Renewables] for Category 5
df3.loc[df3["Category"] == 5, ["Solar", "Wind", "Hydro", "Other Renewables"]] = df3[df3["Category"] == 5].apply(
    lambda row: pd.Series(disaggregate(row, merged_df, ["Solar", "Wind", "Hydro", "Other Renewables"], "Renewables")), axis=1
) 
# Drop unnecessary columns
df3.drop(columns=["Rest of renewables", "Hydro, bio and other renewables", "Renewables"], inplace=True)

# Multiply 2030 target columns by 1000 to convert from GW to MW (if needed)
for col in ["res_capacity_target", "Hydro", "Other Renewables", "Solar", "Wind"]:
    if col in df3.columns:
        df3[col] = df3[col] * 1000

# Rename columns for 2030 targets
df3.rename(columns={
    "Hydro": "Hydro_2030",
    "Other Renewables": "Other Renewables_2030",
    "Solar": "Solar_2030",
    "Wind": "Wind_2030"
}, inplace=True)

# Merge the third dataset with the previously merged dataset, pulling only the needed columns from df3
columns_to_pull = [
    "country_code", "res_capacity_target", "res_share_target",
    "Hydro_2030", "Solar_2030", "Wind_2030", "Other Renewables_2030", "Category"
]

final_merged_df = pd.merge(
    merged_df,
    df3[columns_to_pull],
    left_on="ISO3_code",
    right_on="country_code",
    how="left"
)


# Task 4
# Load the capacity conversion factors from the original p1_b_ember_gem_2023.xlsx file
capacity_conversion_factors_df = pd.read_excel(p1_b_output_path, sheet_name="Grouped_cur")

# Extract the capacity conversion factors for each country
capacity_conversion_factors = capacity_conversion_factors_df.set_index("Country Code")[
    ["Hydro_ConvRate", "Solar_ConvRate", "Wind_ConvRate", "Other Renewables_ConvRate"]
].to_dict(orient="index")

# Calculate MWh for 2030 based on the conversion factors and compute the total
for index, row in final_merged_df.iterrows():
    country_code = row["ISO3_code"]
    if country_code in capacity_conversion_factors:
        conversion_factors = capacity_conversion_factors[country_code]
        hydro_mwh = row["Hydro_2030"] * conversion_factors["Hydro_ConvRate"]
        solar_mwh = row["Solar_2030"] * conversion_factors["Solar_ConvRate"]
        wind_mwh = row["Wind_2030"] * conversion_factors["Wind_ConvRate"]
        other_renewables_mwh = row["Other Renewables_2030"] * conversion_factors["Other Renewables_ConvRate"]
        
        # Assign individual MWh values to the dataframe
        # final_merged_df.at[index, "Hydro_2030_MWh"] = hydro_mwh
        # final_merged_df.at[index, "Solar_2030_MWh"] = solar_mwh
        # final_merged_df.at[index, "Wind_2030_MWh"] = wind_mwh
        # final_merged_df.at[index, "Other Renewables_2030_MWh"] = other_renewables_mwh
        
        # Calculate the total MWh for 2030 and assign it to the new column
        final_merged_df.at[index, "HSWO_2030_MWh"] = hydro_mwh + solar_mwh + wind_mwh + other_renewables_mwh


# Reorder columns to have Country Name and ISO3_code first
final_columns = ['Country Name', 'ISO3_code'] + [col for col in final_merged_df.columns if col not in ['Country Name', 'ISO3_code']]
final_merged_df = final_merged_df[final_columns]

# Save the final merged dataset to a CSV file
final_merged_df.to_excel("outputs_processed_data/p1_a_ember_2023_30.xlsx", index=False)