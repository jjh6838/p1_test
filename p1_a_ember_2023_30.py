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
    return {cat: row[column_name] * proportions[cat] for cat in categories} 

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

# Check if cells in 2030 target columns are null and fill them with corresponding 2023 data row by row
columns_to_fill = ["Hydro_2030", "Solar_2030", "Wind_2030"]
for col in columns_to_fill:
    corresponding_2023_col = col.replace("_2030", "_2023")
    if corresponding_2023_col in df3.columns:
        df3.loc[df3[col].isnull(), col] = df3.loc[df3[col].isnull(), corresponding_2023_col]


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

# Check if cells in 2030 target columns are null and fill them with corresponding 2023 data row by row
columns_to_fill = ["Hydro_2030", "Solar_2030", "Wind_2030"]
for col in columns_to_fill:
    corresponding_2023_col = col.replace("_2030", "_2023")
    # Fill missing 2030 target values with corresponding 2023 data for rows with a valid category
    if corresponding_2023_col in final_merged_df.columns:
        mask = (final_merged_df[col].isnull()) & (final_merged_df["Category"].notnull())
        final_merged_df.loc[mask, col] = final_merged_df.loc[mask, corresponding_2023_col]


### Task 4
# Load the capacity conversion factors
capacity_conversion_factors_df = pd.read_excel(p1_b_output_path, sheet_name="Grouped_cur")

# Calculate global mean conversion rates for each type
global_mean_conv_rates = capacity_conversion_factors_df[
    ["Hydro_ConvRate", "Solar_ConvRate", "Wind_ConvRate", "Other Renewables_ConvRate"]
].mean()

# Fill NA or 0 with global mean
def fill_missing_or_zero(row):
    return row.mask((row.isna()) | (row == 0), global_mean_conv_rates)

# Apply the replacement
capacity_conversion_factors = (
    capacity_conversion_factors_df
    .set_index("Country Code")[["Hydro_ConvRate", "Solar_ConvRate", "Wind_ConvRate", "Other Renewables_ConvRate"]]
    .apply(fill_missing_or_zero, axis=1)
    .to_dict(orient="index")
)

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
        final_merged_df.at[index, "Hydro_2030_MWh"] = hydro_mwh
        final_merged_df.at[index, "Solar_2030_MWh"] = solar_mwh
        final_merged_df.at[index, "Wind_2030_MWh"] = wind_mwh
        final_merged_df.at[index, "Other Renewables_2030_MWh"] = other_renewables_mwh
        
        # Calculate the total MWh for 2030 and assign it to the new column
        final_merged_df.at[index, "HSWO_2030_MWh"] = hydro_mwh + solar_mwh + wind_mwh + other_renewables_mwh

# Compute total RE generation per capita for 2030
final_merged_df["RE_Generation_Per_Capita_2030"] = (
    final_merged_df["HSWO_2030_MWh"] / final_merged_df["PopTotal_2030"]
)


### 7/1/2025 to be at home. + in dataset b, use global mean conversion rates for each type!!
### Task 5
# Extrapolate the 2030 targets for countries without NDCs using regional benchmarks
# Load/add WB Country Class (https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)
wb_country_class_path = "wb_country_class/CLASS.xlsx"
wb_country_class_df = pd.read_excel(wb_country_class_path, sheet_name="List of economies")

# Merge the WB Country Class data with the final merged dataset and drop the "Code" column in one step
final_merged_df = pd.merge(
    final_merged_df,
    wb_country_class_df[["Code", "Region", "Income group"]],
    left_on="ISO3_code",
    right_on="Code",
    how="left"
).drop(columns=["Code"])

# Drop rows with missing values in Region or Income group
final_merged_df.dropna(subset=["Region", "Income group"], inplace=True)

# Define the energy types to be used for regional benchmarks
energy_types = ["Hydro_2030_MWh", "Solar_2030_MWh", "Wind_2030_MWh", "Other Renewables_2030_MWh"]

# Filter the dataset to include only countries with valid NDCs (non-null "Category")
# Melt the dataframe to reshape it for grouping by region, income group, and energy type
regional_benchmarks = (
    final_merged_df[final_merged_df["Category"].notnull()]  # Filter rows with valid NDCs
    .melt(
        id_vars=["Region", "Income group", "PopTotal_2030"],  # Include population for per capita calculation
        value_vars=energy_types,  # Columns to unpivot (energy types)
        var_name="Energy_Type",  # Name of the new column for energy types
        value_name="MWh"  # Name of the new column for energy values
    )
    .assign(MWh_Per_Capita=lambda df: df["MWh"] / df["PopTotal_2030"])  # Calculate MWh per capita
    .groupby(["Region", "Income group", "Energy_Type"])["MWh_Per_Capita"]  # Group by region, income group, and energy type
    .mean()  # Compute the average MWh per capita for each group
    .reset_index()  # Reset the index to return a flat dataframe
)

# Pivot so each energy type is a column with _Regional_Benchmark suffix
regional_benchmarks_pivot = regional_benchmarks.pivot(
    index=["Region", "Income group"], columns="Energy_Type", values="MWh_Per_Capita"
).reset_index()
regional_benchmarks_pivot.columns = [
    "Region", "Income group"
] + [f"{col}_Regional_Benchmark" for col in regional_benchmarks_pivot.columns[2:]]

# Merge regional benchmarks back into the dataset
final_merged_df = pd.merge(
    final_merged_df,
    regional_benchmarks_pivot,
    on=["Region", "Income group"],
    how="left"
)

# Apply regional benchmarks to countries without 2030 targets
for energy_type in energy_types:
    benchmark_col = energy_type + "_Regional_Benchmark"
    final_merged_df.loc[final_merged_df["Category"].isnull(), energy_type] = (
        final_merged_df.loc[final_merged_df["Category"].isnull(), "PopTotal_2030"]
        * final_merged_df.loc[final_merged_df["Category"].isnull(), benchmark_col]
    ).fillna(0)  # Fill any missing values with 0

# Save the final merged dataset to a CSV file
final_merged_df.to_excel("outputs_processed_data/p1_a_ember_2023_30.xlsx", index=False)