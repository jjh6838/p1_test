# 6/30/2025 at work

import pandas as pd

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



# Load the processed data from p1_b_ember_gem_2023.py (# 2nd dataset)
p1_b_output_path = r"outputs_processed_data\p1_b_ember_gem_2023.xlsx"
granular_df = pd.read_excel(p1_b_output_path, sheet_name="Grouped_cur")

# Extract relevant columns for merging
granular_columns = ["Country Code", "Total_Potential_MWh"] + [col for col in granular_df.columns if "_Larger_MW" in col]
granular_df = granular_df[granular_columns]

# Rename columns for clarity
granular_df.rename(columns={"Country Code": "ISO3_code"}, inplace=True)
granular_df.rename(columns={"Total_Potential_MWh": "Total_MWh_2023"}, inplace=True)

# Rename columns to follow the {type}_2023 naming convention
granular_df.rename(columns={col: col.replace("_Larger_MW", "_2023") for col in granular_df.columns if "_Larger_MW" in col}, inplace=True)

# Merge the two datasets by "ISO3_code"
merged_df = pd.merge(pivot_df1, granular_df, on="ISO3_code", how="inner")

#Calcluate "per capita MWh 2023"
merged_df["Per_Capita_MWh_2023"] = merged_df["Total_MWh_2023"] / merged_df["PopTotal_2023"]

# Extrapolate the 2023 data to 2030/2050 using the population growth factors for Total_MWh_2023
merged_df["Total_MWh_2030"] = merged_df["Total_MWh_2023"] * merged_df["Growth_Factor_2023_2030"] * 1.2  # energy consumption 20% up for 2030
merged_df["Total_MWh_2050"] = merged_df["Total_MWh_2023"] * merged_df["Growth_Factor_2023_2050"] * 1.5  # energy consumption 50% up for 2050



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

# Define a function to disaggregate based on 2023 data
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

    # Calculate the total 2023 values for the categories
    total_2023 = filtered_merged_df[[f"{cat}_2023" for cat in categories]].fillna(0).sum(axis=1).values[0]

    # Handle cases where total_2023 is zero
    if total_2023 == 0:
        print(f"Total 2023 for country code {country_code} is 0")
        return {cat: 0 for cat in categories}

    # Calculate proportions for each category based on 2023 data
    proportions = {
        cat: filtered_merged_df[f"{cat}_2023"].fillna(0).values[0] / total_2023
        for cat in categories
    }

    # Log the calculated proportions for debugging purposes
    print(f"Proportions for country code {country_code}: {proportions}")
    
    # Disaggregate the values in the specified column based on the calculated proportions
    return {cat: row[column_name] * proportions[cat] for cat in categories}

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

    # Log the calculated proportions for debugging purposes
    import logging
    logging.debug(f"Proportions for country code {country_code}: {proportions}")

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

# Rename columns for 2030 targets
df3.rename(columns={
    "Hydro": "Hydro_2030",
    "Other Renewables": "Other Renewables_2030",
    "Solar": "Solar_2030",
    "Wind": "Wind_2030"
}, inplace=True)

# Ensure the column name is correct for merging
df3.rename(columns={"Country code": "country_code"}, inplace=True)

# Merge the third dataset with the previously merged dataset
final_merged_df = pd.merge(merged_df, df3, left_on="ISO3_code", right_on="country_code", how="left")

# Ensure extrapolated columns exist before using them
for col in ["Solar", "Wind", "Hydro", "Other Renewables"]:
    extrapolated_col = f"{col}_2030_extrapolated"
    if extrapolated_col not in final_merged_df.columns:
        print(f"Warning: Column '{extrapolated_col}' not found in final_merged_df. Skipping extrapolation for '{col}_2030'.")
        continue
    final_merged_df[f"{col}_2030"] = final_merged_df[f"{col}_2030"].fillna(final_merged_df[extrapolated_col])

# Drop the extrapolated columns after use
final_merged_df.drop(columns=[f"{col}_2030_extrapolated" for col in ["Solar", "Wind", "Hydro", "Other Renewables"] if f"{col}_2030_extrapolated" in final_merged_df.columns], inplace=True)

# Save the final merged dataset to a CSV file
final_merged_df.to_excel("outputs_processed_data/p1_a_ember_2023_30.xlsx", index=False)