import pandas as pd

# Load the first dataset
file_path1 = "un_pop/WPP2024_TotalPopulationBySex2025-05-02.csv" # updated 5/2/2025 
# https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format

df1 = pd.read_csv(file_path1, low_memory=False)

# Filter the first dataset for "Medium" variant and years 2023, 2030, and 2050
filtered_df1 = df1[(df1["Variant"] == "Medium") & (df1["Time"].isin([2023, 2030, 2050]))]

# Select required columns from the first dataset
selected_columns1 = ["ISO3_code", "ISO2_code", "Time", "PopTotal"]
filtered_df1 = filtered_df1[selected_columns1]

# Remove rows with missing ISO codes from the first dataset
filtered_df1 = filtered_df1.dropna(subset=["ISO3_code", "ISO2_code"])

# Pivot the first dataset to have years as columns
pivot_df1 = filtered_df1.pivot(index=["ISO3_code", "ISO2_code"], columns="Time", values="PopTotal").reset_index()

# Rename columns for clarity in the first dataset
pivot_df1.columns = ["ISO3_code", "ISO2_code", "PopTotal_2023", "PopTotal_2030", "PopTotal_2050"]

# Calculate the population growth factor from 2023 to 2030 and 2050
pivot_df1["Growth_Factor_2023_2030"] = pivot_df1["PopTotal_2030"] / pivot_df1["PopTotal_2023"]
pivot_df1["Growth_Factor_2023_2050"] = pivot_df1["PopTotal_2050"] / pivot_df1["PopTotal_2023"]



# Load the second dataset
file_path2 = "ember_energy_data/yearly_full_release_long_format2025-05-02.csv" # updated 5/2/2025
df2 = pd.read_csv(file_path2)

# Filter the second dataset for years 2023 through 2019
years = [2023, 2022, 2021, 2020, 2019]
filtered_df2 = df2[
    (df2["Year"].isin(years)) &
    (df2["Category"] == "Capacity") &
    (df2["Subcategory"] == "Fuel") &
    (df2["Unit"] == "GW") &
    (df2["Variable"].isin(["Bioenergy", "Coal", "Gas", "Hydro", "Nuclear", "Other Fossil", 
                           "Other Renewables", "Solar", "Wind"]))
]

# Prioritize the most recent year for each country and variable
filtered_df2 = filtered_df2.sort_values(by="Year", ascending=False).drop_duplicates(subset=["Country code", "Variable"], keep='first')

# Group "Coal", "Gas", "Other Fossil" into "Fossil" in the second dataset
filtered_df2.loc[filtered_df2["Variable"].isin(["Coal", "Gas", "Other Fossil"]), "Variable"] = "Fossil"

# Merge "Bioenergy" with "Other Renewables" in the second dataset
filtered_df2.loc[filtered_df2["Variable"] == "Bioenergy", "Variable"] = "Other Renewables"

# Remove rows where "Country code" is NaN in the second dataset
filtered_df2 = filtered_df2.dropna(subset=["Country code"])

# Aggregate duplicate entries by summing their values in the second dataset
filtered_df2 = filtered_df2.groupby(["Country code", "Variable"], as_index=False)["Value"].sum()

# Pivot the cleaned second dataset
pivot_df2 = filtered_df2.pivot(index="Country code", columns="Variable", values="Value").reset_index()

# Fill NaN values with zero
pivot_df2 = pivot_df2.fillna(0)

# Rename columns by adding "_2023" except for "Country code" in the second dataset
pivot_df2.columns = ["Country code"] + [f"{col}_2023" for col in pivot_df2.columns[1:]]

# Merge the two datasets by "Country code"
merged_df = pd.merge(pivot_df1, pivot_df2, left_on="ISO3_code", right_on="Country code", how="inner")

# Extrapolate the 2023 data to 2030/2050 using the population growth factors
for col in pivot_df2.columns[1:]:
    merged_df[f"{col.replace('_2023', '')}_2030_extrapolated"] = merged_df[col] * merged_df["Growth_Factor_2023_2030"]
for col in pivot_df2.columns[1:]:
    merged_df[f"{col.replace('_2023', '')}_2050_extrapolated"] = merged_df[col] * merged_df["Growth_Factor_2023_2050"]


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

# Print the unique country codes in df3
print("Unique country codes in df3:", df3["country_code"].unique())

# Print the unique country codes in merged_df
print("Unique country codes in merged_df:", merged_df["ISO3_code"].unique())

# Define a function to disaggregate based on 2023 data
def disaggregate(row, merged_df, categories, column_name):
    country_code = row["country_code"]
    if country_code not in merged_df["ISO3_code"].values:
        print(f"Country code {country_code} not found in merged_df")
        return {cat: 0 for cat in categories}
    total_2023 = sum(merged_df.loc[merged_df["ISO3_code"] == country_code, f"{cat}_2023"].fillna(0).values[0] for cat in categories)
    if total_2023 == 0:
        print(f"Total 2023 for country code {country_code} is 0")
        return {cat: 0 for cat in categories}
    proportions = {cat: merged_df.loc[merged_df["ISO3_code"] == country_code, f"{cat}_2023"].fillna(0).values[0] / total_2023 for cat in categories}
    print(f"Proportions for country code {country_code}: {proportions}")
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

# Use extrapolated 2030 data for countries without 2030 data
for col in ["Solar", "Wind", "Hydro", "Other Renewables"]:
    final_merged_df[f"{col}_2030"] = final_merged_df[f"{col}_2030"].fillna(final_merged_df[f"{col}_2030_extrapolated"])

# Drop the extrapolated columns
final_merged_df.drop(columns=[f"{col}_2030_extrapolated" for col in ["Solar", "Wind", "Hydro", "Other Renewables"]], inplace=True)

# Save the final merged dataset to a CSV file
save_path_final = "outputs_processed_data/p1_a_ember_2023_30.csv"
final_merged_df.to_csv(save_path_final, index=False)