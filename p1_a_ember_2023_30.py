# 6/30/2025 at work

import pandas as pd
import itertools

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
granular_columns = (
    ["Country Name", "Country Code"] +
    [col for col in grouped_df.columns if "_Larger_MW" in col] +
    [col for col in grouped_df.columns if "_Potential_MWh" in col]
)

grouped_df = grouped_df[granular_columns]

# Rename columns for clarity
grouped_df.rename(columns={"Country Code": "ISO3_code"}, inplace=True)
grouped_df.rename(columns={"Total_Potential_MWh": "Total_MWh_2023"}, inplace=True)

# Rename columns to follow the {type}_2023 naming convention
grouped_df.rename(columns={col: col.replace("_Larger_MW", "_2023") for col in grouped_df.columns if "_Larger_MW" in col}, inplace=True)
grouped_df.rename(columns={col: col.replace("_Potential_MWh", "_2023_MWh") for col in grouped_df.columns if "_Potential_MWh" in col}, inplace=True)


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
        
        # Calculate MWh for each energy type
        hydro_mwh = row["Hydro_2030"] * conversion_factors["Hydro_ConvRate"]
        solar_mwh = row["Solar_2030"] * conversion_factors["Solar_ConvRate"]
        wind_mwh = row["Wind_2030"] * conversion_factors["Wind_ConvRate"]
        other_renewables_mwh = row["Other Renewables_2030"] * conversion_factors["Other Renewables_ConvRate"]
        
        # Assign individual MWh values to the dataframe
        final_merged_df.at[index, "Hydro_2030_MWh"] = hydro_mwh
        final_merged_df.at[index, "Solar_2030_MWh"] = solar_mwh
        final_merged_df.at[index, "Wind_2030_MWh"] = wind_mwh
        final_merged_df.at[index, "Other Renewables_2030_MWh"] = other_renewables_mwh
        
        # Calculate the total MWh for 2030
        total_mwh = hydro_mwh + solar_mwh + wind_mwh + other_renewables_mwh
        final_merged_df.at[index, "HSWO_2030_MWh"] = total_mwh

# Compute total RE generation per capita for 2030
final_merged_df["Hydro_2030_MWh_per_capita"] = final_merged_df["Hydro_2030_MWh"] / final_merged_df["PopTotal_2030"]
final_merged_df["Solar_2030_MWh_per_capita"] = final_merged_df["Solar_2030_MWh"] / final_merged_df["PopTotal_2030"]
final_merged_df["Wind_2030_MWh_per_capita"] = final_merged_df["Wind_2030_MWh"] / final_merged_df["PopTotal_2030"]
final_merged_df["Other Renewables_2030_MWh_per_capita"] = final_merged_df["Other Renewables_2030_MWh"] / final_merged_df["PopTotal_2030"]
final_merged_df["HSWO_2030_MWh_per_capita"] = final_merged_df["HSWO_2030_MWh"] / final_merged_df["PopTotal_2030"]

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

# Fill 2030 Capacity Percentage targets for countries without values in the "res_share_target" column

# Calculate regional and income group benchmarks for "res_share_target"
res_share_benchmarks = (
    final_merged_df[final_merged_df["res_share_target"].notnull()]
    .groupby(["Region", "Income group"])["res_share_target"]
    .mean()
    .reset_index()
)

# Compute global mean for "res_share_target"
global_res_share_mean = final_merged_df["res_share_target"].mean()

# Merge regional benchmarks back into the dataset
final_merged_df = pd.merge(
    final_merged_df,
    res_share_benchmarks,
    on=["Region", "Income group"],
    how="left",
    suffixes=("", "_R_Benchmark")
)

# Fill missing "res_share_target" values
final_merged_df["res_share_target"] = final_merged_df["res_share_target"].fillna(
    final_merged_df["res_share_target_R_Benchmark"]
).fillna(global_res_share_mean)

# Calculate and fill MWh per capita for 2030
# Define the energy types to be used for regional benchmarks
energy_types = ["Hydro_2030_MWh_per_capita", "Solar_2030_MWh_per_capita", "Wind_2030_MWh_per_capita", "Other Renewables_2030_MWh_per_capita"]

# Filter the dataset to include only countries with valid NDCs (non-null "Category")
# Filter only countries with valid NDCs
valid_ndc_df = final_merged_df[final_merged_df["Category"].notnull()]

# Melt the dataframe for per capita calculations
melted_df = valid_ndc_df.melt(
    id_vars=["Region", "Income group"],
    value_vars=energy_types,
    var_name="Energy_Type",
    value_name="MWh_Per_Capita"
)
# Calculate regional benchmarks (mean per capita by Region, Income group, and Energy Type)
regional_benchmarks = (
    melted_df
    .groupby(["Region", "Income group", "Energy_Type"])["MWh_Per_Capita"]
    .mean()
    .reset_index()
)
# Create a full index of all combinations
regions = valid_ndc_df["Region"].dropna().unique()
income_groups = valid_ndc_df["Income group"].dropna().unique()
energy_types_unique = melted_df["Energy_Type"].dropna().unique()

full_index = pd.DataFrame(
    list(itertools.product(regions, income_groups, energy_types_unique)),
    columns=["Region", "Income group", "Energy_Type"]
)

# Merge full index with actual data
regional_benchmarks = full_index.merge(
    regional_benchmarks,
    on=["Region", "Income group", "Energy_Type"],
    how="left"
)

# Compute global means by Income group and Energy Type
global_means_by_income = (
    valid_ndc_df
    .melt(
        id_vars=["Income group"],
        value_vars=energy_types,
        var_name="Energy_Type",
        value_name="MWh_Per_Capita"
    )
    .groupby(["Income group", "Energy_Type"])["MWh_Per_Capita"]
    .mean()
    .to_dict()
)

# Fill missing or zero values with global means
regional_benchmarks["MWh_Per_Capita"] = regional_benchmarks.apply(
    lambda row: global_means_by_income.get((row["Income group"], row["Energy_Type"]), 0)
    if pd.isna(row["MWh_Per_Capita"]) or row["MWh_Per_Capita"] == 0 else row["MWh_Per_Capita"],
    axis=1
)

# Pivot the table to wide format
regional_benchmarks_pivot = regional_benchmarks.pivot_table(
    index=["Region", "Income group"],
    columns="Energy_Type",
    values="MWh_Per_Capita",
    aggfunc="mean"
).reset_index()

# Rename columns to indicate these are regional benchmarks
regional_benchmarks_pivot.columns = [
    "Region", "Income group"
] + [f"{col}_R_Benchmark" for col in regional_benchmarks_pivot.columns[2:]]

# Merge regional benchmarks back into the dataset
final_merged_df = pd.merge(
    final_merged_df,
    regional_benchmarks_pivot,
    on=["Region", "Income group"],
    how="left"
)

# Apply regional benchmarks to countries without 2030 targets
for energy_type in energy_types:
    benchmark_col = energy_type + "_R_Benchmark"
    final_merged_df.loc[final_merged_df["Category"].isnull(), energy_type] = (
        final_merged_df.loc[final_merged_df["Category"].isnull(), benchmark_col]
    ).fillna(0)  # Fill any missing values with 0

### Task 6
# Fill in missing generation data (MWh) for 2030 using regional benchmarks:
# 1. Calculate missing generation for Hydro_2030, Solar_2030, Wind_2030, and Other Renewables_2030.
#    This is done by multiplying the regional benchmark per capita (MWh per capita) * the 2030 population 
#    and dividing by the conversion factor for each energy type.
# 2. Adjust the generation of each type for 2030 proportionally to meet the following formula:
#   post-calculation total renewable generations / post-calculation total generation = res_share_target / 100

# Step 1: Calculate missing generation
# For each renewable energy type (Hydro, Solar, Wind, Other Renewables), calculate the missing generation (MWh) for 2030.
# This is done for countries without valid NDCs (i.e., where "Category" is null).
# The calculation uses the regional benchmark per capita (MWh per capita) multiplied by the 2030 population.
energy_types2 = [
    "Hydro_2030_MWh", "Solar_2030_MWh",
    "Wind_2030_MWh", "Other Renewables_2030_MWh"
]

for energy_type2 in energy_types2:
    # Get the corresponding regional benchmark column for MWh per capita
    benchmark_col = energy_type2 + "_per_capita_R_Benchmark"
    # Calculate the missing generation for countries without valid NDCs
    final_merged_df.loc[final_merged_df['Category'].isnull(), energy_type2] = (
        final_merged_df.loc[final_merged_df['Category'].isnull(), benchmark_col]
        * final_merged_df.loc[final_merged_df['Category'].isnull(), "PopTotal_2030"]
    ).fillna(0)  # Fill any remaining missing values with 0

# Step 2a: Adjust the generation to match the res_share_target - Based on NDCs extrapolation
# For countries without valid NDCs, ensure renewable generation values are at least equal to 2023 values,
# then apply regional benchmarks, and finally adjust proportions to meet the target renewable share (res_share_target).

for index, row in final_merged_df.iterrows():
    if pd.isnull(row["Category"]):  # Only adjust for countries without valid NDCs
        fossil = row.get("Fossil_2023_MWh", 0)
        nuclear = row.get("Nuclear_2023_MWh", 0)
        target_share = row.get("res_share_target", 0) / 100 if pd.notnull(row.get("res_share_target", 0)) else 0
        renewables_2023 = {energy_type: row.get(energy_type.replace("_2030_MWh", "_2023_MWh"), 0) for energy_type in energy_types2}
        renewables_2030 = {energy_type: row.get(energy_type, 0) for energy_type in energy_types2}
        renewables_sum = sum(renewables_2030.values())
        total_gen_2030 = renewables_sum + fossil + nuclear
        if total_gen_2030 > 0 and target_share > 0 and renewables_sum > 0:
            if fossil + nuclear == 0:
                desired_renewable_total = renewables_sum
            else:
                desired_renewable_total = target_share * (fossil + nuclear) / (1 - target_share)
            # Initial scaling
            scaling_factor = desired_renewable_total / renewables_sum
            scaled = {k: v * scaling_factor for k, v in renewables_2030.items()}
            # Enforce minimum = 2023 value
            fixed = {k: max(scaled[k], renewables_2023[k]) for k in energy_types2}
            # If sum is now above target, re-scale only those not fixed at 2023 value
            fixed_sum = sum(v for k, v in fixed.items() if v == renewables_2023[k])
            to_scale = [k for k in energy_types2 if fixed[k] > renewables_2023[k]]
            if to_scale:
                remaining = desired_renewable_total - fixed_sum
                if remaining < 0:
                    # If fixed_sum already exceeds target, set all to 2023 value
                    for k in energy_types2:
                        final_merged_df.at[index, k] = renewables_2023[k]
                else:
                    scale_sum = sum(fixed[k] for k in to_scale)
                    if scale_sum > 0:
                        rescale_factor = remaining / scale_sum
                        for k in to_scale:
                            final_merged_df.at[index, k] = fixed[k] * rescale_factor
                    for k in energy_types2:
                        if fixed[k] == renewables_2023[k]:
                            final_merged_df.at[index, k] = renewables_2023[k]
            else:
                for k in energy_types2:
                    final_merged_df.at[index, k] = fixed[k]
        else:
            # If not enough data, just copy 2030 values as fallback
            for k in energy_types2:
                final_merged_df.at[index, k] = renewables_2030[k]

# Step 2b: Adjust the generation to match the res_share_target - Based on Estimates of IEA tracker
# (https://www.iea.org/data-and-statistics/data-tools/renewable-energy-progress-tracker)
# For countries without valid NDCs, ensure renewable generation values are at least equal to 2023 values,
# then apply regional benchmarks, and finally adjust proportions to meet the target renewable share (IEA_targets).

IEA_targets = {
    "High income": 0.49,
    "Upper middle income": 0.44,
    "Lower middle income": 0.44,
    "Low income": 0.44
}

for index, row in final_merged_df.iterrows():
    if pd.isnull(row["Category"]):  # Only adjust for countries without valid NDCs
        income_group = row.get("Income group", None)
        iea_target_share = IEA_targets.get(income_group, 0.44)  # Default to 44% if not found
        fossil = row.get("Fossil_2023_MWh", 0)
        nuclear = row.get("Nuclear_2023_MWh", 0)
        renewables_2023 = {energy_type: row.get(energy_type.replace("_2030_MWh", "_2023_MWh"), 0) for energy_type in energy_types2}
        renewables_2030 = {energy_type: row.get(energy_type, 0) for energy_type in energy_types2}
        renewables_sum = sum(renewables_2030.values())
        total_gen_2030 = renewables_sum + fossil + nuclear
        if total_gen_2030 > 0 and iea_target_share > 0 and renewables_sum > 0:
            if fossil + nuclear == 0:
                desired_renewable_total = renewables_sum
            else:
                desired_renewable_total = iea_target_share * (fossil + nuclear) / (1 - iea_target_share)
            # Initial scaling
            scaling_factor = desired_renewable_total / renewables_sum
            scaled = {k: v * scaling_factor for k, v in renewables_2030.items()}
            # Enforce minimum = 2023 value
            fixed = {k: max(scaled[k], renewables_2023[k]) for k in energy_types2}
            # If sum is now above target, re-scale only those not fixed at 2023 value
            fixed_sum = sum(v for k, v in fixed.items() if v == renewables_2023[k])
            to_scale = [k for k in energy_types2 if fixed[k] > renewables_2023[k]]
            if to_scale:
                remaining = desired_renewable_total - fixed_sum
                if remaining < 0:
                    # If fixed_sum already exceeds target, set all to 2023 value
                    for k in energy_types2:
                        final_merged_df.at[index, k + '_IEA'] = renewables_2023[k]
                else:
                    scale_sum = sum(fixed[k] for k in to_scale)
                    if scale_sum > 0:
                        rescale_factor = remaining / scale_sum
                        for k in to_scale:
                            final_merged_df.at[index, k + '_IEA'] = fixed[k] * rescale_factor
                    for k in energy_types2:
                        if fixed[k] == renewables_2023[k]:
                            final_merged_df.at[index, k + '_IEA'] = renewables_2023[k]
            else:
                for k in energy_types2:
                    final_merged_df.at[index, k + '_IEA'] = fixed[k]
        else:
            # If not enough data, just copy 2030 values as fallback
            for k in energy_types2:
                final_merged_df.at[index, k + '_IEA'] = renewables_2030[k]


### Task 7
# Fill in missing Capacity (MW) data for 2030 based on [MWh / conversion factor]
energy_types3 = ["Hydro_2030", "Solar_2030", "Wind_2030", "Other Renewables_2030"]
for energy_type_mw in energy_types3:
    energy_type_mwh = energy_type_mw + "_MWh"
    conversion_factor_col = energy_type_mw.replace("_2030", "_ConvRate")
    # Calculate missing capacity (MW) for 2030 using conversion factors
    final_merged_df.loc[final_merged_df["Category"].isnull(), energy_type_mw] = (
        final_merged_df.loc[final_merged_df["Category"].isnull(), energy_type_mwh]
        / final_merged_df["ISO3_code"].map(lambda code: capacity_conversion_factors.get(code, {}).get(conversion_factor_col, global_mean_conv_rates.get(conversion_factor_col, 0)))
    ).fillna(0)  # Fill any remaining missing values with 0

### Task Last
# Clean up the column order in the final merged dataset
column_order = [
    "Country Name", "ISO3_code", "Region", "Income group", 
    "PopTotal_2023", "PopTotal_2030", "PopTotal_2050", "Growth_Factor_2023_2030", "Growth_Factor_2023_2050",
    "Total_MWh_2023", "Total_MWh_2030", "Total_MWh_2050",
    "Hydro_2023", "Solar_2023", "Wind_2023", "Other Renewables_2023", "Nuclear_2023", "Fossil_2023",
    "Hydro_2023_MWh", "Solar_2023_MWh", "Wind_2023_MWh", "Other Renewables_2023_MWh","Nuclear_2023_MWh", "Fossil_2023_MWh",
    "res_capacity_target", "res_share_target", "Category",
    "Hydro_2030", "Solar_2030", "Wind_2030", "Other Renewables_2030",
    "Hydro_2030_MWh", "Solar_2030_MWh", "Wind_2030_MWh", "Other Renewables_2030_MWh",
    "Hydro_2030_MWh_IEA", "Solar_2030_MWh_IEA", "Wind_2030_MWh_IEA", "Other Renewables_2030_MWh_IEA",
    ]
final_merged_df = final_merged_df[column_order]

print("Final number of countries: ", final_merged_df["ISO3_code"].nunique())

# Save the final merged dataset to a CSV file
final_merged_df.to_excel("outputs_processed_data/p1_a_ember_2023_30d.xlsx", index=False)

